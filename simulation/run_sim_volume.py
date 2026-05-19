"""Multi-slice driver: run :func:`run_all_qmri_simulations` over every slice
of a phantom and stack the per-pipeline results.

Typical usage
-------------
    from run_sim_volume import run_all_qmri_simulations_volume
    from utils_sim_lib import PathConfig

    paths = PathConfig(...)
    results = run_all_qmri_simulations_volume(
        paths,
        slice_indices=None,          # None → every slice in the phantom
        b_values=range(0, 2001, 100),
    )
    # results["adc_sse"]["adc_nlls"]            (n_slices, Ny, Nx)
    # results["adc_sse"]["mae_per_tissue"]      {"wm": (n_slices,), ...}
    # results["adc_sse"]["mae_total"]           (n_slices,)
    # results["adc_sse"]["slice_indices"]       (n_slices,)
"""
from __future__ import annotations

import os
import time
from typing import Iterable, Optional, Sequence

import nibabel as nib
import numpy as np

from utils_sim_lib import PathConfig, PreloadedPhantom, preload_phantom_for_sim, probe_phantom_n_slices
from run_sim import (
    DEFAULT_ADC_BVALUES,
    DEFAULT_T2_TES,
    DEFAULT_T2_TRIPLE_TE1,
    PIPELINES,
    run_all_qmri_simulations,
)


def _fmt(seconds: float) -> str:
    """Format a duration as 'Xm Ys' (≥60 s) or 'X.Xs' (< 60 s)."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


# Per-pipeline list of map-like keys (2-D arrays) to stack across slices.
# Anything not listed is collected as a Python list (preserves order).
_MAP_KEYS = {
    "adc_sse": ("adc_nlls", "fa_map", "md_map", "reference_map"),
    "t2_sse": ("t2_nlls", "reference_map"),
    "adc_multishot": ("adc_nlls", "fa_map", "md_map", "reference_map"),
    "t2_multishot": ("t2_nlls", "reference_map"),
    "adc_triple": ("adc_nlls", "fa_map", "md_map", "reference_map"),
    "t2_triple": ("t2_nlls", "reference_map"),
}


def _save_volume_nifti(vol: np.ndarray, path: str, res_mm: float) -> None:
    """Save an (Nx, Ny, Nz) float32 volume as NIfTI with an isotropic diagonal affine."""
    affine = np.diag([res_mm, res_mm, res_mm, 1.0]).astype(np.float64)
    nib.save(nib.Nifti1Image(vol.astype(np.float32), affine), path)



def run_all_qmri_simulations_volume(
    paths: PathConfig,
    *,
    slice_indices: Optional[Sequence[int]] = None,
    fov: float = 224e-3,
    res: float = 2.33333333,
    system_type=None,
    b_values: Sequence[int] = DEFAULT_ADC_BVALUES,
    TEs_t2: Sequence[int] = DEFAULT_T2_TES,
    TE1_values_triple: Sequence[int] = DEFAULT_T2_TRIPLE_TE1,
    TE_adc: int = 100,
    TR: int = 5000,
    b_directions: int = 6,
    small_delta: Optional[float] = None,
    big_DELTA: Optional[float] = None,
    small_delta_t2: Optional[float] = None,
    big_DELTA_t2: Optional[float] = None,
    ETL: int = 1,
    blip_down: bool = True,
    use_gpu: Optional[bool] = None,
    pipelines: Optional[Iterable[str]] = None,
    save_volume_nifti: bool = True,
) -> dict:
    """Run every requested pipeline on every slice of the phantom.

    Parameters
    ----------
    slice_indices:
        List of slice indices to run. ``None`` → every slice in the phantom
        (auto-probed via :func:`probe_phantom_n_slices`).
    save_volume_nifti:
        When ``True`` (default), save the stacked NLLS maps for each pipeline
        as ``<phantom_name>-<MAP>_<pipeline>_volume.nii.gz`` in ``paths.volumes_dir``.
    See :func:`run_sim.run_all_qmri_simulations` for all other params.

    Returns
    -------
    dict keyed by pipeline name. Each value is a dict containing:
        - 2-D maps stacked along the slice axis (e.g. ``adc_nlls`` of shape
          ``(n_slices, Ny, Nx)``)
        - ``mae_per_tissue``: ``{tissue_name: (n_slices,) array}``
        - ``mae_total``: ``(n_slices,)`` array
        - ``mae_per_tissue_volume`` / ``mae_total_volume``: voxel-weighted
          mean across all slices (single scalar each)
        - ``tissue_masks``: ``{tissue_name: (n_slices, Ny, Nx) bool array}``
        - ``slice_indices``: the indices that were actually executed
    Plus a top-level ``"slice_indices"`` key with the same vector.
    """
    if pipelines is None:
        pipelines = PIPELINES
    pipelines = tuple(pipelines)

    if slice_indices is None:
        n_slices = probe_phantom_n_slices(
            paths, fov_mm=fov * 1e3, resolution_mm=res
        )
        slice_indices = list(range(n_slices))
    slice_indices = list(slice_indices)
    print(
        f"[volume] running {len(slice_indices)} slices "
        f"(idx {slice_indices[0]}..{slice_indices[-1]}) "
        f"× {len(pipelines)} pipelines",
        flush=True,
    )

    # Load and preprocess phantom once for the entire volume run.
    preloaded = preload_phantom_for_sim(paths, fov_m=fov, resolution_mm=res)

    # Save ground-truth reference volumes before any simulation runs.
    # Rotate each axial (Ny,Nx) slice 90° CW — matching the SSE per-slice convention.
    _d_ref  = np.rot90(preloaded.phantom.D.cpu().numpy(),  k=-1, axes=(0, 1))
    _t2_ref = np.rot90(preloaded.phantom.T2.cpu().numpy(), k=-1, axes=(0, 1))
    _save_volume_nifti(_d_ref,  os.path.join(paths.volumes_dir, f"{paths.phantom_name}-D_ref_volume.nii.gz"),  res)
    _save_volume_nifti(_t2_ref, os.path.join(paths.volumes_dir, f"{paths.phantom_name}-T2_ref_volume.nii.gz"), res)
    print(f"[volume] saved D_ref_volume and T2_ref_volume  shape={_d_ref.shape}", flush=True)

    _t_adc_total = 0.0
    _t_t2_total  = 0.0
    _t_volume_start = time.perf_counter()

    _Ny = _Nx = round(fov * 1e3 / res)
    _n_slices = len(slice_indices)

    # Pre-allocate output volumes (NaN-init so skipped slices stay as NaN).
    vol: dict[str, dict] = {}
    for _p in pipelines:
        _d: dict = {}
        for _key in _MAP_KEYS.get(_p, ()):
            _d[_key] = np.full((_n_slices, _Ny, _Nx), np.nan, dtype=np.float32)
        _d["mae_total"]          = np.full(_n_slices, np.nan, dtype=float)
        _d["mae_n_voxels_total"] = np.zeros(_n_slices, dtype=int)
        _d["mae_per_tissue"]          = {t: np.full(_n_slices, np.nan, dtype=float) for t in preloaded.tissue_names}
        _d["mae_n_voxels_per_tissue"] = {t: np.zeros(_n_slices, dtype=int)          for t in preloaded.tissue_names}
        _d["tissue_masks"]            = {t: np.zeros((_n_slices, _Ny, _Nx), dtype=bool) for t in preloaded.tissue_names}
        vol[_p] = _d

    for s_idx, slice_idx in enumerate(slice_indices):
        print(
            f"\n========= slice {s_idx + 1}/{len(slice_indices)} "
            f"(index {slice_idx}) =========",
            flush=True,
        )

        # Skip slices where the phantom has no tissue — mr0.compute_graph() panics
        # with a Rust NaN error when PD is all-zero.
        _pd_max = float(preloaded.phantom.PD[:, :, slice_idx].max())
        if _pd_max < 1e-9:
            print(f"[volume] slice {slice_idx} empty (PD_max={_pd_max:.1e}), filling NaN", flush=True)
            continue

        slice_res = run_all_qmri_simulations(
            paths,
            slice_idx=slice_idx,
            system_type=system_type,
            fov=fov,
            res=res,
            b_values=b_values,
            TEs_t2=TEs_t2,
            TE1_values_triple=TE1_values_triple,
            TE_adc=TE_adc,
            TR=TR,
            b_directions=b_directions,
            small_delta=small_delta,
            big_DELTA=big_DELTA,
            small_delta_t2=small_delta_t2,
            big_DELTA_t2=big_DELTA_t2,
            ETL=ETL,
            blip_down=blip_down,
            use_gpu=use_gpu,
            pipelines=pipelines,
            save_slice_npy=False,
            preloaded_phantom=preloaded,
        )
        for p in pipelines:
            pr = slice_res[p]
            for key in _MAP_KEYS.get(p, ()):
                vol[p][key][s_idx] = pr[key]
            vol[p]["mae_total"][s_idx]          = pr["mae_total"]
            vol[p]["mae_n_voxels_total"][s_idx] = pr["mae_n_voxels_total"]
            for name in preloaded.tissue_names:
                vol[p]["mae_per_tissue"][name][s_idx]          = pr["mae_per_tissue"][name]
                vol[p]["mae_n_voxels_per_tissue"][name][s_idx] = pr["mae_n_voxels_per_tissue"][name]
                _tm = pr["tissue_masks"][name]
                vol[p]["tissue_masks"][name][s_idx] = _tm.cpu().numpy() if hasattr(_tm, "cpu") else np.asarray(_tm)
            if "_metadata_captured" not in vol[p]:
                for key in ("b_values", "TEs", "echo_TEs_ms"):
                    if key in pr:
                        vol[p][key] = pr[key]
                vol[p]["_metadata_captured"] = True

        # Per-slice timing breakdown
        _slice_timings = slice_res.get("_timings", {})
        _t_adc_slice = sum(v for k, v in _slice_timings.items() if k.startswith("adc"))
        _t_t2_slice  = sum(v for k, v in _slice_timings.items() if k.startswith("t2"))
        _t_adc_total += _t_adc_slice
        _t_t2_total  += _t_t2_slice
        _elapsed = time.perf_counter() - _t_volume_start
        print(
            f"[time] slice {s_idx + 1}/{len(slice_indices)}  "
            f"ADC={_fmt(_t_adc_slice)}  T2={_fmt(_t_t2_slice)}  "
            f"slice total={_fmt(_t_adc_slice + _t_t2_slice)}",
            flush=True,
        )
        print(
            f"[time] cumulative  "
            f"ADC={_fmt(_t_adc_total)}  T2={_fmt(_t_t2_total)}  "
            f"elapsed={_fmt(_elapsed)}",
            flush=True,
        )

        # Incremental save: slice into pre-allocated array — no re-stacking needed.
        if save_volume_nifti:
            for p in pipelines:
                map_key = "adc_nlls" if p.startswith("adc") else "t2_nlls"
                if map_key in vol[p]:
                    partial = vol[p][map_key][:s_idx + 1]
                    fname = f"{paths.phantom_name}-{map_key.upper()}_{p}_volume.nii.gz"
                    _m = np.rot90(partial, k=-1, axes=(1, 2))
                    _m = np.transpose(_m, (1, 2, 0))
                    _save_volume_nifti(_m, os.path.join(paths.volumes_dir, fname), res)
            print(
                f"[volume] updated volume NIfTI files "
                f"({s_idx + 1}/{len(slice_indices)} slices done)",
                flush=True,
            )

    slice_idx_arr = np.asarray(slice_indices, dtype=int)
    np.save(os.path.join(paths.volumes_dir, f"{paths.phantom_name}-slice_indices.npy"), slice_idx_arr)

    # Compute volume-level MAE from pre-allocated arrays and finalise output.
    out: dict = {"slice_indices": slice_idx_arr}
    for p in pipelines:
        n_tot = vol[p]["mae_n_voxels_total"]
        m_tot = vol[p]["mae_total"]
        valid = np.isfinite(m_tot) & (n_tot > 0)
        vol[p]["mae_total_volume"] = (
            float(np.sum(m_tot[valid] * n_tot[valid]) / n_tot[valid].sum())
            if valid.any() else float("nan")
        )
        vol[p]["mae_per_tissue_volume"] = {}
        for name in preloaded.tissue_names:
            m = vol[p]["mae_per_tissue"][name]
            n = vol[p]["mae_n_voxels_per_tissue"][name]
            v = np.isfinite(m) & (n > 0)
            vol[p]["mae_per_tissue_volume"][name] = (
                float(np.sum(m[v] * n[v]) / n[v].sum()) if v.any() else float("nan")
            )
        vol[p].pop("_metadata_captured", None)
        vol[p]["slice_indices"] = slice_idx_arr
        out[p] = vol[p]

    _t_volume_total = time.perf_counter() - _t_volume_start
    print("\n========== Volume timing summary ==========", flush=True)
    print(f"  ADC total  : {_fmt(_t_adc_total)}", flush=True)
    print(f"  T2 total   : {_fmt(_t_t2_total)}", flush=True)
    print(f"  Volume     : {_fmt(_t_volume_total)}", flush=True)

    # Print a compact MAE summary
    print("\n========== Volume MAE summary ==========", flush=True)
    for p in pipelines:
        s = out[p]
        per_t = ", ".join(
            f"{k}={v:.4f}" for k, v in s["mae_per_tissue_volume"].items()
        )
        print(f"  {p:<14} total={s['mae_total_volume']:.4f}  [{per_t}]", flush=True)

    return out


if __name__ == "__main__":
    import os

    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    paths = PathConfig(
        seq_lib_path  = os.path.join(_ROOT, "pulseq_diffusion_mese"),
        phantoms_dir  = os.path.join(_ROOT, "brainweb_phantoms"),
        phantom_name  = "brainweb-subj04",
        sequences_dir = os.path.join(_ROOT, "simulation", "simulated", "seq"),
        volumes_dir   = os.path.join(_ROOT, "simulation", "simulated", "volumes"),
        masks_dir     = os.path.join(_ROOT, "simulation", "simulated", "masks"),
        diff_img_dir  = os.path.join(_ROOT, "simulation", "simulated", "diff_img"),
        t2_img_dir    = os.path.join(_ROOT, "simulation", "simulated", "t2_img"),
    )
    
    pipelines = ("adc_triple", "t2_triple", "adc_multishot", "t2_multishot", "adc_sse", "t2_sse")  
    results = run_all_qmri_simulations_volume(
        paths,
        slice_indices=None,                # None → all slices; pass e.g. [55, 60, 65] to limit
        pipelines=pipelines,   # remove kwarg to run all 6 pipelines
        save_volume_nifti=True,
    )

    for pipe in pipelines:
        s = results[pipe]
        per_t = ", ".join(f"{k}={v:.4f}" for k, v in s["mae_per_tissue_volume"].items())
        print(f"  {pipe:<14}  total={s['mae_total_volume']:.4f}  [{per_t}]")
        map_arr = s.get("adc_nlls", s.get("t2_nlls"))
        if map_arr is not None:
            print(f"  map shape      : {map_arr.shape}")