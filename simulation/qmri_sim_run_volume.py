"""Multi-slice driver: run :func:`run_all_qmri_simulations` over every slice
of a phantom and stack the per-pipeline results.

Typical usage
-------------
    from qmri_sim_run_volume import run_all_qmri_simulations_volume
    from qmri_sim_lib import PathConfig

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

from qmri_sim_lib import PathConfig, PreloadedPhantom, preload_phantom_for_sim, probe_phantom_n_slices
from qmri_sim_run import (
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
    "adc_sse": ("adc_nlls", "adc_loglinear", "fa_map", "md_map", "reference_map"),
    "t2_sse": ("t2_nlls", "t2_loglinear", "reference_map"),
    "adc_multishot": ("adc_nlls", "adc_loglinear", "fa_map", "md_map", "reference_map"),
    "t2_multishot": ("t2_nlls", "t2_loglinear", "reference_map"),
    "adc_triple": ("adc_nlls", "adc_loglinear", "fa_map", "md_map", "reference_map"),
    "t2_triple": ("t2_nlls", "t2_loglinear", "reference_map"),
}


def _save_volume_nifti(vol: np.ndarray, path: str, res_mm: float) -> None:
    """Save an (Nx, Ny, Nz) float32 volume as NIfTI with an isotropic diagonal affine."""
    affine = np.diag([res_mm, res_mm, res_mm, 1.0]).astype(np.float64)
    nib.save(nib.Nifti1Image(vol.astype(np.float32), affine), path)


def _make_empty_slice_result(
    pipeline: str, Nx: int, Ny: int, tissue_names: list[str]
) -> dict:
    """Zero/NaN result dict for a phantom slice that contains no tissue voxels."""
    map_keys = _MAP_KEYS.get(pipeline, ())
    result: dict = {k: np.zeros((Ny, Nx), dtype=np.float32) for k in map_keys}
    result["mae_total"] = float("nan")
    result["mae_per_tissue"] = {name: float("nan") for name in tissue_names}
    result["mae_n_voxels_per_tissue"] = {name: 0 for name in tissue_names}
    result["mae_n_voxels_total"] = 0
    result["tissue_masks"] = {
        name: np.zeros((Ny, Nx), dtype=bool) for name in tissue_names
    }
    return result


def _stack_per_pipeline(slice_results: list[dict], pipeline: str) -> dict:
    """Stack a pipeline's per-slice result dicts into volume-shaped arrays."""
    out: dict = {}
    map_keys = _MAP_KEYS.get(pipeline, ())

    # Stack 2-D maps along the slice axis.
    for key in map_keys:
        if all(key in r for r in slice_results):
            out[key] = np.stack([np.asarray(r[key]) for r in slice_results], axis=0)

    # mae_total: 1-D vector across slices.
    out["mae_total"] = np.array(
        [r["mae_total"] for r in slice_results], dtype=float
    )

    # mae_per_tissue: dict[tissue, (n_slices,) array].
    tissue_names = list(slice_results[0]["mae_per_tissue"].keys())
    out["mae_per_tissue"] = {
        name: np.array(
            [r["mae_per_tissue"][name] for r in slice_results], dtype=float
        )
        for name in tissue_names
    }
    out["mae_n_voxels_per_tissue"] = {
        name: np.array(
            [r["mae_n_voxels_per_tissue"][name] for r in slice_results], dtype=int
        )
        for name in tissue_names
    }
    out["mae_n_voxels_total"] = np.array(
        [r["mae_n_voxels_total"] for r in slice_results], dtype=int
    )

    # Whole-volume MAE: voxel-weighted average across slices.
    n_tot = out["mae_n_voxels_total"]
    m_tot = out["mae_total"]
    valid_tot = np.isfinite(m_tot) & (n_tot > 0)
    if valid_tot.any():
        out["mae_total_volume"] = float(
            np.sum(m_tot[valid_tot] * n_tot[valid_tot]) / n_tot[valid_tot].sum()
        )
    else:
        out["mae_total_volume"] = float("nan")

    out["mae_per_tissue_volume"] = {}
    for name in tissue_names:
        m = out["mae_per_tissue"][name]
        n = out["mae_n_voxels_per_tissue"][name]
        valid = np.isfinite(m) & (n > 0)
        if valid.any():
            out["mae_per_tissue_volume"][name] = float(
                np.sum(m[valid] * n[valid]) / n[valid].sum()
            )
        else:
            out["mae_per_tissue_volume"][name] = float("nan")

    # Carry through fixed per-pipeline metadata from the first slice
    # (b_values, TEs, etc. — identical across slices).
    for key in ("b_values", "TEs", "echo_TEs_ms"):
        if key in slice_results[0]:
            out[key] = slice_results[0][key]

    # tissue_masks per slice — keep them stacked as well.
    if "tissue_masks" in slice_results[0]:
        tm_names = list(slice_results[0]["tissue_masks"].keys())
        tm_stacked: dict = {}
        for name in tm_names:
            arrs = []
            for r in slice_results:
                m = r["tissue_masks"][name]
                arrs.append(m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m))
            tm_stacked[name] = np.stack(arrs, axis=0)
        out["tissue_masks"] = tm_stacked

    return out


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
    See :func:`qmri_sim_run.run_all_qmri_simulations` for all other params.

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

    # Accumulate per-pipeline lists of per-slice result dicts.
    per_pipeline_slices: dict[str, list[dict]] = {p: [] for p in pipelines}
    _t_adc_total = 0.0
    _t_t2_total  = 0.0
    _t_volume_start = time.perf_counter()

    _Nx = _Ny = round(fov * 1e3 / res)

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
            for p in pipelines:
                per_pipeline_slices[p].append(
                    _make_empty_slice_result(p, _Nx, _Ny, preloaded.tissue_names)
                )
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
            per_pipeline_slices[p].append(slice_res[p])

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

        # Incremental save: update volume NIfTI after every completed slice.
        if save_volume_nifti:
            for p in pipelines:
                partial = _stack_per_pipeline(per_pipeline_slices[p], p)
                map_key = "adc_nlls" if p.startswith("adc") else "t2_nlls"
                if map_key in partial:
                    fname = f"{paths.phantom_name}-{map_key.upper()}_{p}_volume.nii.gz"
                    # partial[map_key]: (n_slices, Ny, Nx) — image space, z-first.
                    # Rotate each 2-D slice 90° CW (matching the SSE per-slice
                    # convention), then move the slice axis last → (Nx, Ny, n_slices).
                    _m = np.rot90(partial[map_key], k=-1, axes=(1, 2))
                    _m = np.transpose(_m, (1, 2, 0))
                    _save_volume_nifti(_m, os.path.join(paths.volumes_dir, fname), res)
            print(
                f"[volume] updated volume NIfTI files "
                f"({s_idx + 1}/{len(slice_indices)} slices done)",
                flush=True,
            )

    # Stack each pipeline for the final return value.
    out: dict = {"slice_indices": np.asarray(slice_indices, dtype=int)}
    for p in pipelines:
        stacked = _stack_per_pipeline(per_pipeline_slices[p], p)
        stacked["slice_indices"] = np.asarray(slice_indices, dtype=int)
        out[p] = stacked

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
    
    pipelines = ("adc_triple", "t2_triple", "adc_multishot", "t2_multishot")  
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