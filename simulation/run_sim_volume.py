"""Multi-slice driver: run :func:`run_all_qmri_simulations` over every slice
of a phantom and stack the per-pipeline results into 3-D volumes.

The driver:

1. Preloads the phantom once (3-D interpolation is the expensive step) and
   slices it per simulation, so the per-slice call to ``run_sim`` never
   reloads from disk.
2. For EPI pipelines (single-shot SE and triple-SE) runs each slice twice
   with opposite EPI blip polarity (``blipdown`` and ``blipup``) so the
   downstream FSL topup pipeline can estimate B0 distortion.
3. Pre-allocates ``(Ny, Nx, n_slices)`` volumes with NaN initialisation so
   skipped (empty) slices stay as NaN and never bias the volume-MAE
   summary.
4. Incrementally saves a NIfTI per pipeline / blip variant after each slice
   completes, so a long run can be inspected (or interrupted) without
   losing progress.

Author      : Aron Gimesi <aron.gimesi@tecnico.ulisboa.pt>
Affiliation : Instituto Superior Técnico | MSCA-DN IQ-BRAIN
Date        : 2026
Context     : ESMRMB 2026 — Pulseq DiffusionMESE showcase

Funding acknowledgement (mandatory):
    IQ-BRAIN is funded by the European Union (MSCA Doctoral Network,
    December 2024–November 2028, Grant Agreement No. 101169519).

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
    # results["adc_sse_blipdown"]["adc_nlls"]    (Ny, Nx, n_slices)
    # results["adc_sse_blipdown"]["mae_per_tissue"]   {"wm": (n_slices,), ...}
    # results["adc_sse_blipdown"]["mae_total"]   (n_slices,)
    # results["adc_sse_blipdown"]["slice_indices"]    (n_slices,)
"""
from __future__ import annotations

import os
import time
from typing import Iterable, Optional, Sequence

import nibabel as nib
import numpy as np
import torch

from utils_sim_lib import PathConfig, PreloadedPhantom, preload_phantom_for_sim, probe_phantom_n_slices
from run_sim import (
    DEFAULT_ADC_BVALUES,
    DEFAULT_T2_TES,
    DEFAULT_T2_TRIPLE_TE1,
    PIPELINES,
    run_all_qmri_simulations,
)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch.jit")

def _fmt(seconds: float) -> str:
    """Format a duration as 'Xm Ys' (>=60 s) or 'X.Xs' (< 60 s)."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs:02d}s"


# Per-pipeline list of map-like keys (2-D arrays) to stack across slices.
# Anything not listed is collected as a Python list (preserves order).
_MAP_KEYS = {
    "adc_sse": ("adc_nlls", ),
    "t2_sse": ("t2_nlls", ),
    "adc_multishot": ("adc_nlls",),
    "t2_multishot": ("t2_nlls",),
    "adc_triple": ("adc_nlls",),
    "t2_triple": ("t2_nlls",),
}

# EPI pipelines whose ``blip_down`` flag changes the simulated geometric
# distortion direction. These are run twice per slice (down then up).
_BLIP_PIPES = frozenset({"adc_sse", "t2_sse", "adc_triple", "t2_triple"})


def _variants_for(pipeline_name: str) -> list[tuple[bool, str]]:
    """Return [(blip_down_flag, filename_suffix)] for a pipeline.

    EPI pipelines are simulated twice with opposite blip polarity for B0
    field mapping; multishot pipelines have no such dependency and run once.
    """
    if pipeline_name in _BLIP_PIPES:
        return [(True, "_blipdown"), (False, "_blipup")]
    return [(True, "")]

def _save_volume_nifti(volume: np.ndarray, path: str, res_mm: float, fill_nan: bool = True) -> None:
    """Save an (Nx, Ny, Nz) float32 volume as NIfTI with an isotropic diagonal affine.
    
    Args:
        volume:   3D array to save.
        path:     Output file path.
        res_mm:   Isotropic voxel size in mm.
        fill_nan: If True (default), replace NaN/Inf with 0 before saving.
    """
    affine = np.diag([res_mm, res_mm, res_mm, 1.0]).astype(np.float64)


def _reorient_volume(vol_3d: np.ndarray) -> np.ndarray:
    """Apply the canonical orientation transform shared by every saved NIfTI volume.

    Matches the per-slice reorientation used by the qMRI map writers so reference
    maps, tissue masks, NLLS maps, and weighted-image volumes all land on disk in
    the same frame and overlay correctly in a viewer.
    """
    return np.flip(np.rot90(vol_3d, k=1, axes=(0, 1)), axis=1)


def _flush_variant_volumes(
    vol: dict,
    variant_keys: list[tuple[str, str, bool]],
    paths: PathConfig,
    res_mm: float,
) -> None:
    """Write the full pre-allocated NLLS map and weighted-image volumes to disk.

    Always writes the entire pre-allocated array (NaN-initialised), never a
    partial slice prefix, so the saved NIfTI shape stays equal to the phantom's
    full ``(Ny, Nx, n_slices)`` even if leading / middle / trailing slices were
    skipped because the phantom was empty there.
    """
    for pipeline_name, suffix, _ in variant_keys:
        variant_key = f"{pipeline_name}{suffix}"
        map_key = "adc_nlls" if pipeline_name.startswith("adc") else "t2_nlls"
        if map_key in vol[variant_key]:
            full = vol[variant_key][map_key]
            fname = f"{paths.phantom_name}-{map_key.upper()}_{variant_key}_volume.nii.gz"
            _save_volume_nifti(
                _reorient_volume(full),
                os.path.join(paths.volumes_dir, fname),
                res_mm,
            )
        weighted_vols = vol[variant_key].get("weighted_vols", {})
        if weighted_vols:
            output_dir = paths.t2_vol_dir if pipeline_name.startswith("t2_") else paths.adc_vol_dir
            for stem, weighted_volume in weighted_vols.items():
                out_fname = f"{paths.phantom_name}-{stem}.nii.gz"
                _save_volume_nifti(
                    _reorient_volume(weighted_volume),
                    os.path.join(output_dir, out_fname),
                    res_mm,
                )



def run_all_qmri_simulations_volume(
    paths: PathConfig,
    *,
    slice_indices: Optional[Sequence[int]] = None,
    fov: float = 224e-3,
    res: float = 2.33333333,
    system_type: Optional[object] = None,
    b_values: Sequence[int] = DEFAULT_ADC_BVALUES,
    TEs_t2: Sequence[int] = DEFAULT_T2_TES,
    TE1_values_triple: Sequence[int] = DEFAULT_T2_TRIPLE_TE1,
    TE_adc: int = 100,
    TR: int = 5000,
    b_directions: int = 3,
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
        - 2-D maps stacked along the last (slice) axis (e.g. ``adc_nlls`` of
          shape ``(Ny, Nx, n_slices)``)
        - ``mae_per_tissue``: ``{tissue_name: (n_slices,) array}``
        - ``mae_total``: ``(n_slices,)`` array
        - ``mae_per_tissue_volume`` / ``mae_total_volume``: voxel-weighted
          mean across all slices (single scalar each)
        - ``tissue_masks``: ``{tissue_name: (Ny, Nx, n_slices) bool array}``
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
    ref_D_volume  = preloaded.phantom.D.cpu().numpy()
    ref_T2_volume = preloaded.phantom.T2.cpu().numpy()
    _save_volume_nifti(ref_D_volume, os.path.join(paths.volumes_dir, f"{paths.phantom_name}-D_ref_volume.nii.gz"), res)
    _save_volume_nifti(ref_T2_volume, os.path.join(paths.volumes_dir, f"{paths.phantom_name}-T2_ref_volume.nii.gz"), res)
    print(f"[volume] saved D_ref_volume and T2_ref_volume  shape={ref_D_volume.shape}", flush=True)

    # Build 3D tissue masks from the preloaded phantom (mirrors the per-slice
    # argmax+threshold logic in phantom_loader.slice_preloaded_phantom) and save
    # them upfront so the masks land on disk together with the reference maps.
    pd_stack = torch.stack(
        [preloaded.phantom.tissue_masks[name].squeeze(-1).float()
         for name in preloaded.tissue_names], dim=0
    )
    tissue_assignment = pd_stack.argmax(dim=0)
    volume_tissue_masks = {
        name: ((tissue_assignment == idx) & (pd_stack[idx] > 0.05)).cpu().numpy()
        for idx, name in enumerate(preloaded.tissue_names)
    }
    for tissue_name, mask_volume in volume_tissue_masks.items():
        _save_volume_nifti(
            _reorient_volume(mask_volume.astype(np.float32)),
            os.path.join(paths.masks_dir, f"{paths.phantom_name}-mask_{tissue_name}_volume.nii.gz"),
            res,
        )
    print(
        f"[volume] saved {len(volume_tissue_masks)} tissue-mask NIfTI files "
        f"({list(volume_tissue_masks.keys())})",
        flush=True,
    )

    elapsed_adc_total = 0.0
    elapsed_t2_total  = 0.0
    volume_start_time = time.perf_counter()

    Ny = Nx = round(fov * 1e3 / res)
    n_slices_total = len(slice_indices)

    # Variant table: (pipeline, suffix, blip_down). EPI pipelines run twice
    # (blip-down then blip-up); multishot runs once (blip_down is ignored).
    variant_keys: list[tuple[str, str, bool]] = []
    for pipeline_name in pipelines:
        for blip_down_flag, suffix in _variants_for(pipeline_name):
            variant_keys.append((pipeline_name, suffix, blip_down_flag))

    # Pre-allocate output volumes (NaN-init so skipped slices stay as NaN).
    # Shape: (Ny, Nx, n_slices) — slice is the last dimension. One entry per
    # variant key (e.g. "t2_sse_blipdown", "t2_sse_blipup", "t2_multishot").
    vol: dict[str, dict] = {}
    for pipeline_name, suffix, _ in variant_keys:
        variant_key = f"{pipeline_name}{suffix}"
        variant_data: dict = {}
        for map_name in _MAP_KEYS.get(pipeline_name, ()):
            variant_data[map_name] = np.full((Ny, Nx, n_slices_total), np.nan, dtype=np.float32)
        variant_data["mae_total"]          = np.full(n_slices_total, np.nan, dtype=float)
        variant_data["mae_n_voxels_total"] = np.zeros(n_slices_total, dtype=int)
        variant_data["mae_per_tissue"]          = {t: np.full(n_slices_total, np.nan, dtype=float) for t in preloaded.tissue_names}
        variant_data["mae_n_voxels_per_tissue"] = {t: np.zeros(n_slices_total, dtype=int)          for t in preloaded.tissue_names}
        variant_data["tissue_masks"]            = {t: np.zeros((Ny, Nx, n_slices_total), dtype=bool) for t in preloaded.tissue_names}
        # Weighted-image volumes are allocated lazily on the first non-empty
        # slice for each variant (stems aren't known until then).
        variant_data["weighted_vols"] = {}
        vol[variant_key] = variant_data

    # Split requested pipelines into "needs both blip directions" and "doesn't".
    # Pass 1 (blip_down=True) covers EPI-blipdown + all multishot; pass 2
    # (blip_down=False) only re-runs the EPI pipelines.
    blip_pipes_requested    = tuple(p for p in pipelines if p in _BLIP_PIPES)
    nonblip_pipes_requested = tuple(p for p in pipelines if p not in _BLIP_PIPES)
    pass1_pipelines = blip_pipes_requested + nonblip_pipes_requested
    pass2_pipelines = blip_pipes_requested

    def _ingest(
        target: dict,
        slice_result: dict,
        pipeline_name: str,
        slice_position: int,
    ) -> None:
        """Write one slice's worth of results into the pre-allocated volumes."""
        for map_name in _MAP_KEYS.get(pipeline_name, ()):
            target[map_name][..., slice_position] = slice_result[map_name]
        target["mae_total"][slice_position]          = slice_result["mae_total"]
        target["mae_n_voxels_total"][slice_position] = slice_result["mae_n_voxels_total"]
        for tissue_name in preloaded.tissue_names:
            target["mae_per_tissue"][tissue_name][slice_position]          = slice_result["mae_per_tissue"][tissue_name]
            target["mae_n_voxels_per_tissue"][tissue_name][slice_position] = slice_result["mae_n_voxels_per_tissue"][tissue_name]
            tissue_mask = slice_result["tissue_masks"][tissue_name]
            target["tissue_masks"][tissue_name][..., slice_position] = (
                tissue_mask.cpu().numpy() if hasattr(tissue_mask, "cpu") else np.asarray(tissue_mask)
            )
        # Weighted-image volumes: lazy-allocate on first encounter of each
        # stem, then write the 2D slice into the matching 3D array.
        weighted_imgs = slice_result.get("weighted_images") or {}
        weighted_vols = target["weighted_vols"]
        for stem, img_2d in weighted_imgs.items():
            if stem not in weighted_vols:
                weighted_vols[stem] = np.full((Ny, Nx, n_slices_total), np.nan, dtype=np.float32)
            arr = np.asarray(img_2d)
            if hasattr(arr, "cpu"):
                arr = arr.cpu().numpy()
            weighted_vols[stem][..., slice_position] = arr
        if "_metadata_captured" not in target:
            for metadata_key in ("b_values", "TEs", "echo_TEs_ms"):
                if metadata_key in slice_result:
                    target[metadata_key] = slice_result[metadata_key]
            target["_metadata_captured"] = True

    shared_call_kwargs = dict(
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
        use_gpu=use_gpu,
        save_slice_npy=False,
        preloaded_phantom=preloaded,
    )

    for slice_position, slice_idx in enumerate(slice_indices):
        print(
            f"\n========= slice {slice_position + 1}/{len(slice_indices)} "
            f"(index {slice_idx}) =========",
            flush=True,
        )

        # Skip slices where the phantom has no tissue — mr0.compute_graph() panics
        # with a Rust NaN error when PD is all-zero.
        pd_max = float(preloaded.phantom.PD[:, :, slice_idx].max())
        if pd_max < 1e-9:
            print(f"[volume] slice {slice_idx} empty (PD_max={pd_max:.1e}), filling NaN", flush=True)
            continue

        slice_timings: dict[str, float] = {}

        # ----- Pass 1: blip-down (all pipelines that were requested) -----
        if pass1_pipelines:
            print(f"[volume] pass 1/2  blip_down=True  pipelines={pass1_pipelines}", flush=True)
            slice_results_down = run_all_qmri_simulations(
                paths,
                slice_idx=slice_idx,
                blip_down=True,
                pipelines=pass1_pipelines,
                **shared_call_kwargs,
            )
            for pipeline_name in pass1_pipelines:
                suffix = "_blipdown" if pipeline_name in _BLIP_PIPES else ""
                _ingest(vol[f"{pipeline_name}{suffix}"], slice_results_down[pipeline_name], pipeline_name, slice_position)
            for pipeline_name, elapsed in slice_results_down.get("_timings", {}).items():
                slice_timings[pipeline_name] = slice_timings.get(pipeline_name, 0.0) + elapsed

        # ----- Pass 2: blip-up (EPI pipelines only) -----
        if pass2_pipelines:
            print(f"[volume] pass 2/2  blip_down=False pipelines={pass2_pipelines}", flush=True)
            slice_results_up = run_all_qmri_simulations(
                paths,
                slice_idx=slice_idx,
                blip_down=False,
                pipelines=pass2_pipelines,
                **shared_call_kwargs,
            )
            for pipeline_name in pass2_pipelines:
                _ingest(vol[f"{pipeline_name}_blipup"], slice_results_up[pipeline_name], pipeline_name, slice_position)
            for pipeline_name, elapsed in slice_results_up.get("_timings", {}).items():
                slice_timings[pipeline_name] = slice_timings.get(pipeline_name, 0.0) + elapsed

        # Per-slice timing breakdown (combined across both passes).
        elapsed_adc_slice = sum(v for k, v in slice_timings.items() if k.startswith("adc"))
        elapsed_t2_slice  = sum(v for k, v in slice_timings.items() if k.startswith("t2"))
        elapsed_adc_total += elapsed_adc_slice
        elapsed_t2_total  += elapsed_t2_slice
        elapsed_so_far = time.perf_counter() - volume_start_time
        print(
            f"[time] slice {slice_position + 1}/{len(slice_indices)}  "
            f"ADC={_fmt(elapsed_adc_slice)}  T2={_fmt(elapsed_t2_slice)}  "
            f"slice total={_fmt(elapsed_adc_slice + elapsed_t2_slice)}",
            flush=True,
        )
        print(
            f"[time] cumulative  "
            f"ADC={_fmt(elapsed_adc_total)}  T2={_fmt(elapsed_t2_total)}  "
            f"elapsed={_fmt(elapsed_so_far)}",
            flush=True,
        )

        # Incremental save: write the FULL pre-allocated volume each tick so
        # that any slice we haven't reached yet — or one that was skipped as
        # empty — lands on disk as NaN. This keeps the saved volume's shape
        # equal to (Ny, Nx, n_slices_total) for the entire run, instead of
        # silently dropping trailing empty slices.
        if save_volume_nifti:
            _flush_variant_volumes(vol, variant_keys, paths, res)
            print(
                f"[volume] updated volume NIfTI files "
                f"({slice_position + 1}/{len(slice_indices)} slices done)",
                flush=True,
            )

    # Final flush — guarantees trailing empty slices (which hit `continue` and
    # never reached the in-loop save block) end up on disk as NaN.
    if save_volume_nifti:
        _flush_variant_volumes(vol, variant_keys, paths, res)
        print("[volume] final save: full pre-allocated volumes written", flush=True)

    slice_idx_arr = np.asarray(slice_indices, dtype=int)
    np.save(os.path.join(paths.volumes_dir, f"{paths.phantom_name}-slice_indices.npy"), slice_idx_arr)

    # Compute volume-level MAE from pre-allocated arrays and finalise output.
    out: dict = {"slice_indices": slice_idx_arr}
    variant_key_names = [f"{name}{suffix}" for name, suffix, _ in variant_keys]
    for variant_key in variant_key_names:
        n_voxels_total = vol[variant_key]["mae_n_voxels_total"]
        mae_total_per_slice = vol[variant_key]["mae_total"]
        valid = np.isfinite(mae_total_per_slice) & (n_voxels_total > 0)
        vol[variant_key]["mae_total_volume"] = (
            float(np.sum(mae_total_per_slice[valid] * n_voxels_total[valid]) / n_voxels_total[valid].sum())
            if valid.any() else float("nan")
        )
        vol[variant_key]["mae_per_tissue_volume"] = {}
        for tissue_name in preloaded.tissue_names:
            mae_per_slice = vol[variant_key]["mae_per_tissue"][tissue_name]
            n_voxels_per_slice = vol[variant_key]["mae_n_voxels_per_tissue"][tissue_name]
            valid_per_tissue = np.isfinite(mae_per_slice) & (n_voxels_per_slice > 0)
            vol[variant_key]["mae_per_tissue_volume"][tissue_name] = (
                float(np.sum(mae_per_slice[valid_per_tissue] * n_voxels_per_slice[valid_per_tissue])
                      / n_voxels_per_slice[valid_per_tissue].sum())
                if valid_per_tissue.any() else float("nan")
            )
        vol[variant_key].pop("_metadata_captured", None)
        vol[variant_key]["slice_indices"] = slice_idx_arr
        out[variant_key] = vol[variant_key]

    elapsed_volume_total = time.perf_counter() - volume_start_time
    print("\n========== Volume timing summary ==========", flush=True)
    print(f"  ADC total  : {_fmt(elapsed_adc_total)}", flush=True)
    print(f"  T2 total   : {_fmt(elapsed_t2_total)}", flush=True)
    print(f"  Volume     : {_fmt(elapsed_volume_total)}", flush=True)

    # Print a compact MAE summary
    print("\n========== Volume MAE summary ==========", flush=True)
    for variant_key in variant_key_names:
        variant_summary = out[variant_key]
        per_tissue_summary = ", ".join(
            f"{tissue}={mae:.4f}" for tissue, mae in variant_summary["mae_per_tissue_volume"].items()
        )
        print(
            f"  {variant_key:<24} total={variant_summary['mae_total_volume']:.4f}  [{per_tissue_summary}]",
            flush=True,
        )

    return out


if __name__ == "__main__":
    import os

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    paths = PathConfig(
        seq_lib_path  = os.path.join(project_root, "pulseq_diffusion_mese"),
        phantoms_dir  = os.path.join(project_root, "brainweb_phantoms"),
        phantom_name  = "brainweb-subj04",
        sequences_dir = os.path.join(project_root, "simulation", "simulated", "seq"),
        volumes_dir   = os.path.join(project_root, "simulation", "simulated", "volumes"),
        masks_dir     = os.path.join(project_root, "simulation", "simulated", "masks"),
        diff_img_dir  = os.path.join(project_root, "simulation", "simulated", "diff_img"),
        t2_img_dir    = os.path.join(project_root, "simulation", "simulated", "t2_img"),
        t2_vol_dir    = os.path.join(project_root, "simulation", "simulated", "t2_vol"),
        adc_vol_dir   = os.path.join(project_root, "simulation", "simulated", "adc_vol"),
    )

    # pipelines = ("adc_multishot", "t2_multishot",)
    # results = run_all_qmri_simulations_volume(
    #     paths,
    #     slice_indices=None,
    #     pipelines=pipelines,
    #     save_volume_nifti=True,
    # )

    # pipelines = ("adc_sse", "t2_sse",)
    # results = run_all_qmri_simulations_volume(
    #     paths,
    #     slice_indices=None,
    #     pipelines=pipelines,
    #     save_volume_nifti=True,
    # )

    pipelines = ("adc_triple", "t2_triple",)
    results = run_all_qmri_simulations_volume(
        paths,
        TE1_values_triple=range(65, 66, 5),   # (3 per TR)
        b_values=range(0, 1000, 1000),      # fewer b-values to speed up testing
        b_directions=1,                    # fewer directions to speed up testing
        pipelines=("t2_triple", "adc_triple"),   # remove kwarg to run all 6 pipelines
    )


    for pipeline_name in pipelines:
        summary = results[pipeline_name]
        per_tissue_summary = ", ".join(
            f"{tissue}={mae:.4f}" for tissue, mae in summary["mae_per_tissue_volume"].items()
        )
        print(f"  {pipeline_name:<14}  total={summary['mae_total_volume']:.4f}  [{per_tissue_summary}]")
        map_arr = summary.get("adc_nlls", summary.get("t2_nlls"))
        if map_arr is not None:
            print(f"  map shape      : {map_arr.shape}")