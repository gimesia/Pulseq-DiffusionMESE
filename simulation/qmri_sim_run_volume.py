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
from typing import Iterable, Optional, Sequence

import numpy as np

from qmri_sim_lib import PathConfig, probe_phantom_n_slices
from qmri_sim_run import (
    DEFAULT_T2_TES,
    DEFAULT_T2_TRIPLE_TE1,
    PIPELINES,
    run_all_qmri_simulations,
)


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
    b_values: Sequence[int] = tuple(range(0, 2001, 100)),
    TEs_t2: Sequence[int] = DEFAULT_T2_TES,
    TE1_values_triple: Sequence[int] = DEFAULT_T2_TRIPLE_TE1,
    TE_adc: int = 100,
    TR: int = 5000,
    b_directions: int = 6,
    small_delta: Optional[float] = None,
    big_DELTA: Optional[float] = None,
    small_delta_t2: float = 0.018,
    big_DELTA_t2: float = 0.03,
    ETL: int = 1,
    blip_down: bool = True,
    use_gpu: Optional[bool] = None,
    pipelines: Optional[Iterable[str]] = None,
    save_volume_npy: bool = True,
) -> dict:
    """Run every requested pipeline on every slice of the phantom.

    Parameters
    ----------
    slice_indices:
        List of slice indices to run. ``None`` → every slice in the phantom
        (auto-probed via :func:`probe_phantom_n_slices`).
    save_volume_npy:
        When ``True`` (default), save the stacked NLLS maps for each pipeline
        as ``<phantom_name>-<MAP>_<pipeline>_volume.npy`` in ``paths.volumes_dir``.
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
        f"× {len(pipelines)} pipelines"
    )

    # Accumulate per-pipeline lists of per-slice result dicts.
    per_pipeline_slices: dict[str, list[dict]] = {p: [] for p in pipelines}

    for s_idx, slice_idx in enumerate(slice_indices):
        print(
            f"\n========= slice {s_idx + 1}/{len(slice_indices)} "
            f"(index {slice_idx}) ========="
        )
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
        )
        for p in pipelines:
            per_pipeline_slices[p].append(slice_res[p])

    # Stack each pipeline.
    out: dict = {"slice_indices": np.asarray(slice_indices, dtype=int)}
    for p in pipelines:
        stacked = _stack_per_pipeline(per_pipeline_slices[p], p)
        stacked["slice_indices"] = np.asarray(slice_indices, dtype=int)
        out[p] = stacked

        if save_volume_npy:
            map_key = "adc_nlls" if p.startswith("adc") else "t2_nlls"
            if map_key in stacked:
                fname = f"{paths.phantom_name}-{map_key.upper()}_{p}_volume.npy"
                np.save(os.path.join(paths.volumes_dir, fname), stacked[map_key])
                print(f"[volume] saved {fname}")

    # Print a compact MAE summary
    print("\n========== Volume MAE summary ==========")
    for p in pipelines:
        s = out[p]
        per_t = ", ".join(
            f"{k}={v:.4f}" for k, v in s["mae_per_tissue_volume"].items()
        )
        print(f"  {p:<14} total={s['mae_total_volume']:.4f}  [{per_t}]")

    return out
