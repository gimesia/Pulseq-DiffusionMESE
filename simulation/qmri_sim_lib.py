"""Shared helpers for the qMRI simulation pipelines.

Provides:
    - PathConfig: dataclass holding all directory paths and the pulseq lib path
    - resolve_phantom_path / load_phantom_for_sim: phantom loading wrapper
    - simulate_signal: MR0 graph build + execute on GPU when available
    - save_magnitude_nifti: write a 2-D magnitude image as NIfTI
    - compute_trace_dwi: data-floored geometric mean across diffusion directions
    - ensure_seq_path_on_syspath: inject the pulseq_diffusion_mese directory
"""
from __future__ import annotations

import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import MRzeroCore as mr0
import nibabel as nib
import numpy as np
import torch


# ----------------------------------------------------------------------------
#  Compatibility shims required by the upstream MRzero / pulseq libraries.
# ----------------------------------------------------------------------------
np.int = int
np.float = float
np.complex = complex

warnings.filterwarnings("ignore", category=UserWarning, module="mrinufft")


# ----------------------------------------------------------------------------
#  Path container
# ----------------------------------------------------------------------------
@dataclass
class PathConfig:
    """All filesystem paths used by the simulation pipelines.

    Attributes
    ----------
    seq_lib_path:
        Directory containing EPIDiffusionSEPulseqSeq.py etc.
        (the `pulseq_diffusion_mese` package). Appended to sys.path.
    phantoms_dir:
        Root folder with brainweb_phantom_* subdirectories.
    phantom_name:
        Folder name of the phantom to load (e.g. "brainweb_phantom_0").
        The JSON descriptor is expected at
        ``<phantoms_dir>/<phantom_name>/<phantom_name>-3T.json``.
    sequences_dir, volumes_dir, masks_dir:
        Output directories for .seq files, .npy maps and tissue masks.
    diff_img_dir, t2_img_dir:
        Output directories for per-echo magnitude NIfTI images.
    """

    seq_lib_path: str
    phantoms_dir: str
    phantom_name: str
    sequences_dir: str
    volumes_dir: str
    masks_dir: str
    diff_img_dir: str
    t2_img_dir: str

    def ensure_dirs(self) -> None:
        for d in (
            self.sequences_dir,
            self.volumes_dir,
            self.masks_dir,
            self.diff_img_dir,
            self.t2_img_dir,
        ):
            os.makedirs(d, exist_ok=True)


# ----------------------------------------------------------------------------
#  sys.path / logging setup
# ----------------------------------------------------------------------------
def ensure_seq_path_on_syspath(seq_lib_path: str) -> None:
    if seq_lib_path and seq_lib_path not in sys.path:
        sys.path.append(seq_lib_path)


def make_quiet_logger() -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.FATAL)
    return logger


# ----------------------------------------------------------------------------
#  Phantom loading
# ----------------------------------------------------------------------------
def resolve_phantom_json(paths: PathConfig) -> str:
    return os.path.join(
        paths.phantoms_dir, paths.phantom_name, f"{paths.phantom_name}-3T.json"
    )


def load_phantom_for_sim(
    paths: PathConfig,
    resolution_mm: float,
    slice_idx: Optional[int],
):
    """Load the phantom using the project's phantom_loader.

    Returns ``(phantom, phantom_data, tissue_masks)``.
    """
    # Imported lazily so that phantom_loader can be picked up from the
    # working directory regardless of how this library is imported.
    import phantom_loader

    phantom_path = resolve_phantom_json(paths)
    return phantom_loader.load_phantom(
        json_path=phantom_path,
        resolution_mm=resolution_mm,
        slice_idx=slice_idx,
    )


# ----------------------------------------------------------------------------
#  Simulation
# ----------------------------------------------------------------------------
def simulate_signal(
    seq_file_path: str,
    phantom_data,
    use_gpu: bool,
    *,
    gpu_max_states: int = 50000,
    gpu_min_emit: float = 1e-6,
    cpu_max_states: int = 5000,
    cpu_min_emit: float = 1e-5,
    print_progress: bool = False,
):
    """Build the MR0 graph for the .seq file and execute it.

    Returns a CPU tensor of complex signal samples.
    """
    seq0 = mr0.Sequence.import_file(seq_file_path)
    if use_gpu:
        seq0_gpu = seq0.cuda()
        phantom_gpu = phantom_data.cuda()
        graph = mr0.compute_graph(seq0_gpu, phantom_gpu, gpu_max_states, gpu_min_emit)
        signal = mr0.execute_graph(
            graph, seq0_gpu, phantom_gpu, print_progress=print_progress
        ).cpu()
        del seq0_gpu, phantom_gpu
        torch.cuda.empty_cache()
    else:
        phantom_cpu = phantom_data.cpu()
        graph = mr0.compute_graph(seq0, phantom_cpu, cpu_max_states, cpu_min_emit)
        signal = mr0.execute_graph(graph, seq0, phantom_cpu, print_progress=print_progress)
    return signal


# ----------------------------------------------------------------------------
#  NIfTI helpers
# ----------------------------------------------------------------------------
def affine_from_res(res: float) -> np.ndarray:
    return np.array([[res, 0, 0, 0], [0, res, 0, 0], [0, 0, res, 0], [0, 0, 0, 1]])


def save_magnitude_nifti(mag_img: np.ndarray, path: str, res: float) -> None:
    """Save a (Ny, Nx) magnitude image as a single-slice NIfTI volume."""
    affine = affine_from_res(res)
    vol = np.asarray(mag_img[:, :, np.newaxis], dtype=np.float32)
    nib.save(nib.Nifti1Image(vol, affine), path)


# ----------------------------------------------------------------------------
#  Diffusion post-processing
# ----------------------------------------------------------------------------
def compute_trace_dwi(mag: np.ndarray) -> np.ndarray:
    """Geometric mean across diffusion directions.

    ``mag`` is shaped ``(n_b, n_dirs, Ny, Nx)``. The floor is data-scaled so
    that a single noise-floor voxel in one direction at high b does not drag
    the geometric mean to ~0 and produce a spurious fast-decay.
    """
    eps_local = 1e-3 * float(mag[0].max())
    return np.exp(np.mean(np.log(np.maximum(mag, eps_local)), axis=1))


def save_tissue_masks(tissue_masks, masks_dir: str, phantom_name: str) -> None:
    """Stack the tissue-mask dict into ``(n_tissues, Ny, Nx)`` and save as .npy."""
    masks = torch.stack(list(tissue_masks.values()), dim=0).numpy()
    os.makedirs(masks_dir, exist_ok=True)
    np.save(os.path.join(masks_dir, f"{phantom_name}-tissue_masks.npy"), masks)


# ----------------------------------------------------------------------------
#  Per-tissue MAE
# ----------------------------------------------------------------------------
def _mask_to_2d_bool(mask) -> np.ndarray:
    """Coerce a tissue-mask tensor to a 2-D boolean numpy array."""
    if hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()
    return np.squeeze(np.asarray(mask)).astype(bool)


def combined_tissue_mask(tissue_masks: dict) -> np.ndarray:
    """Union of all tissue masks → the 'combined' phantom region."""
    if "combined" in tissue_masks:
        return _mask_to_2d_bool(tissue_masks["combined"])
    out = None
    for m in tissue_masks.values():
        m2 = _mask_to_2d_bool(m)
        out = m2 if out is None else (out | m2)
    return out if out is not None else np.zeros((0, 0), dtype=bool)


def compute_mae_per_tissue(
    estimated: np.ndarray,
    reference: np.ndarray,
    tissue_masks: dict,
    *,
    exclude_zeros: bool = True,
) -> dict:
    """Mean Absolute Error per tissue and across the combined phantom region.

    Both ``estimated`` and ``reference`` are squeezed and assumed to be on the
    same 2-D grid as ``tissue_masks``. The 'combined' MAE is computed over
    the union of all tissue masks. With ``exclude_zeros`` (default) voxels
    where either map is exactly zero are dropped from the average — keeps
    background voxels with collapsed fits from biasing the mean.

    Returns
    -------
    dict with keys:
        - ``per_tissue``: ``{tissue_name: mae_value, ...}``
        - ``total``: scalar MAE over the combined mask
        - ``n_voxels_per_tissue``: count of valid voxels per tissue
        - ``n_voxels_total``: count of valid voxels for the combined mask
    """
    est = np.squeeze(np.asarray(estimated))
    ref = np.squeeze(np.asarray(reference))
    if est.shape != ref.shape:
        raise ValueError(
            f"Estimated {est.shape} and reference {ref.shape} shapes differ."
        )

    def _mae_in(mask: np.ndarray):
        if mask.shape != est.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match map shape {est.shape}."
            )
        valid = mask
        if exclude_zeros:
            valid = valid & (est != 0) & (ref != 0)
        n = int(valid.sum())
        if n == 0:
            return float("nan"), 0
        return float(np.mean(np.abs(est[valid] - ref[valid]))), n

    per_tissue: dict[str, float] = {}
    n_per_tissue: dict[str, int] = {}
    for name, mask in tissue_masks.items():
        m = _mask_to_2d_bool(mask)
        v, n = _mae_in(m)
        per_tissue[name] = v
        n_per_tissue[name] = n

    combined = combined_tissue_mask(tissue_masks)
    total, n_total = _mae_in(combined)

    return {
        "per_tissue": per_tissue,
        "total": total,
        "n_voxels_per_tissue": n_per_tissue,
        "n_voxels_total": n_total,
    }


def phantom_map_to_2d(t) -> np.ndarray:
    """Squeeze a phantom map (torch tensor or ndarray) to a 2-D numpy array."""
    if hasattr(t, "cpu"):
        t = t.cpu().numpy()
    return np.squeeze(np.asarray(t))


# ----------------------------------------------------------------------------
#  Slice-count probe (no modification to phantom_loader)
# ----------------------------------------------------------------------------
def probe_phantom_n_slices(
    paths: "PathConfig",
    *,
    fov_mm: float = 224.0,
    resolution_mm: float = 224.0 / 96,
) -> int:
    """Determine the number of usable z-slices for a phantom JSON.

    Mirrors the arithmetic inside :func:`phantom_loader.load_phantom` so the
    multi-slice runner can default to ``range(0, n_slices)`` without
    requiring the user to pre-compute the count.
    """
    tissue_dict = mr0.TissueDict.load(resolve_phantom_json(paths))
    tissue_dict.pop("fat", None)
    tissue_dict.pop("vessels", None)
    phantom = tissue_dict.combine()
    size_z_m = float(phantom.size[2].item())
    return int(round(size_z_m / (resolution_mm * 1e-3)))
