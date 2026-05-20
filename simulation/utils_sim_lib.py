"""Shared helpers for the qMRI simulation pipelines.

Provides:
    - ``PathConfig`` — dataclass holding all directory paths and the pulseq
      library path.
    - ``load_phantom_for_sim`` / ``preload_phantom_for_sim`` /
      ``extract_phantom_slice`` — phantom-loading wrappers around
      :mod:`phantom_loader`.
    - ``simulate_signal`` — MR0 graph build + execute, picking GPU when
      ``torch.cuda.is_available()``.
    - ``save_magnitude_nifti`` — write a 2-D magnitude image as a single-slice
      NIfTI volume.
    - ``compute_trace_dwi`` — data-floored geometric mean across diffusion
      directions (trace-DWI for ADC fitting).
    - ``compute_mae_per_tissue`` — per-tissue and combined MAE for validation.
    - ``ensure_seq_path_on_syspath`` — inject the ``pulseq_diffusion_mese``
      directory into ``sys.path`` so the sequence classes import cleanly.

Author      : Aron Gimesi <aron.gimesi@tecnico.ulisboa.pt>
Affiliation : Instituto Superior Técnico | MSCA-DN IQ-BRAIN
Date        : 2026
Context     : ESMRMB 2026 — Pulseq DiffusionMESE showcase

Funding acknowledgement (mandatory):
    IQ-BRAIN is funded by the European Union (MSCA Doctoral Network,
    December 2024–November 2028, Grant Agreement No. 101169519).
"""
from __future__ import annotations

import contextlib
import io
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
    t2_vol_dir, adc_vol_dir:
        Output directories for 3D (Ny, Nx, n_slices) weighted-image
        volumes — one NIfTI per echo / b-value / direction / blip variant,
        used as FSL topup input.
    """

    seq_lib_path: str
    phantoms_dir: str
    phantom_name: str
    sequences_dir: str
    volumes_dir: str
    masks_dir: str
    diff_img_dir: str
    t2_img_dir: str
    t2_vol_dir: str = ""
    adc_vol_dir: str = ""

    def ensure_dirs(self) -> None:
        """Create every output directory listed on this config (idempotent)."""
        for output_dir in (
            self.sequences_dir,
            self.volumes_dir,
            self.masks_dir,
            self.diff_img_dir,
            self.t2_img_dir,
            self.t2_vol_dir,
            self.adc_vol_dir,
        ):
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)


# ----------------------------------------------------------------------------
#  sys.path / logging setup
# ----------------------------------------------------------------------------
def ensure_seq_path_on_syspath(seq_lib_path: str) -> None:
    """Append the ``pulseq_diffusion_mese`` library path to :mod:`sys.path`.

    Idempotent — repeat calls with the same path are no-ops.
    """
    if seq_lib_path and seq_lib_path not in sys.path:
        sys.path.append(seq_lib_path)


def make_quiet_logger() -> logging.Logger:
    """Silence all logging globally so progress bars are not interrupted."""
    logging.disable(logging.CRITICAL)
    logger = logging.getLogger()
    logger.setLevel(logging.FATAL)
    return logger


# ----------------------------------------------------------------------------
#  Phantom loading
# ----------------------------------------------------------------------------
def resolve_phantom_json(paths: PathConfig) -> str:
    """Build the absolute path to the phantom JSON descriptor."""
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
    import phantom_loader

    phantom_path = resolve_phantom_json(paths)
    return phantom_loader.load_phantom(
        json_path=phantom_path,
        resolution_mm=resolution_mm,
        slice_idx=slice_idx,
    )


@dataclass
class PreloadedPhantom:
    """Full 3-D interpolated phantom ready for repeated slice extraction.

    Obtain via :func:`preload_phantom_for_sim`; pass to
    :func:`extract_phantom_slice` for each slice.
    """
    phantom: object          # mr0.VoxelGridPhantom, full 3-D post-interpolation
    tissue_names: list
    nz: int


def preload_phantom_for_sim(
    paths: PathConfig,
    fov_m: float = 224e-3,
    resolution_mm: float = 224.0 / 96,
) -> PreloadedPhantom:
    """Load and preprocess the phantom once (no slicing).

    Call this once before a volume loop, then use :func:`extract_phantom_slice`
    per slice instead of :func:`load_phantom_for_sim`.
    """
    import phantom_loader

    phantom, tissue_names, nz = phantom_loader.load_phantom_preloaded(
        json_path=resolve_phantom_json(paths),
        fov_mm=fov_m * 1e3,
        resolution_mm=resolution_mm,
    )
    return PreloadedPhantom(phantom=phantom, tissue_names=tissue_names, nz=nz)


def extract_phantom_slice(
    preloaded: PreloadedPhantom,
    slice_idx: Optional[int],
):
    """Extract one slice from a preloaded phantom.

    Returns the same ``(phantom, phantom_data, tissue_masks)`` tuple as
    :func:`load_phantom_for_sim`.
    """
    import phantom_loader

    return phantom_loader.slice_preloaded_phantom(
        preloaded.phantom,
        preloaded.tissue_names,
        preloaded.nz,
        slice_idx=slice_idx,
    )


# ----------------------------------------------------------------------------
#  Simulation
# ----------------------------------------------------------------------------
@contextlib.contextmanager
def _suppress_c_stdout():
    """Redirect C-level file descriptor 1 to devnull.

    compute_graph() prints >>>> Rust >>>> timing lines directly to the C
    runtime's stdout — Python's logging/redirect_stdout cannot intercept them.
    os.dup2 works at the OS level and suppresses any C/Rust write to fd 1.
    Falls back to a no-op when stdout has no real file descriptor (IDEs, etc.).
    """
    sys.stdout.flush()
    try:
        fd = sys.stdout.fileno()
    except io.UnsupportedOperation:
        yield
        return
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_fd = os.dup(fd)
    os.dup2(devnull_fd, fd)
    os.close(devnull_fd)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(old_fd, fd)
        os.close(old_fd)


def simulate_signal(
    seq_file_path: str,
    phantom_data,
    use_gpu: bool,
    *,
    gpu_max_states: int = 50000,
    gpu_min_emit: float = 1e-6,
    cpu_max_states: int = 5000,
    cpu_min_emit: float = 1e-5,
):
    """Build the MR0 graph for the .seq file and execute it.

    Returns a CPU tensor of complex signal samples.
    """
    seq0 = mr0.Sequence.import_file(seq_file_path)
    if use_gpu:
        seq0_gpu = seq0.cuda()
        phantom_gpu = phantom_data.cuda()
        with _suppress_c_stdout():
            graph = mr0.compute_graph(seq0_gpu, phantom_gpu, gpu_max_states, gpu_min_emit)
        signal = mr0.execute_graph(
            graph, seq0_gpu, phantom_gpu, print_progress=False
        ).cpu()
        del seq0_gpu, phantom_gpu
        torch.cuda.empty_cache()
    else:
        phantom_cpu = phantom_data.cpu()
        with _suppress_c_stdout():
            graph = mr0.compute_graph(seq0, phantom_cpu, cpu_max_states, cpu_min_emit)
        signal = mr0.execute_graph(graph, seq0, phantom_cpu, print_progress=False)
    return signal


# ----------------------------------------------------------------------------
#  NIfTI helpers
# ----------------------------------------------------------------------------
def affine_from_res(res: float) -> np.ndarray:
    """Diagonal NIfTI affine for an isotropic voxel of size ``res`` mm."""
    return np.array([[res, 0, 0, 0], [0, res, 0, 0], [0, 0, res, 0], [0, 0, 0, 1]])


def save_magnitude_nifti(mag_img: np.ndarray, path: str, res: float) -> None:
    """Save a (Ny, Nx) magnitude image as a single-slice NIfTI volume."""
    affine = affine_from_res(res)
    vol = np.asarray(mag_img[:, :, np.newaxis], dtype=np.float32)
    nib.save(nib.Nifti1Image(vol, affine), path)


# ----------------------------------------------------------------------------
#  Diffusion post-processing
# ----------------------------------------------------------------------------
def compute_trace_dwi(mag_images: np.ndarray) -> np.ndarray:
    """Geometric mean across diffusion directions (trace-DWI estimator).

    ``mag_images`` is shaped ``(n_b, n_dirs, Ny, Nx)``. The floor is
    data-scaled so that a single noise-floor voxel in one direction at high
    b does not drag the geometric mean to ~0 and produce a spurious
    fast-decay.
    """
    noise_floor = 1e-3 * float(mag_images[0].max())
    return np.exp(np.mean(np.log(np.maximum(mag_images, noise_floor)), axis=1))


def save_tissue_masks(
    tissue_masks: dict,
    masks_dir: str,
    phantom_name: str,
) -> None:
    """Stack the tissue-mask dict into ``(n_tissues, Ny, Nx)`` and save as .npy."""
    stacked_masks = torch.stack(list(tissue_masks.values()), dim=0).numpy()
    os.makedirs(masks_dir, exist_ok=True)
    np.save(os.path.join(masks_dir, f"{phantom_name}-tissue_masks.npy"), stacked_masks)


# ----------------------------------------------------------------------------
#  Per-tissue MAE
# ----------------------------------------------------------------------------
def _mask_to_2d_bool(mask: object) -> np.ndarray:
    """Coerce a tissue-mask tensor (torch or ndarray) to a 2-D boolean array."""
    if hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()
    return np.squeeze(np.asarray(mask)).astype(bool)


def combined_tissue_mask(tissue_masks: dict) -> np.ndarray:
    """Union of all tissue masks → the 'combined' phantom region."""
    if "combined" in tissue_masks:
        return _mask_to_2d_bool(tissue_masks["combined"])
    union: np.ndarray | None = None
    for mask in tissue_masks.values():
        mask_2d = _mask_to_2d_bool(mask)
        union = mask_2d if union is None else (union | mask_2d)
    return union if union is not None else np.zeros((0, 0), dtype=bool)


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

    def _mae_in(mask: np.ndarray) -> tuple[float, int]:
        if mask.shape != est.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match map shape {est.shape}."
            )
        valid_voxels = mask
        if exclude_zeros:
            valid_voxels = valid_voxels & (est != 0) & (ref != 0)
        n_valid = int(valid_voxels.sum())
        if n_valid == 0:
            return float("nan"), 0
        return float(np.mean(np.abs(est[valid_voxels] - ref[valid_voxels]))), n_valid

    mae_per_tissue: dict[str, float] = {}
    n_voxels_per_tissue: dict[str, int] = {}
    for tissue_name, mask in tissue_masks.items():
        mask_2d = _mask_to_2d_bool(mask)
        mae_value, n_voxels = _mae_in(mask_2d)
        mae_per_tissue[tissue_name] = mae_value
        n_voxels_per_tissue[tissue_name] = n_voxels

    combined_mask = combined_tissue_mask(tissue_masks)
    mae_total, n_voxels_total = _mae_in(combined_mask)

    return {
        "per_tissue": mae_per_tissue,
        "total": mae_total,
        "n_voxels_per_tissue": n_voxels_per_tissue,
        "n_voxels_total": n_voxels_total,
    }


def phantom_map_to_2d(parameter_map: object) -> np.ndarray:
    """Squeeze a phantom map (torch tensor or ndarray) to a 2-D numpy array."""
    if hasattr(parameter_map, "cpu"):
        parameter_map = parameter_map.cpu().numpy()
    return np.squeeze(np.asarray(parameter_map))


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
