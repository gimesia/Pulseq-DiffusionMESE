"""Multishot SE T2 relaxometry pipeline (refactor of qMRI_t2relax_multishot_se.py)."""
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import torch

from utils_sim_lib import (
    PathConfig,
    compute_mae_per_tissue,
    ensure_seq_path_on_syspath,
    load_phantom_for_sim,
    make_quiet_logger,
    phantom_map_to_2d,
    save_magnitude_nifti,
    save_tissue_masks,
    simulate_signal,
)


def run_t2_multishot(
    paths: PathConfig,
    *,
    slice_idx: Optional[int] = None,
    fov: float = 224e-3,
    res: float = 2.33333333,
    TEs: Sequence[int] = (
        65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 123, 125, 128,
        130, 133, 135, 138, 140, 143, 145, 148, 150, 153, 158, 163, 168, 173,
        178, 181, 183, 186, 188, 191, 193, 196, 198, 201, 203, 206, 208, 211,
        216, 221, 226, 231, 236, 241, 246, 251, 256, 261, 266,
    ),
    TR: int = 5000,
    ETL: int = 1,
    system_type=None,
    use_gpu: Optional[bool] = None,
    save_slice_npy: bool = True,
    phantom_slice: Optional[tuple] = None,
) -> dict:
    """Run the multishot SE T2-relaxometry simulation (Cartesian → iFFT)."""
    paths.ensure_dirs()
    ensure_seq_path_on_syspath(paths.seq_lib_path)
    logger = make_quiet_logger()

    from DiffusionSEMultishotPulseqSeq import DiffusionSEMultishotPulseqSeq
    from utils import SystemLimitType
    from utils_simulation import fft_reconstruct_image
    from utils_relaxometry import create_t2_map

    if system_type is None:
        system_type = SystemLimitType.EXTREME
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()

    TEs = np.asarray(TEs, dtype=int)
    slice_thickness = res * 1e-3
    Nx = Ny = int(fov / slice_thickness)

    if phantom_slice is not None:
        phantom, phantom_data, tissue_masks = phantom_slice
    else:
        phantom, phantom_data, tissue_masks = load_phantom_for_sim(paths, res, slice_idx)

    reconstructed_images = []
    for te in TEs:
        print(f"[T2-MS] TE={te} ms")
        seq = DiffusionSEMultishotPulseqSeq(
            name="DiffusionSEMultishot",
            fov=fov,
            Nx=Nx,
            Ny=Ny,
            slice_thickness=slice_thickness,
            TR=TR,
            TE=int(te),
            ETL=ETL,
            save_dir=paths.sequences_dir,
            v141_compat=True,
            system_type=system_type,
            logger=logger,
        )
        seq.build_seq()
        seq.write()
        seq_filename = seq.get_save_filename()

        signal = simulate_signal(
            os.path.join(paths.sequences_dir, seq_filename),
            phantom_data,
            use_gpu,
            gpu_max_states=20000,
            gpu_min_emit=1e-5,
            cpu_max_states=2000,
            cpu_min_emit=1e-4,

        )

        kspace = signal.numpy().reshape(Ny, Nx)
        img_mag, _ = fft_reconstruct_image(kspace, use_gpu=use_gpu)
        img_mag = img_mag.squeeze()
        reconstructed_images.append(img_mag)

        save_magnitude_nifti(
            img_mag,
            os.path.join(paths.t2_img_dir, f"MultiShotSE-TE{int(te)}.nii.gz"),
            res,
        )

    images_stack = np.array(reconstructed_images)  # (n_TEs, Ny, Nx)
    t2_nlls, _ = create_t2_map(images_stack, TEs, method="nlls")
    t2_ll, _ = create_t2_map(images_stack, TEs, method="loglinear")

    t2_nlls_oriented = np.fliplr(np.rot90(t2_nlls, 0)) / 1000.0
    if save_slice_npy:
        np.save(
            os.path.join(paths.volumes_dir, f"{paths.phantom_name}-T2_multishot_se.npy"),
            t2_nlls_oriented,
        )
    save_tissue_masks(tissue_masks, paths.masks_dir, paths.phantom_name)
    print(f"[T2-MS] saved T2 map to {paths.volumes_dir}")

    ref_T2 = phantom_map_to_2d(phantom.T2)
    est_T2 = t2_nlls / 1000.0
    mae = compute_mae_per_tissue(est_T2, ref_T2, tissue_masks)

    return {
        "t2_nlls": t2_nlls,
        "t2_loglinear": t2_ll,
        "TEs": TEs,
        "reference_map": ref_T2,
        "tissue_masks": tissue_masks,
        "mae_per_tissue": mae["per_tissue"],
        "mae_total": mae["total"],
        "mae_n_voxels_per_tissue": mae["n_voxels_per_tissue"],
        "mae_n_voxels_total": mae["n_voxels_total"],
    }
