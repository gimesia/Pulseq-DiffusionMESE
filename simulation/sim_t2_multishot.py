"""Multishot SE (Cartesian FSE) T2-relaxometry simulation pipeline.

Pipeline
--------
For each TE in ``TEs``:

1. Build a ``DiffusionSEMultishotPulseqSeq`` at b=0 with the requested TE.
2. Bloch-simulate; reshape the (Ny, Nx) k-space and inverse-FFT to image.
3. After all TEs, fit ``S(TE) = S0 * exp(-TE/T2)`` per voxel with NLLS.

Cartesian Nyquist sampling means plain inverse FFT is sufficient — no NUFFT.

Author      : Aron Gimesi <aron.gimesi@tecnico.ulisboa.pt>
Affiliation : Instituto Superior Técnico | MSCA-DN IQ-BRAIN
Date        : 2026
Context     : ESMRMB 2026 — Pulseq DiffusionMESE showcase

Funding acknowledgement (mandatory):
    IQ-BRAIN is funded by the European Union (MSCA Doctoral Network,
    December 2024–November 2028, Grant Agreement No. 101169519).
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import torch

from utils_sim_lib import (
    PathConfig,
    ensure_seq_path_on_syspath,
    load_phantom_for_sim,
    make_quiet_logger,
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
    system_type: Optional[object] = None,
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

    # Empty slice: return zero-filled maps for the volume runner.
    if not any(bool(mask.any()) for mask in tissue_masks.values()):
        zeros_2d = np.zeros((Ny, Nx), dtype=np.float64)
        return {
            "t2_nlls": zeros_2d.copy(),
            "TEs": TEs,
            "weighted_images": {},
            "phantom": phantom,
        }

    reconstructed_images = []
    for te in TEs:
        print(f"[T2-MS]  b=0 s/mm²  TE={te} ms", flush=True, end="\r")
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

    images_stack = np.array(reconstructed_images)  # (n_TEs, Ny, Nx)
    weighted_images = {
        f"T2w_TE{int(te)}": images_stack[i]
        for i, te in enumerate(TEs)
    }
    t2_nlls, _ = create_t2_map(images_stack, TEs, method="nlls")

    t2_nlls_oriented = t2_nlls / 1000.0
    if save_slice_npy:
        save_magnitude_nifti(
            t2_nlls_oriented,
            os.path.join(paths.volumes_dir, f"{paths.phantom_name}-T2_multishot_se.nii.gz"),
            res,
        )
    save_tissue_masks(tissue_masks, paths.masks_dir, paths.phantom_name)
    # print(f"[T2-MS] saved T2 map to {paths.volumes_dir}")

    return {
        "t2_nlls": t2_nlls_oriented,
        "TEs": TEs,
        "weighted_images": weighted_images,
        "phantom": phantom,
    }
