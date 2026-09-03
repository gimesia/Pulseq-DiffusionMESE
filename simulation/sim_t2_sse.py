"""Single-shot EPI spin-echo T2-relaxometry simulation pipeline.

Pipeline
--------
For each TE in ``TEs``:

1. Build an ``EPIDiffusionSEPulseqSeq`` at b=0 with the requested TE
   (PGSE timing is kept fixed so the only acquisition difference between
   echo times is the spin-echo refocusing position).
2. Bloch-simulate, drop the calibration prefix, reshape the single-direction
   k-space, and reconstruct via NUFFT (cufinufft / finufft).
3. After all TEs have been collected, fit ``S(TE) = S0 * exp(-TE/T2)``
   per voxel with NLLS (warm-started by a log-linear fit).

One NIfTI per TE plus the final T2 NLLS map are saved.

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
    simulate_signal,
)


def run_t2_sse(
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
    b_value: int = 0,
    blip_down: bool = False,
    small_delta: float = 0.018,
    big_DELTA: float = 0.03,
    system_type: Optional[object] = None,
    use_gpu: Optional[bool] = None,
    save_slice_npy: bool = True,
    phantom_slice: Optional[tuple] = None,
) -> dict:
    """Run the single-shot SE EPI T2-relaxometry simulation (b=0 series)."""
    paths.ensure_dirs()
    ensure_seq_path_on_syspath(paths.seq_lib_path)
    logger = make_quiet_logger()

    from EPIDiffusionSEPulseqSeq import EPIDiffusionSEPulseqSeq
    from utils import SystemLimitType
    from mrinufft import get_operator
    from utils_relaxometry import create_t2_map

    if system_type is None:
        system_type = SystemLimitType.EXTREME
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()

    TEs = np.asarray(TEs, dtype=int)
    slice_thickness = res * 1e-3
    Nx = Ny = int(fov / slice_thickness)
    blip_tag = "blipdown" if blip_down else "blipup"

    if phantom_slice is not None:
        phantom, phantom_data, tissue_masks = phantom_slice
    else:
        phantom, phantom_data, tissue_masks = load_phantom_for_sim(paths, res, slice_idx)

    # Empty slice: return zero-filled maps so the volume runner stacks cleanly.
    if not any(bool(mask.any()) for mask in tissue_masks.values()):
        zeros_2d = np.zeros((Ny, Nx), dtype=np.float64)
        return {
            "t2_nlls": zeros_2d.copy(),
            "TEs": TEs,
            "weighted_images": {},
            "phantom": phantom,
        }

    reconstructed_b0 = []  # one complex image per TE
    for te in TEs:
        print(f"[T2-SSE]  b={b_value} s/mm²  TE={te} ms", flush=True, end="\r")
        name = f"DiffSE-TE{int(te)}-{blip_tag}"
        seq = EPIDiffusionSEPulseqSeq(
            name=name,
            resolution=res,
            Nx=Nx,
            Ny=Ny,
            fov=fov,
            slice_thickness=slice_thickness,
            TE=int(te),
            TR=TR,
            b_value=b_value,
            b_directions=1,
            b_0_frequency=0,
            save_dir=paths.sequences_dir,
            save_name=name,
            v141_compat=True,
            small_delta=small_delta,
            big_DELTA=big_DELTA,
            system_type=system_type,
            calibration_readout=True,
            blip_down=blip_down,
            logger=logger,
            fit_epi=True,
        )
        seq.write()

        signal = simulate_signal(
            os.path.join(paths.sequences_dir, f"{name}.seq"),
            phantom_data,
            use_gpu,
            gpu_max_states=20000,
            gpu_min_emit=1e-5,
            cpu_max_states=2000,
            cpu_min_emit=1e-4,
        )

        samples_per_cal = int(3 * seq.adc.num_samples)
        samples_per_dir = int(seq.Ny * seq.partial_fourier_factor) * seq.adc.num_samples
        epi_signal = signal[samples_per_cal:].squeeze()
        dir_signal = np.array(
            [
                epi_signal[i * samples_per_dir : (i + 1) * samples_per_dir].reshape(
                    int(seq.Ny * seq.partial_fourier_factor), seq.adc.num_samples
                )
                for i in range(len(seq.b_directions))
            ]
        )

        k_traj_adc, _, _, _, _ = seq.seq.calculate_kspace()
        kx_norm = k_traj_adc[0] * fov / Nx
        ky_norm = k_traj_adc[1] * fov / Ny
        traj = np.stack([kx_norm, ky_norm], axis=-1)[samples_per_cal:]
        traj_dir0 = traj[:samples_per_dir]

        nufft_op = get_operator(
            backend_name="cufinufft" if use_gpu else "finufft",
            samples=traj_dir0,
            shape=(Ny, Nx),
            n_coils=1,
            density=True,
        )
        kspace_tensor = torch.from_numpy(dir_signal[0]).to(torch.complex64)
        # No `.T`: an untransposed nufft_op.adj_op() output already lands
        # on the same array grid as the phantom's own parameter maps
        # (verified with a synthetic marker through this exact
        # trajectory/operator pair). Transposing it here would silently
        # misalign the saved/returned map relative to phantom.T2 — a
        # reflection that rigid registration (see
        # utils_sim_lib.mae_after_registration) cannot undo.
        img_complex = nufft_op.adj_op(kspace_tensor.flatten()).squeeze().cpu().numpy()
        reconstructed_b0.append(img_complex)

    reconstructed_b0 = np.array(reconstructed_b0)  # (n_TEs, Ny, Nx)
    mag_b0 = np.abs(reconstructed_b0)

    weighted_images = {
        f"T2w_TE{int(te)}_{blip_tag}": mag_b0[i]
        for i, te in enumerate(TEs)
    }

    t2_nlls, _ = create_t2_map(mag_b0, TEs, method="nlls")

    t2_nlls_oriented = t2_nlls / 1000.0
    if save_slice_npy:
        out_path = os.path.join(
            paths.volumes_dir, f"{paths.phantom_name}-T2_SSE_{blip_tag}.nii.gz"
        )
        save_magnitude_nifti(t2_nlls_oriented, out_path, res)
        print(f"[T2-SSE] saved {out_path}")

    return {
        "t2_nlls": t2_nlls_oriented,
        "TEs": TEs,
        "weighted_images": weighted_images,
        "phantom": phantom,
    }
