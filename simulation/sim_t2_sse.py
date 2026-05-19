"""Single-shot EPI SE T2 relaxometry pipeline (refactor of qMRI_t2relax.py)."""
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
    system_type=None,
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

    reconstructed_b0 = []  # one complex image per TE
    for te in TEs:
        print(f"[T2-SSE] TE={te} ms", flush=True, end="\r")
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

        op = get_operator(
            backend_name="cufinufft",
            samples=traj_dir0,
            shape=(Ny, Nx),
            n_coils=1,
            density=True,
        )
        sig = torch.from_numpy(dir_signal[0]).to(torch.complex64)
        img_complex = op.adj_op(sig.flatten()).squeeze().cpu().numpy().T
        reconstructed_b0.append(img_complex)

        save_magnitude_nifti(
            np.abs(img_complex),
            os.path.join(paths.t2_img_dir, f"{name}.nii.gz"),
            res,
        )

    reconstructed_b0 = np.array(reconstructed_b0)  # (n_TEs, Ny, Nx)
    mag_b0 = np.abs(reconstructed_b0)

    t2_nlls, _ = create_t2_map(mag_b0, TEs, method="nlls")
    t2_ll, _ = create_t2_map(mag_b0, TEs, method="loglinear")

    t2_nlls_oriented = t2_nlls / 1000.0
    if save_slice_npy:
        out_path = os.path.join(
            paths.volumes_dir, f"{paths.phantom_name}-T2_SSE_{blip_tag}.npy"
        )
        np.save(out_path, t2_nlls_oriented)
        print(f"[T2-SSE] saved {out_path}")

    # MAE per tissue + combined. phantom.T2 is in seconds; t2_nlls is in ms.
    ref_T2 = phantom_map_to_2d(phantom.T2)
    est_T2 = t2_nlls / 1000.0
    mae = compute_mae_per_tissue(est_T2, ref_T2, tissue_masks)

    return {
        "t2_nlls": t2_nlls_oriented,
        "t2_loglinear": t2_ll,
        "TEs": TEs,
        "reference_map": ref_T2,
        "tissue_masks": tissue_masks,
        "mae_per_tissue": mae["per_tissue"],
        "mae_total": mae["total"],
        "mae_n_voxels_per_tissue": mae["n_voxels_per_tissue"],
        "mae_n_voxels_total": mae["n_voxels_total"],
    }
