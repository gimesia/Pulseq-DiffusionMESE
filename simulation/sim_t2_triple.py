"""Triple SE EPI T2 relaxometry pipeline (refactor of qMRI_t2relax_triple_se.py)."""
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


def run_t2_triple(
    paths: PathConfig,
    *,
    slice_idx: Optional[int] = None,
    fov: float = 224e-3,
    res: float = 2.33333333,
    TE1_values: Sequence[int] = tuple(range(65, 155, 5)),
    TR: int = 5000,
    b_value: int = 0,
    blip_down: bool = True,
    small_delta: float = 0.018,
    big_DELTA: float = 0.03,
    system_type=None,
    use_gpu: Optional[bool] = None,
    save_slice_npy: bool = True,
    phantom_slice: Optional[tuple] = None,
) -> dict:
    """Run the triple SE EPI T2-relaxometry simulation.

    Each TR yields 3 echoes (TE1, auto-computed TE2, TE3), so ~15 TR values
    produce ~45 (TE, image) datapoints for the T2 fit.
    """
    paths.ensure_dirs()
    ensure_seq_path_on_syspath(paths.seq_lib_path)
    logger = make_quiet_logger()

    from EPIDiffusionTripleSEPulseqSeq import EPIDiffusionTripleSEPulseqSeq
    from utils import SystemLimitType
    from mrinufft import get_operator
    from utils_relaxometry import create_t2_map

    if system_type is None:
        system_type = SystemLimitType.EXTREME
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()

    TE1_values = np.asarray(TE1_values, dtype=int)
    slice_thickness = res * 1e-3
    Nx = Ny = int(fov / slice_thickness)
    blip_tag = "blipdown" if blip_down else "blipup"

    if phantom_slice is not None:
        phantom, phantom_data, tissue_masks = phantom_slice
    else:
        phantom, phantom_data, tissue_masks = load_phantom_for_sim(paths, res, slice_idx)

    all_echo_images: list[np.ndarray] = []
    all_echo_tes: list[float] = []

    for te1 in TE1_values:
        print(f"[T2-3SE] TE1={te1} ms", flush=True, end="\r")
        name = f"DiffTripleSE-TE1-{int(te1)}"
        seq = EPIDiffusionTripleSEPulseqSeq(
            name=name,
            resolution=res,
            Nx=Nx,
            Ny=Ny,
            fov=fov,
            slice_thickness=slice_thickness,
            TE=int(te1),
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
            uniform_spoiler_directions=False,
            uniform_spoiler_areas=True,
            phase_cycling=False,
            blip_down=blip_down,
            logger=logger,
            fit_epi=True,
        )
        seq.write()

        te1_ms = seq.TE * 1e3
        te2_ms = seq.TE2 * 1e3
        te3_ms = seq.TE3 * 1e3

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
        samples_epi1 = int(seq.Ny * seq.partial_fourier_factor) * seq.adc.num_samples
        samples_epi2 = seq.Ny * seq.adc.num_samples
        samples_epi3 = seq.Ny * seq.adc.num_samples
        samples_per_dir = samples_epi1 + samples_epi2 + samples_epi3

        epi_signal = signal[samples_per_cal:].squeeze()
        echo1_ksp = epi_signal[:samples_epi1].reshape(
            int(seq.Ny * seq.partial_fourier_factor), seq.adc.num_samples
        )
        echo2_ksp = epi_signal[samples_epi1 : samples_epi1 + samples_epi2].reshape(
            seq.Ny, seq.adc.num_samples
        )
        echo3_ksp = epi_signal[samples_epi1 + samples_epi2 : samples_per_dir].reshape(
            seq.Ny, seq.adc.num_samples
        )

        k_traj_adc, _, _, _, _ = seq.seq.calculate_kspace()
        kx_norm = k_traj_adc[0] * fov / Nx
        ky_norm = k_traj_adc[1] * fov / Ny
        traj = np.stack([kx_norm, ky_norm], axis=-1)
        traj_epi1 = traj[samples_per_cal : samples_per_cal + samples_epi1]
        traj_epi2 = traj[
            samples_per_cal + samples_epi1 : samples_per_cal + samples_epi1 + samples_epi2
        ]
        traj_epi3 = traj[
            samples_per_cal + samples_epi1 + samples_epi2 : samples_per_cal + samples_per_dir
        ]

        echo_imgs = []
        for ksp, traj_echo in (
            (echo1_ksp, traj_epi1),
            (echo2_ksp, traj_epi2),
            (echo3_ksp, traj_epi3),
        ):
            op = get_operator(
                backend_name="cufinufft" if use_gpu else "finufft",
                samples=traj_echo,
                shape=(Ny, Nx),
                n_coils=1,
                density=True,
            )
            sig_t = ksp.to(torch.complex64)
            echo_imgs.append(op.adj_op(sig_t.flatten()).squeeze().cpu().numpy().T)

        for img, te_ms in zip(echo_imgs, (te1_ms, te2_ms, te3_ms)):
            mag_img = np.abs(img)
            all_echo_images.append(mag_img)
            all_echo_tes.append(te_ms)
            nii = f"DiffTripleSE-TE{int(te_ms)}-{blip_tag}.nii.gz"
            save_magnitude_nifti(mag_img, os.path.join(paths.t2_img_dir, nii), res)

    order = np.argsort(all_echo_tes)
    images_stack = np.array(all_echo_images)[order]
    te_sorted = np.array(all_echo_tes)[order]

    t2_nlls, _ = create_t2_map(images_stack, te_sorted, method="nlls")
    t2_ll, _ = create_t2_map(images_stack, te_sorted, method="loglinear")

    t2_nlls_oriented = t2_nlls / 1000.0
    if save_slice_npy:
        out_path = os.path.join(
            paths.volumes_dir, f"{paths.phantom_name}-T2_MSE_{blip_tag}.npy"
        )
        np.save(out_path, t2_nlls_oriented)
        print(f"[T2-3SE] saved {out_path}")

    ref_T2 = phantom_map_to_2d(phantom.T2)
    est_T2 = t2_nlls / 1000.0
    mae = compute_mae_per_tissue(est_T2, ref_T2, tissue_masks)

    return {
        "t2_nlls": t2_nlls_oriented,
        "t2_loglinear": t2_ll,
        "TEs": te_sorted,
        "reference_map": ref_T2,
        "tissue_masks": tissue_masks,
        "mae_per_tissue": mae["per_tissue"],
        "mae_total": mae["total"],
        "mae_n_voxels_per_tissue": mae["n_voxels_per_tissue"],
        "mae_n_voxels_total": mae["n_voxels_total"],
    }
