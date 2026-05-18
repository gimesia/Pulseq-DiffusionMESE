"""Triple SE EPI ADC pipeline (refactor of qMRI_adc_triple_se.py)."""
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import torch

from qmri_sim_lib import (
    PathConfig,
    compute_mae_per_tissue,
    compute_trace_dwi,
    ensure_seq_path_on_syspath,
    load_phantom_for_sim,
    make_quiet_logger,
    phantom_map_to_2d,
    save_magnitude_nifti,
    simulate_signal,
)


def run_adc_triple(
    paths: PathConfig,
    *,
    slice_idx: Optional[int] = None,
    fov: float = 224e-3,
    res: float = 2.33333333,
    b_values: Sequence[int] = tuple(range(0, 2001, 100)),
    TE: int = 100,
    TR: int = 5000,
    b_directions: int = 6,
    blip_down: bool = True,
    small_delta: Optional[float] = None,
    big_DELTA: Optional[float] = None,
    system_type=None,
    use_gpu: Optional[bool] = None,
) -> dict:
    """Run the triple SE EPI diffusion / ADC simulation.

    TE1 is fixed across b-values; TE2 and TE3 are auto-computed by the sequence.
    """
    paths.ensure_dirs()
    ensure_seq_path_on_syspath(paths.seq_lib_path)
    logger = make_quiet_logger()

    from EPIDiffusionTripleSEPulseqSeq import EPIDiffusionTripleSEPulseqSeq
    from utils import SystemLimitType
    from mrinufft import get_operator
    from utils_diffusion import create_adc_map, create_dti_maps

    if system_type is None:
        system_type = SystemLimitType.EXTREME
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()

    b_values = np.asarray(b_values, dtype=int)
    slice_thickness = res * 1e-3
    Nx = Ny = int(fov / slice_thickness)
    blip_tag = "blipdown" if blip_down else "blipup"

    phantom, phantom_data, tissue_masks = load_phantom_for_sim(paths, res, slice_idx)

    all_echo_images = []
    te1_ms = te2_ms = te3_ms = None
    last_seq = None

    for b_value in b_values:
        print(f"[ADC-3SE] b={b_value} s/mm²")
        name = f"DiffTripleSE-b{int(b_value)}"
        seq = EPIDiffusionTripleSEPulseqSeq(
            name=name,
            resolution=res,
            Nx=Nx,
            Ny=Ny,
            fov=fov,
            slice_thickness=slice_thickness,
            TE=TE,
            TR=TR,
            b_value=int(b_value),
            b_directions=b_directions,
            b_0_frequency=0,
            save_dir=paths.sequences_dir,
            save_name=name,
            v141_compat=True,
            small_delta=small_delta,
            big_DELTA=big_DELTA,
            system_type=system_type,
            calibration_readout=True,
            blip_down=blip_down,
            uniform_spoiler_directions=False,
            uniform_spoiler_areas=False,
            phase_cycling=True,
            partial_fourier_factor=1,
            logger=logger,
        )
        seq.write()
        last_seq = seq

        if te1_ms is None:
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
            print_progress=use_gpu,
        )

        # Per-echo k-space split
        samples_per_cal = int(3 * seq.adc.num_samples)
        samples_epi1 = int(seq.Ny * seq.partial_fourier_factor) * seq.adc.num_samples
        samples_epi2 = seq.Ny * seq.adc.num_samples
        samples_epi3 = seq.Ny * seq.adc.num_samples
        samples_per_dir = samples_epi1 + samples_epi2 + samples_epi3
        n_dirs = len(seq.b_directions)

        epi_signal = signal[samples_per_cal:].squeeze()
        echo1_ks, echo2_ks, echo3_ks = [], [], []
        for d in range(n_dirs):
            base = d * samples_per_dir
            echo1_ks.append(
                epi_signal[base : base + samples_epi1].reshape(
                    int(seq.Ny * seq.partial_fourier_factor), seq.adc.num_samples
                )
            )
            echo2_ks.append(
                epi_signal[base + samples_epi1 : base + samples_epi1 + samples_epi2]
                .reshape(seq.Ny, seq.adc.num_samples)
            )
            echo3_ks.append(
                epi_signal[base + samples_epi1 + samples_epi2 : base + samples_per_dir]
                .reshape(seq.Ny, seq.adc.num_samples)
            )

        # Trajectory split (constant across directions)
        k_traj_adc, _, _, _, _ = seq.seq.calculate_kspace()
        kx_norm = k_traj_adc[0] * fov / Nx
        ky_norm = k_traj_adc[1] * fov / Ny
        traj = np.stack([kx_norm, ky_norm], axis=-1)
        d0 = samples_per_cal
        traj_epi1 = traj[d0 : d0 + samples_epi1]
        traj_epi2 = traj[d0 + samples_epi1 : d0 + samples_epi1 + samples_epi2]
        traj_epi3 = traj[d0 + samples_epi1 + samples_epi2 : d0 + samples_per_dir]

        # One NUFFT operator per echo, reused across directions
        img_size = (Ny, Nx)
        op_e1 = get_operator(backend_name="finufft", samples=traj_epi1, shape=img_size,
                             n_coils=1, density=True)
        op_e2 = get_operator(backend_name="finufft", samples=traj_epi2, shape=img_size,
                             n_coils=1, density=True)
        op_e3 = get_operator(backend_name="finufft", samples=traj_epi3, shape=img_size,
                             n_coils=1, density=True)

        dir_echo_images = []
        for d in range(n_dirs):
            imgs = []
            for ksp, op in (
                (echo1_ks[d], op_e1),
                (echo2_ks[d], op_e2),
                (echo3_ks[d], op_e3),
            ):
                sig_t = torch.from_numpy(np.array(ksp)).to(torch.complex64)
                imgs.append(op.adj_op(sig_t.flatten()).squeeze().cpu().numpy())
            dir_echo_images.append(np.stack(imgs, axis=0))  # (3, Ny, Nx)

            for echo_idx, te_ms_echo in enumerate([te1_ms, te2_ms, te3_ms]):
                echo_name = (
                    f"DiffTripleSE-b{int(b_value)}-dir{d}-TE{int(te_ms_echo)}-{blip_tag}.nii.gz"
                )
                save_magnitude_nifti(
                    np.abs(imgs[echo_idx]),
                    os.path.join(paths.diff_img_dir, echo_name),
                    res,
                )

        all_echo_images.append(np.stack(dir_echo_images, axis=0))  # (n_dirs, 3, Ny, Nx)

    all_echo_images = np.array(all_echo_images)  # (n_b, n_dirs, 3, Ny, Nx)
    mag_images = np.abs(all_echo_images)
    mag_echo1 = mag_images[:, :, 0, :, :]
    mag_images_combined = mag_images.mean(axis=2)

    trace_dwi = compute_trace_dwi(mag_echo1)
    adc_nlls, _ = create_adc_map(trace_dwi, b_values, method="nlls")
    adc_ll, _ = create_adc_map(trace_dwi, b_values, method="loglinear")
    fa_map, md_map, eigvals_map, dti_s0_map = create_dti_maps(
        mag_images_combined, b_values, last_seq.b_directions
    )

    adc_nlls_oriented = np.rot90(adc_nlls, -1) * 1e3
    out_path = os.path.join(
        paths.volumes_dir, f"{paths.phantom_name}-ADC_MSE_{blip_tag}.npy"
    )
    np.save(out_path, adc_nlls_oriented)
    print(f"[ADC-3SE] saved {out_path}")

    ref_D = phantom_map_to_2d(phantom.D)
    est_adc = adc_nlls * 1e3
    mae = compute_mae_per_tissue(est_adc, ref_D, tissue_masks)

    return {
        "adc_nlls": adc_nlls,
        "adc_loglinear": adc_ll,
        "fa_map": fa_map,
        "md_map": md_map,
        "echo_TEs_ms": (te1_ms, te2_ms, te3_ms),
        "b_values": b_values,
        "reference_map": ref_D,
        "tissue_masks": tissue_masks,
        "mae_per_tissue": mae["per_tissue"],
        "mae_total": mae["total"],
        "mae_n_voxels_per_tissue": mae["n_voxels_per_tissue"],
        "mae_n_voxels_total": mae["n_voxels_total"],
    }
