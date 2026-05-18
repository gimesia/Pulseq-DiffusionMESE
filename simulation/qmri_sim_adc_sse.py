"""Single-shot EPI Diffusion SE ADC pipeline (refactor of qMRI_adc.py)."""
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import torch

from qmri_sim_lib import (
    PathConfig,
    affine_from_res,
    compute_mae_per_tissue,
    compute_trace_dwi,
    ensure_seq_path_on_syspath,
    load_phantom_for_sim,
    make_quiet_logger,
    phantom_map_to_2d,
    save_magnitude_nifti,
    simulate_signal,
)


def run_adc_sse(
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
    """Run the single-shot SE EPI diffusion / ADC simulation.

    The TE is fixed across b-values (Stejskal-Tanner: identical T2 weighting).
    Saves one NIfTI per (b-value, direction) plus the final ADC NLLS map.
    """
    paths.ensure_dirs()
    ensure_seq_path_on_syspath(paths.seq_lib_path)
    logger = make_quiet_logger()

    from EPIDiffusionSEPulseqSeq import EPIDiffusionSEPulseqSeq
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

    all_echo_images = []  # one entry per b-value: (n_dirs, n_echoes, Ny, Nx)
    last_seq = None

    for b_value in b_values:
        print(f"[ADC-SSE] b={b_value} s/mm²  TE={TE} ms")
        name = f"DiffSE-b{int(b_value)}-TE{TE}"
        seq = EPIDiffusionSEPulseqSeq(
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
            logger=logger,
        )
        seq.write()
        last_seq = seq

        signal = simulate_signal(
            os.path.join(paths.sequences_dir, f"{name}.seq"),
            phantom_data,
            use_gpu,
            print_progress=use_gpu,
        )

        # Separate k-spaces (skip calibration prefix)
        samples_per_cal = int(3 * seq.adc.num_samples)
        samples_per_dir = int(seq.Ny * seq.partial_fourier_factor) * seq.adc.num_samples
        epi_signal = signal[samples_per_cal:].squeeze()

        n_dirs = len(seq.b_directions)
        dir_signal = np.array(
            [
                epi_signal[i * samples_per_dir : (i + 1) * samples_per_dir].reshape(
                    int(seq.Ny * seq.partial_fourier_factor), seq.adc.num_samples
                )
                for i in range(n_dirs)
            ]
        )

        # k-space trajectory (normalised to ±0.5)
        k_traj_adc, _, _, _, _ = seq.seq.calculate_kspace()
        kx_norm = k_traj_adc[0] * fov / Nx
        ky_norm = k_traj_adc[1] * fov / Ny
        traj = np.stack([kx_norm, ky_norm], axis=-1)
        dir_trajs = np.array(
            [
                traj[samples_per_cal:][i * samples_per_dir : (i + 1) * samples_per_dir]
                for i in range(n_dirs)
            ]
        )

        # NUFFT reconstruction (per direction)
        recon = []
        for i in range(n_dirs):
            op = get_operator(
                backend_name="finufft",
                samples=dir_trajs[i],
                shape=(Ny, Nx),
                n_coils=1,
                density=True,
            )
            sig = torch.from_numpy(dir_signal[i]).to(torch.complex64)
            img_complex = op.adj_op(sig.flatten()).squeeze().cpu().numpy()
            recon.append(img_complex)

            nii_name = (
                f"DiffSE-b{int(b_value)}-dir{i}-TE{int(TE)}-{blip_tag}.nii.gz"
            )
            save_magnitude_nifti(
                np.abs(img_complex), os.path.join(paths.diff_img_dir, nii_name), res
            )

        # (n_dirs, n_echoes=1, Ny, Nx)
        all_echo_images.append(np.stack(recon, axis=0)[:, None, :, :])

    all_echo_images = np.array(all_echo_images)  # (n_b, n_dirs, 1, Ny, Nx)
    mag_images = np.abs(all_echo_images)
    mag_images_combined = mag_images.mean(axis=2)  # (n_b, n_dirs, Ny, Nx)

    trace_dwi = compute_trace_dwi(mag_images_combined)
    adc_nlls, _ = create_adc_map(trace_dwi, b_values, method="nlls")
    adc_ll, _ = create_adc_map(trace_dwi, b_values, method="loglinear")
    fa_map, md_map, eigvals_map, dti_s0_map = create_dti_maps(
        mag_images_combined, b_values, last_seq.b_directions
    )

    adc_nlls_oriented = np.rot90(adc_nlls, -1) * 1e3
    out_path = os.path.join(
        paths.volumes_dir, f"{paths.phantom_name}-ADC_SSE_{blip_tag}.npy"
    )
    np.save(out_path, adc_nlls_oriented)
    print(f"[ADC-SSE] saved {out_path}")

    # MAE per tissue + combined. Both maps in x10^-3 mm^2/s units.
    ref_D = phantom_map_to_2d(phantom.D)
    est_adc = adc_nlls * 1e3
    mae = compute_mae_per_tissue(est_adc, ref_D, tissue_masks)

    return {
        "adc_nlls": adc_nlls,
        "adc_loglinear": adc_ll,
        "fa_map": fa_map,
        "md_map": md_map,
        "mag_images": mag_images_combined,
        "b_values": b_values,
        "reference_map": ref_D,
        "tissue_masks": tissue_masks,
        "mae_per_tissue": mae["per_tissue"],
        "mae_total": mae["total"],
        "mae_n_voxels_per_tissue": mae["n_voxels_per_tissue"],
        "mae_n_voxels_total": mae["n_voxels_total"],
    }
