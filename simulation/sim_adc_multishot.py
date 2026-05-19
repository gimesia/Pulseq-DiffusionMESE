"""Multishot SE ADC pipeline (refactor of qMRI_adc_diff_multishot_se.py)."""
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import torch

from utils_sim_lib import (
    PathConfig,
    affine_from_res,
    compute_mae_per_tissue,
    compute_trace_dwi,
    ensure_seq_path_on_syspath,
    load_phantom_for_sim,
    make_quiet_logger,
    phantom_map_to_2d,
    save_magnitude_nifti,
    save_tissue_masks,
    simulate_signal,
)


def run_adc_multishot(
    paths: PathConfig,
    *,
    slice_idx: Optional[int] = None,
    fov: float = 224e-3,
    res: float = 2.33333333,
    b_values: Sequence[int] = tuple(range(0, 2001, 100)),
    TE_values: Sequence[int] = (100,),
    TE_for_DTI: Optional[int] = None,
    TR: int = 5000,
    ETL: int = 1,
    b_directions: int = 6,
    small_delta: Optional[float] = None,
    big_DELTA: Optional[float] = None,
    system_type=None,
    use_gpu: Optional[bool] = None,
    save_slice_npy: bool = True,
    phantom_slice: Optional[tuple] = None,
) -> dict:
    """Run the multishot SE diffusion / ADC simulation.

    Cartesian Nyquist k-space → plain inverse FFT (no NUFFT). The ADC fit
    averages across TEs (T2 cancels in S(b)/S(0) ratio). DTI uses the
    shortest TE only (single-TE log-linear model).
    """
    paths.ensure_dirs()
    ensure_seq_path_on_syspath(paths.seq_lib_path)
    logger = make_quiet_logger()

    from DiffusionSEMultishotPulseqSeq import DiffusionSEMultishotPulseqSeq
    from utils import SystemLimitType
    from utils_simulation import fft_reconstruct_image
    from utils_diffusion import create_adc_map, create_dti_maps

    if system_type is None:
        system_type = SystemLimitType.EXTREME
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()
    if TE_for_DTI is None:
        TE_for_DTI = TE_values[0]

    b_values = np.asarray(b_values, dtype=int)
    slice_thickness = res * 1e-3
    Nx = Ny = int(fov / slice_thickness)

    if phantom_slice is not None:
        phantom, phantom_data, tissue_masks = phantom_slice
    else:
        phantom, phantom_data, tissue_masks = load_phantom_for_sim(paths, res, slice_idx)

    images_per_te = {te: [] for te in TE_values}
    n_dirs: Optional[int] = None
    last_seq = None

    for te in TE_values:
        for b_value in b_values:
            print(f"[ADC-MS] TE={te} ms  b={b_value}")
            name = f"DiffSEMultishot-b{int(b_value)}-te{int(te)}"
            seq = DiffusionSEMultishotPulseqSeq(
                name=name,
                fov=fov,
                Nx=Nx,
                Ny=Ny,
                slice_thickness=slice_thickness,
                TR=TR,
                TE=int(te),
                ETL=ETL,
                b_value=int(b_value),
                b_directions=b_directions,
                small_delta=small_delta,
                big_DELTA=big_DELTA,
                save_dir=paths.sequences_dir,
                v141_compat=True,
                system_type=system_type,
                logger=logger,
            )
            seq.build_seq()
            seq.write()
            last_seq = seq
            if n_dirs is None:
                n_dirs = len(seq.b_directions)

            seq_filename = seq.get_save_filename()
            signal = simulate_signal(
                os.path.join(paths.sequences_dir, seq_filename),
                phantom_data,
                use_gpu,
                gpu_max_states=50000,
                gpu_min_emit=1e-6,
                cpu_max_states=5000,
                cpu_min_emit=1e-5,

            )
            assert signal.shape[0] == n_dirs * Ny * Nx

            dir_images = []
            for d in range(n_dirs):
                kspace = (
                    signal[d * Ny * Nx : (d + 1) * Ny * Nx]
                    .numpy()
                    .reshape(Ny, Nx)
                )
                img_mag, _ = fft_reconstruct_image(kspace, use_gpu=use_gpu)
                img_mag = img_mag.squeeze()
                dir_images.append(img_mag)

                echo_name = (
                    f"DiffMultiShotSE-b{int(b_value)}-dir{d}-TE{int(te)}.nii.gz"
                )
                save_magnitude_nifti(
                    img_mag, os.path.join(paths.diff_img_dir, echo_name), res
                )
            images_per_te[te].append(np.stack(dir_images, axis=0))

    # (n_te, n_b, n_dirs, Ny, Nx)
    stacked = np.stack(
        [np.stack(images_per_te[te], axis=0) for te in TE_values], axis=0
    )
    mag_images_adc = np.abs(stacked.mean(axis=0))  # (n_b, n_dirs, Ny, Nx)
    mag_images_dti = np.abs(stacked[TE_values.index(TE_for_DTI)])

    trace_dwi = compute_trace_dwi(mag_images_adc)
    adc_nlls, _ = create_adc_map(trace_dwi, b_values, method="nlls")
    adc_ll, _ = create_adc_map(trace_dwi, b_values, method="loglinear")
    fa_map, md_map, eigvals_map, dti_s0_map = create_dti_maps(
        mag_images_dti, b_values, last_seq.b_directions
    )

    adc_nlls_oriented = np.fliplr(adc_nlls) * 1e3
    adc_ref_oriented = np.fliplr(adc_ll) * 1e3  # placeholder, kept for parity
    if save_slice_npy:
        np.save(
            os.path.join(paths.volumes_dir, f"{paths.phantom_name}-ADC_multishot_se.npy"),
            adc_nlls_oriented,
        )
    save_tissue_masks(tissue_masks, paths.masks_dir, paths.phantom_name)
    print(
        f"[ADC-MS] saved ADC map to {paths.volumes_dir}"
    )

    ref_D = phantom_map_to_2d(phantom.D)
    est_adc = adc_nlls * 1e3
    mae = compute_mae_per_tissue(est_adc, ref_D, tissue_masks)

    return {
        "adc_nlls": adc_nlls,
        "adc_loglinear": adc_ll,
        "fa_map": fa_map,
        "md_map": md_map,
        "b_values": b_values,
        "reference_map": ref_D,
        "tissue_masks": tissue_masks,
        "mae_per_tissue": mae["per_tissue"],
        "mae_total": mae["total"],
        "mae_n_voxels_per_tissue": mae["n_voxels_per_tissue"],
        "mae_n_voxels_total": mae["n_voxels_total"],
    }
