"""Triple spin-echo EPI T2-relaxometry simulation pipeline.

Each TR produces three echoes (TE1, auto-computed TE2, TE3) from one RF90 +
three RF180s. Scanning ``TE1_values`` therefore samples the T2 decay curve
at 3 * len(TE1_values) effective echo times — for the default ~15 TE1 values
that means ~45 (TE, image) datapoints for a T2 fit per voxel.

Pipeline
--------
For each TE1 in ``TE1_values``:

1. Build an ``EPIDiffusionTripleSEPulseqSeq`` at b=0 with the requested TE1.
2. Bloch-simulate; split the signal into three per-echo k-spaces (echo 1
   with partial Fourier, echoes 2/3 full).
3. NUFFT-reconstruct each echo with its own k-space trajectory.
4. Append the three (image, TE) pairs to the global list.

After all TE1 values are collected, the (image, TE) pairs are sorted by TE
and ``S(TE) = S0 * exp(-TE/T2)`` is fit per voxel with NLLS. Stems for the
saved T2-weighted NIfTI files are disambiguated when TE2/TE3 round to the
same millisecond across different TE1 choices.

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
    system_type: Optional[object] = None,
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

    # Empty slice: return zero-filled maps for the volume runner.
    if not any(bool(mask.any()) for mask in tissue_masks.values()):
        zeros_2d = np.zeros((Ny, Nx), dtype=np.float64)
        return {
            "t2_nlls": zeros_2d.copy(), "t2_loglinear": zeros_2d.copy(),
            "TEs": TE1_values, "reference_map": zeros_2d.copy(),
            "tissue_masks": tissue_masks,
            "mae_per_tissue": {name: 0 for name in tissue_masks},
            "mae_total": 0,
            "mae_n_voxels_per_tissue": {name: 0 for name in tissue_masks},
            "mae_n_voxels_total": 0,
            "weighted_images": {},
        }

    all_echo_images: list[np.ndarray] = []
    all_echo_tes: list[float] = []

    for te1 in TE1_values:
        print(f"[T2-3SE]  b={b_value} s/mm²  TE1={te1} ms", flush=True, end="\r")
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
            alternating_blip_polarity=True,
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

        # NUFFT-reconstruct each of the three echoes with its own trajectory.
        echo_imgs = []
        for kspace_line, traj_echo in (
            (echo1_ksp, traj_epi1),
            (echo2_ksp, traj_epi2),
            (echo3_ksp, traj_epi3),
        ):
            nufft_op = get_operator(
                backend_name="cufinufft" if use_gpu else "finufft",
                samples=traj_echo,
                shape=(Ny, Nx),
                n_coils=1,
                density=True,
            )
            kspace_tensor = kspace_line.to(torch.complex64)
            echo_imgs.append(nufft_op.adj_op(kspace_tensor.flatten()).squeeze().cpu().numpy().T)

        for img, te_ms in zip(echo_imgs, (te1_ms, te2_ms, te3_ms)):
            mag_img = np.abs(img)
            all_echo_images.append(mag_img)
            all_echo_tes.append(te_ms)

    order = np.argsort(all_echo_tes)
    images_stack = np.array(all_echo_images)[order]
    te_sorted = np.array(all_echo_tes)[order]

    # Stems may collide when TE2/TE3 (auto-computed) round to the same ms
    # across different TE1 values — disambiguate with a sequential suffix.
    weighted_images: dict[str, np.ndarray] = {}
    _seen: dict[str, int] = {}
    for i, te_ms in enumerate(te_sorted):
        base = f"T2w_TE{int(round(te_ms))}_{blip_tag}"
        n = _seen.get(base, 0)
        stem = base if n == 0 else f"{base}_e{n}"
        _seen[base] = n + 1
        weighted_images[stem] = images_stack[i]

    t2_nlls, _ = create_t2_map(images_stack, te_sorted, method="nlls")
    t2_nlls_oriented = t2_nlls / 1000.0
    if save_slice_npy:
        out_path = os.path.join(
            paths.volumes_dir, f"{paths.phantom_name}-T2_MSE_{blip_tag}.nii.gz"
        )
        save_magnitude_nifti(np.rot90(np.flipud(t2_nlls_oriented), 1), out_path, res)
        print(f"[T2-3SE] saved {out_path}")

    ref_T2 = phantom_map_to_2d(phantom.T2)
    est_T2 = t2_nlls / 1000.0
    mae = compute_mae_per_tissue(est_T2, ref_T2, tissue_masks)

    return {
        "t2_nlls": t2_nlls_oriented,
        "TEs": te_sorted,
        "reference_map": ref_T2,
        "tissue_masks": tissue_masks,
        "mae_per_tissue": mae["per_tissue"],
        "mae_total": mae["total"],
        "mae_n_voxels_per_tissue": mae["n_voxels_per_tissue"],
        "mae_n_voxels_total": mae["n_voxels_total"],
        "weighted_images": weighted_images,
    }
