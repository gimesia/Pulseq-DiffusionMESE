"""Triple spin-echo EPI diffusion / ADC simulation pipeline.

This is the headline pipeline of the ESMRMB 2026 deliverable: a single TR
produces three echoes (TE1, auto-computed TE2, TE3) from one RF90 +
three RF180s. PGSE diffusion encoding is applied once around the first
RF180, so all three echoes carry the same diffusion weighting but with
progressively stronger T2 attenuation.

Pipeline
--------
For each b-value in ``b_values`` and each gradient direction:

1. Build an ``EPIDiffusionTripleSEPulseqSeq`` at the requested TE1.
   ``alternating_blip_polarity=True`` flips the EPI blip direction
   between echoes for B0 field-mapping; ``phase_cycling=True`` minimises
   stimulated-echo contamination.
2. Bloch-simulate; split the k-space stream into three per-echo k-spaces
   (echo 1 with partial Fourier, echoes 2/3 full).
3. Pull the trajectory, split it into three echo trajectories, and run
   NUFFT once per (echo, direction).
4. Trace-DWI is computed from echo 1 only (the shortest-TE echo has the
   highest SNR); ADC is fit with NLLS.

Saved NIfTI: one per (b-value, direction, echo) + final ADC map.

Author      : Aron Gimesi <aron.gimesi@tecnico.ulisboa.pt>
Affiliation : Instituto Superior Técnico | MSCA-DN IQ-BRAIN
Date        : 2026
Context     : ESMRMB 2026 — Pulseq DiffusionMESE showcase

Funding acknowledgement (mandatory):
    IQ-BRAIN is funded by the European Union (MSCA Doctoral Network,
    December 2024–November 2028, Grant Agreement No. 101169519).
"""
from __future__ import annotations

import warnings
import os
from typing import Optional, Sequence

import numpy as np
import torch

from utils_sim_lib import (
    PathConfig,
    compute_trace_dwi,
    ensure_seq_path_on_syspath,
    load_phantom_for_sim,
    make_quiet_logger,
    save_magnitude_nifti,
    simulate_signal,
)

warnings.filterwarnings("ignore", category=DeprecationWarning, message="__array__")

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
    system_type: Optional[object] = None,
    use_gpu: Optional[bool] = None,
    save_slice_npy: bool = True,
    phantom_slice: Optional[tuple] = None,
    dti_maps: bool = False,
    alternating_blip_polarity: bool = False,
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

    if phantom_slice is not None:
        phantom, phantom_data, tissue_masks = phantom_slice
    else:
        phantom, phantom_data, tissue_masks = load_phantom_for_sim(paths, res, slice_idx)

    # Empty slice: return zero-filled maps for the volume runner.
    if not any(bool(mask.any()) for mask in tissue_masks.values()):
        zeros_2d = np.zeros((Ny, Nx), dtype=np.float64)
        return {
            "adc_nlls": zeros_2d.copy(),
            "b_values": b_values,
            "weighted_images": {},
            "phantom": phantom,
        }

    all_echo_images = []
    te1_ms = te2_ms = te3_ms = None
    last_seq = None

    for b_value in b_values:
        print(f"[ADC-3SE]  b={b_value} s/mm²  TE={TE} ms", flush=True, end="\r")
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
            alternating_blip_polarity=alternating_blip_polarity,
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

        # One NUFFT operator per echo, reused across diffusion directions.
        img_shape = (Ny, Nx)
        nufft_backend = "cufinufft" if use_gpu else "finufft"
        nufft_echo1 = get_operator(backend_name=nufft_backend, samples=traj_epi1,
                                   shape=img_shape, n_coils=1, density=True)
        nufft_echo2 = get_operator(backend_name=nufft_backend, samples=traj_epi2,
                                   shape=img_shape, n_coils=1, density=True)
        nufft_echo3 = get_operator(backend_name=nufft_backend, samples=traj_epi3,
                                   shape=img_shape, n_coils=1, density=True)

        per_direction_echo_images = []
        for d in range(n_dirs):
            echo_images = []
            for kspace_line, nufft_op in (
                (echo1_ks[d], nufft_echo1),
                (echo2_ks[d], nufft_echo2),
                (echo3_ks[d], nufft_echo3),
            ):
                kspace_tensor = torch.from_numpy(np.array(kspace_line)).to(torch.complex64)
                # No `.T`: an untransposed nufft_op.adj_op() output
                # already lands on the same array grid as the phantom's
                # own parameter maps (verified with a synthetic marker
                # through this exact trajectory/operator pair).
                echo_images.append(nufft_op.adj_op(kspace_tensor.flatten()).squeeze().cpu().numpy())
            per_direction_echo_images.append(np.stack(echo_images, axis=0))  # (3, Ny, Nx)

        all_echo_images.append(np.stack(per_direction_echo_images, axis=0))  # (n_dirs, 3, Ny, Nx)

    all_echo_images = np.array(all_echo_images)  # (n_b, n_dirs, 3, Ny, Nx)
    mag_images = np.abs(all_echo_images)
    mag_echo1 = mag_images[:, :, 0, :, :]
    mag_images_combined = mag_images.mean(axis=2)

    weighted_images = {
        f"ADCw_b{int(b)}_dir{d}_TE{int(te_ms_echo)}_{blip_tag}": mag_images[b_i, d, e]
        for b_i, b in enumerate(b_values)
        for d in range(mag_images.shape[1])
        for e, te_ms_echo in enumerate([te1_ms, te2_ms, te3_ms])
    }

    trace_dwi = compute_trace_dwi(mag_echo1)
    adc_nlls, _ = create_adc_map(trace_dwi, b_values, method="nlls")
    if dti_maps:
        fa_map, md_map, eigvals_map, dti_s0_map = create_dti_maps(
            mag_images_combined, b_values, last_seq.b_directions
        )

    adc_nlls_oriented = adc_nlls * 1e3
    if save_slice_npy:
        out_path = os.path.join(
            paths.volumes_dir, f"{paths.phantom_name}-ADC_MSE_{blip_tag}.nii.gz"
        )
        # No rot90/flipud: with the `.T` above removed, adc_nlls_oriented
        # already matches the phantom's D map grid directly (see the
        # comment on the recon step above) — the rot90(flipud(...), 1)
        # this used to apply did not correct for that stray `.T`; composed
        # with it, it netted out to a plain 180° rotation of the
        # already-misaligned map.
        save_magnitude_nifti(adc_nlls_oriented, out_path, res)
        print(f"[ADC-3SE] saved {out_path}")

    result = {
        "adc_nlls": adc_nlls_oriented,
        "b_values": b_values,
        "weighted_images": weighted_images,
        "phantom": phantom,
    }
    if dti_maps:
        result["fa_map"] = fa_map
        result["md_map"] = md_map
    return result
