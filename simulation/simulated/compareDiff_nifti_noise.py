"""Per-tissue ADC precision under image-domain Rician noise injection.

Loads the already-reconstructed diffusion-weighted magnitude volumes (one
NIfTI per b-value per direction per blip per variant, saved by
``run_sim_volume.run_all_qmri_simulations_volume``) from
``simulation/simulated/diff_vol/``, injects Rician noise N times in the
image domain, recomputes the trace-DWI from the geometric mean across
directions, and refits ADC with the *same* ``create_adc_map`` used for the
noise-free maps. Reports per-tissue mean ± SD across realizations.

To stay strictly identical to ``process_dist_corrected_diff.py``:
  - Only TE=100 ms (echo 1) volumes are used for the ADC fit. The MSE/triple
    pipeline writes TE2 and TE3 to disk too, but the canonical ADC fit on
    disk-loaded data uses only echo 1.
  - The trace is the geometric mean across (dir0, dir1, dir2) of the
    magnitude images, with a small floor to avoid log(0).
  - ADC is fit per slice via ``create_adc_map(method='nlls')`` with
    ``adc_max=ADC_MAX`` matching the dist-corrected script.

Calibration uses one scalar sigma per variant, derived from the b=0 / TE1 /
direction-averaged WM mean magnitude. sigma is then reused across every
(b, dir) image of that variant.

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
import re
import sys
import time
from typing import Iterable

import numpy as np
import nibabel as nib

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR = os.path.dirname(_THIS_DIR)
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)

from utils_diffusion import create_adc_map
from utils_noise import (
    PerTissueAccumulator,
    add_rician_noise,
    compute_sigma,
    load_tissue_masks,
    print_summary_table,
    reorient_like_weighted_volume,
    save_volume_nifti,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SUBJECT      = "subj04"
PHANTOM_NAME = f"brainweb-{SUBJECT}"
# N=100 is the asked-for default but the per-voxel ADC NLLS loop is slow on
# this image size. Start at 20 for a quick read; raise for the final table.
N_REAL       = 20
SNR_TARGETS  = (20.0,)     # iterable: extend e.g. (10., 20., 40.) for a sweep
BASE_SEED    = 0
TE_ECHO1_MS  = 100         # MSE/SSE/triple all use TE1 = 100 ms for ADC
ADC_MAX      = 3.6         # mirrors process_dist_corrected_diff.py
EPS          = 1e-12

# Approximate BrainWeb 3T tissue ADC [mm^2/s * 1e3] — for bias readout only.
# Multiplied by 1e3 to match the units of the maps saved by ``run_sim_volume``
# (``adc_nlls_oriented = adc_nlls * 1e3``).
ADC_REFERENCE = {"wm": 0.7, "gm": 0.9, "csf": 3.0}


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
_B_RE   = re.compile(r"_b(\d+)_")
_DIR_RE = re.compile(r"_dir(\d+)_")
_TE_RE  = re.compile(r"_TE(\d+)_")


def _parse(filename: str) -> tuple[int, int, int | None]:
    """Return ``(b_value, direction, TE_ms_or_None)``."""
    b = int(_B_RE.search(filename).group(1))
    d = int(_DIR_RE.search(filename).group(1))
    m_te = _TE_RE.search(filename)
    te = int(m_te.group(1)) if m_te else None
    return b, d, te


def _list_diff_files(
    diff_dir: str, variant: str, blip: str = "blipdown"
) -> list[tuple[int, int, str]]:
    """Return ``[(b, direction, filename), ...]`` for the requested variant.

    Variants share ``diff_vol/`` and are distinguished by their stem shape:
      - ``sse``       : ``ADCw_b{B}_dir{D}_{blip}.nii.gz``       (no TE token)
      - ``triple``    : ``ADCw_b{B}_dir{D}_TE{TE}_{blip}.nii.gz`` (we keep TE1 only)
      - ``multishot`` : ``ADCw_b{B}_dir{D}.nii.gz``              (no TE, no blip)
    """
    out: list[tuple[int, int, str]] = []
    for fname in os.listdir(diff_dir):
        if not fname.endswith(".nii.gz"):
            continue
        if "ADCw_b" not in fname:
            continue
        try:
            b, d, te = _parse(fname)
        except AttributeError:
            continue

        if variant == "multishot":
            # No blip tag, no TE token. Stem ends at dir{d}.nii.gz exactly.
            if te is not None:
                continue
            if f"_{blip}" in fname:
                continue
            if not fname.endswith(f"_dir{d}.nii.gz"):
                continue
            out.append((b, d, fname))
        elif variant == "sse":
            # Blip tag present, no TE token. Stem ends at dir{d}_{blip}.nii.gz.
            if te is not None:
                continue
            if f"_{blip}" not in fname:
                continue
            if not fname.endswith(f"_dir{d}_{blip}.nii.gz"):
                continue
            out.append((b, d, fname))
        elif variant == "triple":
            # Blip tag + TE token. Keep TE1 only (matches dist-corrected ADC).
            if te is None or te != TE_ECHO1_MS:
                continue
            if not fname.endswith(f"_dir{d}_TE{te}_{blip}.nii.gz"):
                continue
            out.append((b, d, fname))
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def _load_per_dir_stack(
    diff_dir: str, files: Iterable[tuple[int, int, str]]
) -> tuple[np.ndarray, np.ndarray]:
    """Stack per-(b, dir) files into ``(n_b, n_dir, Ny, Nx, n_slices)``.

    Returns
    -------
    stack    : ndarray, shape (n_b, n_dir, Ny, Nx, n_slices), float32
    b_values : (n_b,) ndarray of unique sorted b-values [s/mm^2]
    """
    by_b: dict[int, dict[int, np.ndarray]] = {}
    for b, d, fname in files:
        data = nib.load(os.path.join(diff_dir, fname)).get_fdata().astype(np.float32)
        by_b.setdefault(b, {})[d] = data
    b_values = np.array(sorted(by_b.keys()), dtype=int)
    if b_values.size == 0:
        raise FileNotFoundError("no matching diffusion files found")
    dirs_per_b = [sorted(by_b[b].keys()) for b in b_values]
    n_dirs = len(dirs_per_b[0])
    if any(len(d) != n_dirs for d in dirs_per_b):
        raise ValueError(f"inconsistent direction count across b-values: {dirs_per_b}")

    sample = next(iter(by_b[b_values[0]].values()))
    stack = np.empty((b_values.size, n_dirs, *sample.shape), dtype=np.float32)
    for b_i, b in enumerate(b_values):
        for d_i, d in enumerate(sorted(by_b[b].keys())):
            stack[b_i, d_i] = by_b[b][d]
    return stack, b_values


def _trace_dwi(stack: np.ndarray) -> np.ndarray:
    """Geometric mean across the direction axis (axis=1).

    Matches ``process_dist_corrected_diff.py``'s eps-floored geometric mean.
    """
    floored = np.maximum(stack, EPS)
    log_mean = np.mean(np.log(floored), axis=1)
    return np.exp(log_mean)


# ---------------------------------------------------------------------------
# Per-variant Monte Carlo
# ---------------------------------------------------------------------------
def run_variant_adc(
    variant_name: str,
    stack: np.ndarray,
    b_values: np.ndarray,
    tissue_masks: dict[str, np.ndarray],
    snr_target: float,
    n_real: int,
    seed: int,
    out_dir: str,
    save_example_nifti: bool = True,
) -> dict[str, dict[str, float]]:
    """Run the noise-injection Monte Carlo for one ADC variant.

    Parameters
    ----------
    stack : (n_b, n_dir, Ny, Nx, n_slices) float32
        Noise-free per-(b, dir) magnitude volumes.
    b_values : (n_b,) ndarray, s/mm^2
    """
    print(f"\n[ADC-noise] variant={variant_name}  n_b={stack.shape[0]}  "
          f"n_dir={stack.shape[1]}  shape_per_vol={stack.shape[2:]}  "
          f"SNR={snr_target}  N={n_real}")

    # Active-slice filter — many on-disk diff_vol volumes were saved by partial
    # sim runs whose unfilled slices are all zero. Skip those.
    active_slices = np.where(stack.sum(axis=(0, 1, 2, 3)) > 0)[0]
    if active_slices.size == 0:
        raise ValueError(f"{variant_name}: every slice in the stack is empty")
    print(f"[ADC-noise] {variant_name}: {active_slices.size}/{stack.shape[-1]} "
          f"non-empty slices (indices {active_slices[0]}..{active_slices[-1]})")

    # Calibrate sigma from b=0 / TE1 / direction-averaged WM mean.
    if 0 not in b_values.tolist():
        b0_i = int(np.argmin(b_values))
        print(f"[ADC-noise]   {variant_name}: b=0 not present, using b={b_values[b0_i]}")
    else:
        b0_i = b_values.tolist().index(0)
    s_ref_volume = stack[b0_i].mean(axis=0)  # (Ny, Nx, n_slices), average over dirs
    wm_mask = tissue_masks["wm"]
    if s_ref_volume.shape != wm_mask.shape:
        raise ValueError(
            f"{variant_name}: b=0 volume {s_ref_volume.shape} vs WM mask "
            f"{wm_mask.shape} — re-check mask orientation"
        )
    wm_active = np.zeros_like(wm_mask)
    wm_active[..., active_slices] = wm_mask[..., active_slices]
    sigma = compute_sigma(s_ref_volume, wm_active, snr_target)

    masks_active = {}
    for t, m in tissue_masks.items():
        m_active = np.zeros_like(m)
        m_active[..., active_slices] = m[..., active_slices]
        masks_active[t] = m_active

    accumulator = PerTissueAccumulator(masks_active, reference_values=ADC_REFERENCE)
    rng_root = np.random.default_rng(seed)

    n_b, n_dir, Ny, Nx, Nz = stack.shape
    t_start = time.perf_counter()
    for k in range(n_real):
        rng_k = np.random.default_rng(rng_root.integers(0, 2**31 - 1))
        noisy_stack = add_rician_noise(stack, sigma, rng_k)

        if save_example_nifti and k == 0:
            # Save one (b=0, dir0) noise-injected volume per variant.
            fname = (
                f"{PHANTOM_NAME}_{variant_name}_b{int(b_values[b0_i])}_dir0"
                f"_SNR{int(snr_target)}_noise_injected.nii.gz"
            )
            save_volume_nifti(noisy_stack[b0_i, 0], os.path.join(out_dir, fname))

        # Trace-DWI then per-slice ADC fit, only on active slices.
        trace = _trace_dwi(noisy_stack)  # (n_b, Ny, Nx, n_slices)
        adc_volume = np.zeros((Ny, Nx, Nz), dtype=np.float32)
        for z in active_slices:
            slice_data = trace[:, :, :, z]
            adc_map, _ = create_adc_map(
                slice_data, b_values, method="nlls", adc_max=ADC_MAX
            )
            adc_volume[:, :, z] = adc_map.astype(np.float32)
        # Match the saved-map units: noise-free maps in `volumes/` are *1e3.
        adc_volume *= 1e3

        accumulator.update(adc_volume)
        if (k + 1) % max(1, n_real // 10) == 0 or k + 1 == n_real:
            elapsed = time.perf_counter() - t_start
            eta = elapsed / (k + 1) * (n_real - k - 1)
            print(f"  realization {k+1:>4d}/{n_real}  elapsed={elapsed:6.1f}s  ETA={eta:6.1f}s")

    return accumulator.summary()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    file_dir = _THIS_DIR
    diff_dir = os.path.join(file_dir, "diff_vol")
    masks_dir = os.path.join(file_dir, "masks")
    out_dir  = os.path.join(file_dir, "noise_injected")
    os.makedirs(out_dir, exist_ok=True)

    tissue_masks = load_tissue_masks(masks_dir, PHANTOM_NAME, ("wm", "gm", "csf"))
    print(f"[ADC-noise] tissue masks loaded  shape={tissue_masks['wm'].shape}  "
          f"|WM|={int(tissue_masks['wm'].sum())}  |GM|={int(tissue_masks['gm'].sum())}  "
          f"|CSF|={int(tissue_masks['csf'].sum())}")

    variant_to_files = {
        "adc_sse_blipdown":    _list_diff_files(diff_dir, "sse",       "blipdown"),
        "adc_triple_blipdown": _list_diff_files(diff_dir, "triple",    "blipdown"),
        "adc_multishot":       _list_diff_files(diff_dir, "multishot", "blipdown"),
    }

    for snr_target in SNR_TARGETS:
        variant_summaries: dict[str, dict[str, dict[str, float]]] = {}
        for var_name, files in variant_to_files.items():
            if not files:
                print(f"[ADC-noise] {var_name}: no files found in {diff_dir}, skipping")
                continue
            try:
                stack, b_values = _load_per_dir_stack(diff_dir, files)
            except (FileNotFoundError, ValueError) as exc:
                print(f"[ADC-noise] {var_name}: {exc}; skipping")
                continue
            if stack.shape[2:] != tissue_masks["wm"].shape:
                print(
                    f"[ADC-noise] {var_name}: stack {stack.shape[2:]} vs mask "
                    f"{tissue_masks['wm'].shape}; skipping (mask orientation mismatch)"
                )
                continue

            print(f"[ADC-noise] {var_name}: b-values = {b_values.tolist()}")
            summary = run_variant_adc(
                variant_name=var_name,
                stack=stack,
                b_values=b_values,
                tissue_masks=tissue_masks,
                snr_target=snr_target,
                n_real=N_REAL,
                seed=BASE_SEED,
                out_dir=out_dir,
            )
            variant_summaries[var_name] = summary

        print_summary_table(
            f"ADC per-tissue precision  (SNR_target={snr_target}, N={N_REAL})",
            "ADC in 1e-3 mm^2/s (i.e. ADC * 1e3 to match volumes/); "
            "bias = mean - BrainWeb reference",
            variant_summaries,
        )


if __name__ == "__main__":
    main()
