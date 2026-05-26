"""Image-domain Rician noise injection and per-tissue Monte Carlo aggregation.

Shared utilities for ``compareT2_nifti_noise.py`` and
``compareDiff_nifti_noise.py``. Operates on already-reconstructed magnitude
volumes (NIfTI on disk) rather than at the k-space / Bloch level.

The Rician noise model assumes the real-channel signal is the noise-free
magnitude with zero phase. Adding i.i.d. Gaussian noise N(0, sigma^2) to both
the real and imaginary channels and taking the magnitude reproduces the
Rician distribution that magnitude MR images follow at moderate-to-low SNR.
Adding Gaussian noise directly to the magnitude is wrong because it allows
negative samples and does not reproduce the noise-floor lift near zero
signal.

Calibration uses a single scalar sigma per variant, computed from
    sigma = S_ref / SNR_target,
where S_ref is the mean magnitude over the white-matter mask in that
variant's b=0 / TE1 / direction-averaged volume. sigma is then reused across
every (b, TE, direction) volume within the variant — it is *not* recomputed
per echo, because the physical noise floor of an acquisition is set once by
the receiver chain, not by the contrast preparation.

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
from typing import Iterable, Optional

import numpy as np
import nibabel as nib


# ---------------------------------------------------------------------------
# Orientation helper
# ---------------------------------------------------------------------------
def reorient_like_weighted_volume(volume: np.ndarray) -> np.ndarray:
    """Apply the canonical orientation transform used for saved weighted volumes.

    Mirrors ``run_sim_volume._reorient_volume`` so that tissue masks (which
    were saved without this transform) can be aligned with the weighted-image
    volumes loaded from ``t2_vol/`` and ``diff_vol/``.
    """
    return np.flip(np.rot90(volume, k=1, axes=(0, 1)), axis=1)


# ---------------------------------------------------------------------------
# Mask loading
# ---------------------------------------------------------------------------
def load_tissue_masks(
    masks_dir: str,
    phantom_name: str,
    tissue_names: Iterable[str] = ("wm", "gm", "csf"),
    reorient: bool = False,
) -> dict[str, np.ndarray]:
    """Load binary tissue masks saved by ``run_all_qmri_simulations_volume``.

    Parameters
    ----------
    masks_dir : str
        Directory containing ``{phantom_name}-mask_{tissue}_volume.nii.gz``.
    phantom_name : str
        Filename prefix used when the masks were saved.
    tissue_names : iterable of str
        Which tissue masks to load (default WM/GM/CSF).
    reorient : bool
        Whether to apply :func:`reorient_like_weighted_volume` to the loaded
        masks. Default ``False`` — empirically the masks on disk already align
        voxel-for-voxel with the weighted volumes loaded from ``t2_vol/`` /
        ``diff_vol/`` (the per-TE std inside the WM mask is much smaller with
        ``reorient=False``). Flip this if a future simulator version changes
        the on-disk mask convention.

    Returns
    -------
    dict mapping tissue name -> bool ndarray of shape ``(Ny, Nx, n_slices)``.
    """
    masks: dict[str, np.ndarray] = {}
    for tissue in tissue_names:
        path = os.path.join(masks_dir, f"{phantom_name}-mask_{tissue}_volume.nii.gz")
        data = nib.load(path).get_fdata().astype(bool)
        if reorient:
            data = reorient_like_weighted_volume(data)
        masks[tissue] = data
    return masks


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def compute_sigma(s_ref_volume: np.ndarray, wm_mask: np.ndarray, snr_target: float) -> float:
    """Single scalar Rician sigma calibrated from a WM mean signal.

    sigma = mean(S_ref[wm_mask]) / SNR_target

    Parameters
    ----------
    s_ref_volume : ndarray
        The variant's b=0 / TE1 magnitude volume. For multi-direction ADC
        acquisitions, average the per-direction images first so the
        calibration is not dominated by the noisiest direction.
    wm_mask : ndarray of bool
        White-matter mask, same shape as ``s_ref_volume``.
    snr_target : float
        Target SNR (signal / noise std-dev).

    Returns
    -------
    sigma : float
        Standard deviation to use for the per-channel Gaussian noise.
    """
    if s_ref_volume.shape != wm_mask.shape:
        raise ValueError(
            f"shape mismatch: S_ref {s_ref_volume.shape} vs WM mask {wm_mask.shape}"
        )
    wm_values = s_ref_volume[wm_mask]
    wm_values = wm_values[np.isfinite(wm_values)]
    if wm_values.size == 0:
        raise ValueError("WM mask contains no finite voxels in the reference volume")
    s_ref = float(wm_values.mean())
    if not np.isfinite(s_ref) or s_ref <= 0:
        raise ValueError(f"Non-physical S_ref={s_ref}; cannot calibrate sigma")
    sigma = s_ref / float(snr_target)
    print(
        f"[noise]   S_ref(WM, b=0/TE1) = {s_ref:.4g}   SNR_target = {snr_target:g}   "
        f"-> sigma = {sigma:.4g}"
    )
    return sigma


# ---------------------------------------------------------------------------
# Rician noise injection
# ---------------------------------------------------------------------------
def add_rician_noise(
    magnitude_volume: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Inject Rician noise via the complex-magnitude trick.

    The magnitude image is taken as the real channel with phase = 0. Two
    i.i.d. Gaussian noise fields N(0, sigma^2) are added to the real and
    imaginary channels respectively, then the magnitude is taken:

        noisy = sqrt( (S + n_real)^2 + n_imag^2 )

    This is the standard way to synthesise Rician noise on magnitude MRI
    images. Do NOT add Gaussian noise directly to the magnitude — that allows
    negative samples and misses the noise-floor lift near zero signal.

    Parameters
    ----------
    magnitude_volume : ndarray
        Noise-free magnitude image of any shape. NaN voxels are preserved.
    sigma : float
        Per-channel Gaussian standard deviation. Must be non-negative.
    rng : numpy.random.Generator
        Random generator for reproducibility. Pass a freshly-seeded
        generator per realization.

    Returns
    -------
    ndarray, same shape and dtype as ``magnitude_volume`` (float).
    """
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    if sigma == 0:
        return magnitude_volume.astype(float, copy=True)
    s = magnitude_volume.astype(float, copy=False)
    nan_mask = ~np.isfinite(s)
    s_safe = np.where(nan_mask, 0.0, s)
    n_real = rng.normal(0.0, sigma, size=s_safe.shape)
    n_imag = rng.normal(0.0, sigma, size=s_safe.shape)
    noisy = np.sqrt((s_safe + n_real) ** 2 + n_imag ** 2)
    if nan_mask.any():
        noisy = np.where(nan_mask, np.nan, noisy)
    return noisy


# ---------------------------------------------------------------------------
# Per-tissue aggregation across Monte Carlo realizations
# ---------------------------------------------------------------------------
class PerTissueAccumulator:
    """Running per-tissue stats over Monte Carlo realizations.

    For each realization, the caller supplies a parameter map (T2 or ADC).
    For each tissue we record TWO scalars per realization:
      - the mean of the parameter inside the tissue mask
      - the median of the parameter inside the tissue mask
    ignoring non-finite and non-positive voxels (the fitters return 0 for
    masked-out / failed voxels).

    Both summaries are reported because the per-voxel NLLS fit on this data
    saturates a non-trivial fraction of WM/GM voxels at ``t2_bounds[1]``
    (or ``adc_max``), pulling the mean upward. The median is robust to this
    saturation and is closer to the true per-tissue parameter; the mean is
    closer to what ``process_dist_corrected_*.py`` would report. Choose
    whichever matches the downstream comparison you are doing.

    Across realizations we report:
        mean       — mean of per-realization tissue means      (bias proxy)
        sd         — SD of per-realization tissue means        (precision via mean)
        median     — mean of per-realization tissue medians    (robust bias proxy)
        median_sd  — SD of per-realization tissue medians      (precision via median)
        n_real     — number of realizations contributing
        ref        — reference value (set externally)
        bias       — mean - ref
        median_bias— median - ref
    """

    def __init__(
        self,
        tissue_masks: dict[str, np.ndarray],
        reference_values: Optional[dict[str, float]] = None,
    ):
        self.tissue_masks = tissue_masks
        self.reference_values = reference_values or {}
        self._per_realization_means: dict[str, list[float]] = {
            t: [] for t in tissue_masks
        }
        self._per_realization_medians: dict[str, list[float]] = {
            t: [] for t in tissue_masks
        }
        self._per_realization_n_voxels: dict[str, list[int]] = {
            t: [] for t in tissue_masks
        }

    def update(self, parameter_map: np.ndarray) -> None:
        """Record one realization's per-tissue mean and median.

        Only voxels that are finite AND > 0 contribute. Both
        ``create_t2_map`` and ``create_adc_map`` return 0 for masked-out or
        failed-fit voxels, so this filter removes them without us having to
        track the fitter's internal mask.
        """
        for tissue, mask in self.tissue_masks.items():
            if parameter_map.shape != mask.shape:
                raise ValueError(
                    f"shape mismatch: parameter_map {parameter_map.shape} vs "
                    f"{tissue} mask {mask.shape}"
                )
            values = parameter_map[mask]
            values = values[np.isfinite(values) & (values > 0)]
            if values.size == 0:
                self._per_realization_means[tissue].append(float("nan"))
                self._per_realization_medians[tissue].append(float("nan"))
                self._per_realization_n_voxels[tissue].append(0)
            else:
                self._per_realization_means[tissue].append(float(values.mean()))
                self._per_realization_medians[tissue].append(float(np.median(values)))
                self._per_realization_n_voxels[tissue].append(int(values.size))

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for tissue, means in self._per_realization_means.items():
            arr = np.asarray(means, dtype=float)
            med_arr = np.asarray(self._per_realization_medians[tissue], dtype=float)
            finite = arr[np.isfinite(arr)]
            finite_med = med_arr[np.isfinite(med_arr)]
            n_voxels = self._per_realization_n_voxels[tissue]
            if finite.size == 0:
                mean_v = sd_v = float("nan")
            else:
                mean_v = float(finite.mean())
                sd_v = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
            if finite_med.size == 0:
                median_v = median_sd = float("nan")
            else:
                median_v = float(finite_med.mean())
                median_sd = float(finite_med.std(ddof=1)) if finite_med.size > 1 else 0.0
            ref = self.reference_values.get(tissue, float("nan"))
            out[tissue] = {
                "mean": mean_v,
                "sd": sd_v,
                "median": median_v,
                "median_sd": median_sd,
                "n_realizations": int(finite.size),
                "n_voxels_first_real": int(n_voxels[0]) if n_voxels else 0,
                "reference": float(ref),
                "bias": float(mean_v - ref) if np.isfinite(ref) else float("nan"),
                "median_bias": float(median_v - ref) if np.isfinite(ref) else float("nan"),
            }
        return out


# ---------------------------------------------------------------------------
# Pretty-print summary table
# ---------------------------------------------------------------------------
def print_summary_table(
    title: str,
    units: str,
    variant_summaries: dict[str, dict[str, dict[str, float]]],
) -> None:
    """Compact ASCII table of per-tissue mean ± SD and median ± SD per variant.

    Parameters
    ----------
    title : str
        Header line (e.g. "T2 [s]" or "ADC [mm^2/s]").
    units : str
        Unit string appended to mean / SD numbers (for context only).
    variant_summaries : dict
        ``{variant_name: PerTissueAccumulator.summary()}``.
    """
    print()
    print("=" * 118)
    print(f"  {title}")
    print("=" * 118)
    header = (
        f"  {'variant':<26} {'tissue':<5} "
        f"{'mean':>9} {'SD':>9} {'bias':>9}   "
        f"{'median':>9} {'medSD':>9} {'medBias':>9}   "
        f"{'ref':>8} {'N':>3}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for variant_name, summary in variant_summaries.items():
        for tissue, stats in summary.items():
            print(
                f"  {variant_name:<26} {tissue:<5} "
                f"{stats['mean']:>9.4g} {stats['sd']:>9.3g} {stats['bias']:>+9.3g}   "
                f"{stats['median']:>9.4g} {stats['median_sd']:>9.3g} {stats['median_bias']:>+9.3g}   "
                f"{stats['reference']:>8.3g} {stats['n_realizations']:>3d}"
            )
    print("=" * 118)
    print(f"  ({units})")


# ---------------------------------------------------------------------------
# NIfTI writer (variant-agnostic; uses a diagonal affine)
# ---------------------------------------------------------------------------
def save_volume_nifti(volume: np.ndarray, path: str, res_mm: float = 2.33333333) -> None:
    """Save a 3D float volume as NIfTI with an isotropic diagonal affine.

    NaN voxels are zeroed before saving. Matches the convention used by the
    volume runner so that the noise-injected outputs can be overlaid on the
    existing reference / parameter-map NIfTIs without further alignment.
    """
    affine = np.diag([res_mm, res_mm, res_mm, 1.0]).astype(np.float64)
    data = np.nan_to_num(volume.astype(np.float32, copy=False), nan=0.0,
                         posinf=0.0, neginf=0.0)
    nib.save(nib.Nifti1Image(data, affine), path)
