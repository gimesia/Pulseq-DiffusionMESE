"""
Vectorised T2 relaxometry from a multi-echo magnitude image stack.

Author      : Aron Gimesi <aron.gimesi@tecnico.ulisboa.pt>
Affiliation : Instituto Superior Técnico | MSCA-DN IQ-BRAIN
Date        : 2026
Context     : ESMRMB 2026 — Pulseq DiffusionMESE showcase

Funding acknowledgement (mandatory):
    IQ-BRAIN is funded by the European Union (MSCA Doctoral Network,
    December 2024–November 2028, Grant Agreement No. 101169519).

This module fits the mono-exponential decay model

    S(TE) = S0 * exp(-TE / T2)

to per-pixel signal-vs-echo-time curves and returns quantitative T2
and S0 maps.

Two estimators are provided:

1. `t2_loglinear` - vectorised log-linear least squares over all
   pixels at once. Fast, closed-form, and unbiased in the noise-free
   limit. Biased toward shorter T2 in the presence of magnitude /
   Rician noise because `log(S)` distorts the noise distribution and
   weights all echoes equally regardless of SNR.

2. `t2_nlls` - per-pixel non-linear least squares using
   `scipy.optimize.curve_fit`. Slower but statistically better behaved
   on noisy data because it fits in the original (linear) signal
   domain. Uses the log-linear estimate as its starting point, which
   typically halves the iteration count and avoids local minima in
   long-T2 voxels (e.g. CSF).

Both estimators reject implausible fits to 0 rather than reporting a
saturated/cosmetic number:

- `t2_loglinear` zeroes non-decaying voxels (slope >= 0, i.e. T2 =
  +inf) *and* voxels whose fitted T2 exceeds `t2_max` — a near-zero
  negative slope inflates T2 arbitrarily far, and nothing past
  `t2_max` (default 3000 ms) is plausible in-vivo brain tissue at 3T.
- `t2_nlls` fits with `scipy.optimize.least_squares` (not the
  `curve_fit` wrapper) specifically so it can inspect
  `result.active_mask`: a fit pinned at the upper `t2_bounds` edge
  means the optimizer wanted to go higher and got stopped by the
  constraint, not that `t2_bounds[1]` is a real measurement. Clipping
  that to the bound would silently draw a false value at exactly the
  cap; this rejects it to 0 instead, the same reasoning `t2_loglinear`
  applies to its own out-of-range fits.

The acquisition assumptions are the same as for any spin-echo T2
fit: TR long compared to tissue T1, no diffusion weighting (b = 0),
and no stimulated-echo contamination. Violations bias the recovered
T2; they do not produce obvious failure modes.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------


def mono_exponential(te, s0, t2):
    """
    Mono-exponential T2 decay model: S(TE) = s0 * exp(-te / t2).

    Parameters
    ----------
    te : array_like
        Echo time(s) in milliseconds.
    s0 : float
        Signal extrapolated to TE = 0 (PD- and T1-weighted).
    t2 : float
        Transverse relaxation time in milliseconds.

    Returns
    -------
    ndarray or float
        Predicted magnitude signal at each echo time.

    Notes
    -----
    Units of `te` and `t2` must match. The model is only valid above
    the noise floor; near it, magnitude images follow a Rician (not
    Gaussian) distribution, and any least-squares fit will be biased
    toward longer T2.
    """
    return s0 * np.exp(-te / t2)


def _mono_exponential_residual(params, te, signal):
    """Residual vector `model(te; params) - signal` for `least_squares`.

    A thin wrapper around `mono_exponential` so `t2_nlls` can call
    `scipy.optimize.least_squares` directly (instead of the `curve_fit`
    convenience wrapper) and read back `result.active_mask` — see the
    "Implausible long T2" reasoning in the module docstring.
    """
    s0, t2 = params
    return mono_exponential(te, s0, t2) - signal


# ---------------------------------------------------------------------------
# Masking helper
# ---------------------------------------------------------------------------


def _build_mask(data, mask_threshold_frac):
    """
    Build a boolean foreground mask from the first-echo image.

    The first-echo image is the most PD-weighted of the stack and has
    the highest SNR, so it is the natural choice for thresholding.
    Using the global stack maximum (instead of the first-echo maximum)
    biases the threshold toward CSF voxels and tends to exclude
    parenchyma this helper is for this.

    Parameters
    ----------
    data : ndarray, shape (n_te, ny, nx)
        Magnitude image stack.
    mask_threshold_frac : float
        Fraction of the first-echo maximum below which voxels are
        treated as background and skipped during fitting.

    Returns
    -------
    mask : ndarray of bool, shape (ny, nx)
        True where the voxel should be fit, False elsewhere.
    """
    return data[0] > mask_threshold_frac * np.max(data[0])


# ---------------------------------------------------------------------------
# Log-linear estimator (vectorised)
# ---------------------------------------------------------------------------
def t2_loglinear(data, te_list, mask_threshold_frac=0.15, eps=1e-12, t2_max=3000.0):
    """
    Vectorised log-linear T2 fit across all pixels at once.

    Taking the log of the mono-exponential model linearises it:

        log(S) = log(S0) - TE / T2

    so a single least-squares solve over all pixels gives slope and
    intercept maps in one shot. This is several orders of magnitude
    faster than a per-pixel non-linear fit.

    Parameters
    ----------
    data : ndarray, shape (n_te, ny, nx)
        Magnitude image stack. Must be real-valued and non-negative
        (apply `np.abs` to complex reconstructions before calling).
    te_list : array_like, shape (n_te,)
        Echo times in milliseconds, ordered to match axis 0 of `data`.
    mask_threshold_frac : float, optional
        Fraction of the first-echo maximum used to mask background.
        Pixels below this are returned as 0 in both output maps.
        Default 0.15 (15%) works well for brain phantoms; lower it for
        low-contrast objects.
    eps : float, optional
        Floor applied before taking the logarithm to avoid `log(0)`.
        Should be small relative to the noise floor. Default 1e-12.
    t2_max : float, optional
        Upper plausibility ceiling on T2, in milliseconds. Default
        3000 ms comfortably contains in-vivo brain CSF at 3T (~2000
        ms). Fits above it are rejected (see "Implausible long T2"
        below), not clipped to this value.

    Returns
    -------
    t2_map : ndarray, shape (ny, nx)
        T2 in milliseconds. Voxels with non-physical (non-decaying)
        signals, implausibly long fitted T2 (see below), or below the
        mask threshold are set to 0.
    s0_map : ndarray, shape (ny, nx)
        S0 in arbitrary input units (same as `data`).

    Notes
    -----
    Bias
        Equal weighting of `log(S)` samples means later, lower-SNR
        echoes contribute disproportionately. The estimator is
        consistent in the high-SNR limit but biased on real data.
        Treat the output as a robust *initial guess* and refine with
        `t2_nlls` if accuracy matters.

    Negative slopes
        A non-decaying signal (e.g. pure noise, or a voxel where T2
        recovery would be physically meaningless) yields a
        non-negative slope. These are zeroed in the output rather
        than reported as `inf` or negative T2.

    Implausible long T2
        A slope that is negative but very close to zero (e.g. a
        near-flat, low-SNR decay) inverts to a T2 of many seconds —
        not a real long-T2 tissue, just noise dominating a
        numerically unstable division. Rather than let a handful of
        such voxels blow out any downstream colour scale or summary
        statistic, fits with `T2 > t2_max` are rejected to 0, the
        same treatment already given to non-decaying (slope >= 0)
        voxels above.
    """
    data = np.asarray(data, dtype=float)
    te = np.asarray(te_list, dtype=float)
    n_te, ny, nx = data.shape

    mask = _build_mask(data, mask_threshold_frac)

    # log(S) for every voxel and every echo. The eps floor prevents
    # -inf at zero-signal voxels; we will discard those via the mask
    # anyway, but the lstsq below still needs finite inputs.
    log_s = np.log(np.maximum(data, eps))  # (n_te, ny, nx)
    flat = log_s.reshape(n_te, -1)  # (n_te, ny*nx)

    # Design matrix: columns are [TE, 1]. The slope solves for
    # -1/T2 and the intercept for log(S0).
    A = np.column_stack([te, np.ones_like(te)])  # (n_te, 2)

    # One least-squares solve over all pixels simultaneously.
    coeffs, *_ = np.linalg.lstsq(A, flat, rcond=None)  # (2, ny*nx)
    slope = coeffs[0].reshape(ny, nx)
    intercept = coeffs[1].reshape(ny, nx)

    # Convert (slope, intercept) -> (T2, S0). Reject non-decaying
    # voxels (slope >= 0) by leaving them at 0.
    t2_map = np.zeros((ny, nx))
    s0_map = np.zeros((ny, nx))
    valid = mask & (slope < 0)
    t2_map[valid] = -1.0 / slope[valid]
    s0_map[valid] = np.exp(intercept[valid])

    # Reject implausibly long T2 (see "Implausible long T2" above)
    # rather than letting it stand as a cosmetic-looking number.
    implausible = valid & (t2_map > t2_max)
    t2_map[implausible] = 0.0
    s0_map[implausible] = 0.0

    return t2_map, s0_map


# ---------------------------------------------------------------------------
# Non-linear refinement
# ---------------------------------------------------------------------------
def t2_nlls(
    data,
    te_list,
    mask_threshold_frac=0.1,
    t2_bounds=(0.0, 3000.0),
    init=None,
    maxfev=200,
):
    """
    Per-pixel non-linear least-squares T2 fit, optionally warm-started
    by a log-linear estimate.

    This fits `mono_exponential` directly to the magnitude data
    (no log transform), which gives a maximum-likelihood estimator
    under Gaussian noise and substantially reduces the noise-induced
    bias of `t2_loglinear`.

    Parameters
    ----------
    data : ndarray, shape (n_te, ny, nx)
        Magnitude image stack.
    te_list : array_like, shape (n_te,)
        Echo times in milliseconds.
    mask_threshold_frac : float, optional
        Fraction of the first-echo maximum used to mask background.
        Default 0.15.
    t2_bounds : tuple of float, optional
        (lower, upper) bounds on T2 in milliseconds. Default
        (0, 3000), which comfortably contains in-vivo brain CSF at
        3T (~2000 ms). Tighten for higher-field or phantom-only data.
    init : tuple of ndarray or None, optional
        Initial (s0_init, t2_init) maps, each shape (ny, nx). If
        `None`, `t2_loglinear` is called internally to produce them.
        Provide your own when chaining estimators or experimenting
        with custom initialisation.
    maxfev : int, optional
        Maximum function evaluations per pixel passed to
        `least_squares` (as `max_nfev`). Default 200 is plenty for
        warm-started fits; raise it if you see widespread
        non-convergence.

    Returns
    -------
    t2_map : ndarray, shape (ny, nx)
        T2 in milliseconds. Pixels where the fit failed to converge
        fall back to the log-linear estimate (so the map is never
        sparser than the warm-start map); pixels whose fit is pinned
        at the upper `t2_bounds` edge are rejected to 0 (see
        "Bound-pinned rejection" below).
    s0_map : ndarray, shape (ny, nx)
        S0 in input units, with the same fallback/rejection behaviour.

    Notes
    -----
    Fallback strategy (non-convergence)
        If `least_squares` fails to converge, the voxel keeps its
        log-linear estimate rather than being zeroed. This trades a
        bit of accuracy for spatial coverage and avoids speckled
        holes in noisy regions.

    Bound-pinned rejection (implausible fit)
        A converged fit sitting exactly at `t2_bounds[1]` did not
        find an interior optimum — the optimizer wanted to keep
        going and was stopped by the constraint. Reporting that as
        the voxel's T2 would draw a false, cosmetically-capped value
        (and would also blow out any downstream shared colour
        scale). `result.active_mask` says explicitly when this
        happened, so these voxels are rejected to 0 instead of kept
        — the same reasoning `t2_loglinear` uses to reject its own
        implausibly long fits, applied here via the optimizer's own
        constraint-activity report rather than a magnitude threshold.

    Performance
        Roughly 1-10 ms per voxel depending on TE count and machine.
        For a 96x96 slice this is ~10s; for a 256x256x N-slice volume
        it gets uncomfortable. If you need to scale, consider
        - thresholding more aggressively to skip irrelevant voxels,
        - parallelising over slices with `joblib` or `multiprocessing`,
        - or moving to a GPU implementation (PyTorch + Adam works well
          for mono-exponential fits and parallelises trivially).
    """
    data = np.asarray(data, dtype=float)
    te = np.asarray(te_list, dtype=float)
    n_te, ny, nx = data.shape

    mask = _build_mask(data, mask_threshold_frac)

    # Warm start. If the caller supplied init maps we trust them;
    # otherwise compute log-linear estimates first.
    if init is None:
        t2_init, s0_init = t2_loglinear(
            data, te, mask_threshold_frac=mask_threshold_frac
        )
    else:
        s0_init, t2_init = init

    t2_lo, t2_hi = t2_bounds

    # Output maps start from the warm-start estimates so failed fits
    # leave a sensible fallback rather than a hole.
    t2_map = t2_init.copy()
    s0_map = s0_init.copy()

    ys, xs = np.where(mask)
    for y, x in zip(ys, xs):
        # Clip the warm-start T2 into the bound range so least_squares
        # does not reject the initial guess outright.
        t2_guess = np.clip(t2_init[y, x] or 50.0, t2_lo + 1e-6, t2_hi - 1e-6)
        s0_guess = s0_init[y, x] or data[0, y, x]

        try:
            result = least_squares(
                _mono_exponential_residual,
                x0=[s0_guess, t2_guess],
                args=(te, data[:, y, x]),
                bounds=([t2_lo, t2_lo], [np.inf, t2_hi]),
                max_nfev=maxfev,
            )
        except (RuntimeError, ValueError):
            # Keep the log-linear fallback already in the output maps.
            continue

        if not result.success:
            # Non-convergence: keep the log-linear fallback.
            continue

        if result.active_mask[1] > 0:
            # Pinned at the upper T2 bound — the optimizer wanted to
            # go higher and was stopped by the constraint, so this is
            # not a real measurement. Reject it outright (see
            # "Bound-pinned rejection" above) rather than reporting
            # t2_bounds[1] back as if it were the fitted value.
            t2_map[y, x] = 0.0
            s0_map[y, x] = 0.0
            continue

        s0_map[y, x], t2_map[y, x] = result.x

    return t2_map, s0_map


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------
def create_t2_map(data, te_list, method="nlls", **kwargs):
    """
    Top-level entry point for T2 mapping.

    Parameters
    ----------
    data : ndarray, shape (n_te, ny, nx)
        Magnitude image stack.
    te_list : array_like, shape (n_te,)
        Echo times in milliseconds.
    method : {'loglinear', 'nlls'}, optional
        Estimator to use. `loglinear` is the fast vectorised fit;
        `nlls` is the slower but more accurate non-linear refinement
        warm-started by the log-linear result. Default `'nlls'`.
    **kwargs
        Forwarded to the chosen estimator (`mask_threshold_frac`,
        `t2_bounds`, etc.).

    Returns
    -------
    t2_map, s0_map : ndarray, ndarray
        Both shape (ny, nx). T2 in milliseconds, S0 in input units.

    Examples
    --------
    >>> t2, s0 = create_t2_map(images, TEs, method='loglinear')
    >>> t2, s0 = create_t2_map(images, TEs, method='nlls',
    ...                        mask_threshold_frac=0.02,
    ...                        t2_bounds=(0, 1600))
    """
    if method == "loglinear":
        return t2_loglinear(data, te_list, **kwargs)
    elif method == "nlls":
        return t2_nlls(data, te_list, **kwargs)
    else:
        raise ValueError(f"Unknown method {method!r}. Choose 'loglinear' or 'nlls'.")


def _compare_estimators(noise_levels=(0.0, 1.0, 5.0, 20.0), seed=0):
    """
    Generate a small synthetic phantom with three known-T2 regions and
    compare `t2_loglinear` vs. `t2_nlls` across noise levels.

    Prints a per-region accuracy / precision table. This exists mainly
    to give a quick gut check that the estimators are wired up
    correctly and to illustrate the noise-bias gap between them.

    Parameters
    ----------
    noise_levels : iterable of float
        Standard deviations (in S0 units; here 1000) of the additive
        Gaussian noise applied before taking magnitude. 0 disables
        noise entirely.
    seed : int
        Seed for `numpy`'s RNG. Fixed by default so the comparison is
        reproducible.
    """
    rng = np.random.default_rng(seed)

    # Three-region phantom. The bottom-right quadrant is region D and
    # gets the long T2 = 120 ms - it overlaps with B because we set
    # the long-T2 column after the short-T2 row.
    TEs = np.arange(20, 260, 10)
    ny, nx = 32, 32
    true_t2 = np.full((ny, nx), 80.0)
    true_t2[:16, :] = 40.0  # top half:    short T2
    true_t2[:, 16:] = 120.0  # right half:  long T2 (overrides top-right)
    true_s0 = np.full((ny, nx), 1000.0)

    # Define ROIs as the three non-overlapping quadrants where each
    # T2 is the dominant value. Region B is top-right (long T2),
    # region A is top-left (short), region C is bottom-left (mid).
    rois = {
        "A (true 40 ms)": (slice(0, 16), slice(0, 16)),
        "B (true 120 ms)": (slice(0, 16), slice(16, 32)),
        "C (true 80 ms)": (slice(16, 32), slice(0, 16)),
    }
    truth = {"A (true 40 ms)": 40.0, "B (true 120 ms)": 120.0, "C (true 80 ms)": 80.0}

    # Noise-free signal stack, shape (n_te, ny, nx)
    clean = true_s0[None] * np.exp(-TEs[:, None, None] / true_t2[None])

    header = f"{'noise sigma':>12} | {'method':>10} | {'region':<18} | {'mean':>7} | {'std':>6} | {'bias':>7}"
    print(header)
    print("-" * len(header))

    for sigma in noise_levels:
        if sigma > 0:
            noise = rng.standard_normal(clean.shape) * sigma
            data = np.abs(clean + noise)
        else:
            data = clean.copy()

        t2_ll, _ = create_t2_map(
            data, TEs, method="loglinear", mask_threshold_frac=0.05
        )
        t2_nl, _ = create_t2_map(data, TEs, method="nlls", mask_threshold_frac=0.05)

        for label, sl in rois.items():
            true_val = truth[label]
            for method_name, t2_map in [("loglinear", t2_ll), ("nlls", t2_nl)]:
                roi = t2_map[sl]
                # Exclude masked-out (zero) voxels from the statistics.
                valid = roi[roi > 0]
                if valid.size == 0:
                    mean, std, bias = float("nan"), float("nan"), float("nan")
                else:
                    mean = valid.mean()
                    std = valid.std()
                    bias = mean - true_val
                print(
                    f"{sigma:>12.1f} | {method_name:>10} | {label:<18} | "
                    f"{mean:>7.2f} | {std:>6.2f} | {bias:>+7.2f}"
                )
        print()


if __name__ == "__main__":
    _compare_estimators()
