"""
diffusion_utils.py
==================

IQ-BRAIN is funded by the European Union (MSCA Doctoral Network,
December 2024–November 2028, Grant Agreement No. 101169519).

Voxel-wise ADC fitting for diffusion-weighted MRI. Mirrors the structure of
``relaxometry_utils.create_t2_map``: a single public entry point with a
``method`` switch that dispatches to either non-linear least squares or a
closed-form log-linear fit.

Model
-----
    S(b) = S0 * exp(-b * ADC)

with ``b`` in s/mm^2 and ``ADC`` in mm^2/s. The function expects a stack of
magnitude trace-DWI images (one per b-value) - i.e. diffusion directions
should already be combined (typically via geometric mean) before calling
this. If you pass per-direction data instead, you will get a directional
ADC, not mean diffusivity.

Two fitters
-----------
- ``nlls``      : scipy.optimize.curve_fit on the exponential. Naturally
                  bounded and robust to Gaussian-like noise.
- ``loglinear`` : ordinary least squares on ln(S) = ln(S0) - b * ADC.
                  Closed form and very fast, but heteroscedastic in the
                  original signal domain - the log transform over-weights
                  low-SNR high-b points, biasing ADC upward at low SNR.
                  Useful as a sanity check or as initialization for NLLS.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def stejskal_tanner(b, s0, adc):
    """Mono-exponential diffusion signal decay.

    S(b) = S0 * exp(-b * ADC)
    """
    return s0 * np.exp(-b * adc)


# ---------------------------------------------------------------------------
# Per-voxel fitters
# ---------------------------------------------------------------------------
def _fit_adc_nlls(data, b_values, threshold_frac=0.1, adc_init=1e-3, adc_max=5e-3):
    """Per-pixel non-linear least squares fit of S(b) = S0 * exp(-b * ADC).

    Parameters
    ----------
    data : (n_b, ny, nx) ndarray
        Magnitude trace-DWI images, one per b-value.
    b_values : (n_b,) array-like
        b-values in s/mm^2, same order as ``data``'s first axis.
    threshold_frac : float
        Background mask threshold as a fraction of the b=0 image maximum.
        Voxels below this on the b=0 image are skipped (left as 0).
    adc_init : float
        Initial guess for ADC in mm^2/s. 1e-3 ≈ typical brain parenchyma.
    adc_max : float
        Upper bound on ADC. Free water at 37 °C is ~3e-3 mm^2/s, so 5e-3
        is a safe physiological ceiling.

    Returns
    -------
    adc_map : (ny, nx) ndarray
        ADC in mm^2/s. Background and failed-fit voxels are 0.
    s0_map : (ny, nx) ndarray
        Fitted S0.
    """
    n_b, ny, nx = data.shape
    adc_map = np.zeros((ny, nx))
    s0_map = np.zeros((ny, nx))

    b_arr = np.asarray(b_values, dtype=float)

    # Use the b=0 image as the SNR proxy for the background mask. If b=0
    # isn't in the list (rare), fall back to the smallest b-value.
    b_list = list(b_arr)
    if 0.0 in b_list:
        b0_idx = b_list.index(0.0)
    else:
        b0_idx = int(np.argmin(b_arr))
    threshold = np.max(data[b0_idx]) * threshold_frac

    for y in range(ny):
        for x in range(nx):
            pixel_series = data[:, y, x]

            # Skip background / low-SNR voxels.
            if pixel_series[b0_idx] <= threshold:
                continue

            try:
                popt, _ = curve_fit(
                    stejskal_tanner,
                    b_arr,
                    pixel_series,
                    p0=[pixel_series[b0_idx], adc_init],
                    bounds=(0, [np.inf, adc_max]),
                )
                s0_map[y, x], adc_map[y, x] = popt
            except Exception:
                # Non-convergent fit - leave this voxel as 0.
                continue

    return adc_map, s0_map


def _fit_adc_loglinear(data, b_values, threshold_frac=0.1):
    """Vectorised log-linear ADC fit: ln(S) = ln(S0) - b * ADC.

    Closed-form OLS over all voxels at once - much faster than NLLS but
    biased at low SNR because the log transform makes residuals
    heteroscedastic (high-b noisy points dominate).

    Parameters
    ----------
    data : (n_b, ny, nx) ndarray
        Magnitude trace-DWI images. Non-positive values are clipped to a
        small epsilon before taking the log.
    b_values : (n_b,) array-like
    threshold_frac : float
        Background mask threshold on the b=0 image.

    Returns
    -------
    adc_map : (ny, nx) ndarray   [mm^2/s]
    s0_map  : (ny, nx) ndarray
    """
    n_b, ny, nx = data.shape
    b_arr = np.asarray(b_values, dtype=float)

    b_list = list(b_arr)
    if 0.0 in b_list:
        b0_idx = b_list.index(0.0)
    else:
        b0_idx = int(np.argmin(b_arr))
    threshold = np.max(data[b0_idx]) * threshold_frac
    mask = data[b0_idx] > threshold  # (ny, nx)

    # ln(S) is undefined at 0; clip first.
    eps = 1e-12
    log_S = np.log(np.maximum(data, eps))  # (n_b, ny, nx)

    # Design matrix: [1, -b]; OLS for ln(S0) and ADC per voxel.
    # Stack into (n_b, 2) and solve once for all voxels using lstsq on the
    # flattened (n_b, n_voxels) matrix.
    X = np.column_stack([np.ones_like(b_arr), -b_arr])  # (n_b, 2)
    Y = log_S.reshape(n_b, -1)  # (n_b, ny*nx)

    # lstsq returns coeffs of shape (2, n_voxels): [ln(S0); ADC].
    coeffs, *_ = np.linalg.lstsq(X, Y, rcond=None)
    ln_s0 = coeffs[0].reshape(ny, nx)
    adc = coeffs[1].reshape(ny, nx)

    s0_map = np.exp(ln_s0)
    adc_map = adc

    # Apply the background mask and clamp to physiological range. Negative
    # ADCs are an artefact of noise-dominated voxels and are zeroed here
    # rather than left as garbage, matching how the NLLS path handles
    # failed fits.
    adc_map = np.where(mask, adc_map, 0.0)
    adc_map = np.where((adc_map < 0) | (adc_map > 5e-3), 0.0, adc_map)
    s0_map = np.where(mask, s0_map, 0.0)

    return adc_map, s0_map


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def create_adc_map(data, b_values, method="nlls", **kwargs):
    """Build a quantitative ADC map from a stack of trace-DWI images.

    Parameters
    ----------
    data : (n_b, ny, nx) ndarray
        Magnitude trace-DWI images, one per b-value. Directions must
        already be combined (e.g. via geometric mean across the
        directions axis) before calling.
    b_values : (n_b,) array-like
        b-values in s/mm^2, in the same order as the first axis of
        ``data``.
    method : {'nlls', 'loglinear'}
        Fitting method.
    **kwargs
        Forwarded to the underlying fitter (e.g. ``threshold_frac``,
        ``adc_init``, ``adc_max`` for NLLS).

    Returns
    -------
    adc_map : (ny, nx) ndarray   ADC in mm^2/s
    s0_map  : (ny, nx) ndarray   Fitted S0
    """
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(f"`data` must be (n_b, ny, nx); got shape {data.shape}")
    if len(b_values) != data.shape[0]:
        raise ValueError(
            f"len(b_values)={len(b_values)} does not match data.shape[0]={data.shape[0]}"
        )

    method = method.lower()
    if method == "nlls":
        return _fit_adc_nlls(data, b_values, **kwargs)
    elif method == "loglinear":
        return _fit_adc_loglinear(data, b_values, **kwargs)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'nlls' or 'loglinear'.")


# ---------------------------------------------------------------------------
# Diffusion tensor (DTI) → FA and MD
# ---------------------------------------------------------------------------
def create_dti_maps(
    data_per_dir, b_values, b_vectors, threshold_frac=0.1, md_max=5e-3
):
    """Voxel-wise log-linear diffusion tensor fit, returning FA and MD maps.

    Model
    -----
        ln S(b, g) = ln S0 - b * g^T D g

    where D is the symmetric 3x3 diffusion tensor (6 unique components). The
    fit is closed-form per voxel (single OLS solve, vectorised across all
    voxels) and uses every (b, direction) measurement jointly - no need for
    a separate trace-DWI step.

    Two scalar invariants of D are returned:
      - Mean diffusivity (MD) = trace(D) / 3 = (lam1 + lam2 + lam3) / 3.
        Rotation-invariant analogue of the directional ADC.
      - Fractional anisotropy (FA) in [0, 1] (Basser/Pierpaoli):
            FA = sqrt(3/2) * ||D - MD*I||_F / ||D||_F
        evaluated on the eigenvalues. 0 for isotropic, 1 for a stick.

    Parameters
    ----------
    data_per_dir : (n_b, n_dir, ny, nx) ndarray
        Magnitude DWI images, one per (b-value, direction) pair.
    b_values : (n_b,) array-like
        b-values in s/mm^2, in the same order as the first axis of ``data_per_dir``.
    b_vectors : (n_dir, 3) array-like
        Unit gradient directions, in the same order as the second axis of
        ``data_per_dir``. Need not be orthogonal but must span 3D for the
        tensor to be identifiable (>= 6 non-collinear directions).
    threshold_frac : float
        Background mask threshold as a fraction of the maximum on the
        lowest-b mean image. Voxels below this are zeroed.
    md_max : float
        Upper physiological MD bound; voxels exceeding this are zeroed
        (matches the ``adc_max`` convention in the ADC fitters).

    Returns
    -------
    fa_map  : (ny, nx) ndarray   FA in [0, 1]
    md_map  : (ny, nx) ndarray   MD in mm^2/s
    eigvals : (ny, nx, 3) ndarray   eigenvalues sorted descending
    s0_map  : (ny, nx) ndarray   fitted S0
    """
    data = np.asarray(data_per_dir, dtype=float)
    if data.ndim != 4:
        raise ValueError(
            f"`data_per_dir` must be (n_b, n_dir, ny, nx); got shape {data.shape}"
        )
    n_b, n_dir, ny, nx = data.shape

    b_arr = np.asarray(b_values, dtype=float)
    if b_arr.shape != (n_b,):
        raise ValueError(
            f"len(b_values)={b_arr.shape} does not match data.shape[0]={n_b}"
        )

    g = np.asarray(b_vectors, dtype=float)
    if g.shape != (n_dir, 3):
        raise ValueError(
            f"`b_vectors` must be ({n_dir}, 3); got shape {g.shape}"
        )

    # Design matrix X of shape (n_b * n_dir, 7). Each row encodes the
    # contribution of one (b_i, g_j) measurement to the unknowns
    # [ln S0, Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]. The off-diagonal entries get a
    # factor of 2 because g^T D g expands to
    #     Dxx*gx^2 + Dyy*gy^2 + Dzz*gz^2 + 2*Dxy*gx*gy + 2*Dxz*gx*gz + 2*Dyz*gy*gz.
    bb = np.repeat(b_arr, n_dir)                      # (N,)
    gg = np.tile(g, (n_b, 1))                         # (N, 3)
    gx, gy, gz = gg[:, 0], gg[:, 1], gg[:, 2]
    X = np.column_stack(
        [
            np.ones_like(bb),
            -bb * gx * gx,
            -bb * gy * gy,
            -bb * gz * gz,
            -2.0 * bb * gx * gy,
            -2.0 * bb * gx * gz,
            -2.0 * bb * gy * gz,
        ]
    )  # (N, 7)

    eps = 1e-12
    log_S = np.log(np.maximum(data, eps))             # (n_b, n_dir, ny, nx)
    Y = log_S.reshape(n_b * n_dir, ny * nx)           # (N, n_voxels)

    coeffs, *_ = np.linalg.lstsq(X, Y, rcond=None)    # (7, n_voxels)
    ln_s0 = coeffs[0]
    Dxx, Dyy, Dzz = coeffs[1], coeffs[2], coeffs[3]
    Dxy, Dxz, Dyz = coeffs[4], coeffs[5], coeffs[6]

    # Assemble per-voxel symmetric tensor and diagonalise. ``eigvalsh``
    # exploits symmetry and is more stable than ``eig``; it returns
    # eigenvalues in ascending order, so reverse for descending.
    n_vox = ny * nx
    D = np.empty((n_vox, 3, 3))
    D[:, 0, 0] = Dxx
    D[:, 1, 1] = Dyy
    D[:, 2, 2] = Dzz
    D[:, 0, 1] = D[:, 1, 0] = Dxy
    D[:, 0, 2] = D[:, 2, 0] = Dxz
    D[:, 1, 2] = D[:, 2, 1] = Dyz
    eigvals = np.linalg.eigvalsh(D)[:, ::-1]          # (n_vox, 3), descending

    md = eigvals.mean(axis=1)                         # (n_vox,)
    diff = eigvals - md[:, None]
    num = (diff * diff).sum(axis=1)
    den = (eigvals * eigvals).sum(axis=1)
    fa = np.sqrt(np.where(den > 0, 1.5 * num / den, 0.0))
    fa = np.clip(fa, 0.0, 1.0)

    fa_map = fa.reshape(ny, nx)
    md_map = md.reshape(ny, nx)
    eigvals_map = eigvals.reshape(ny, nx, 3)
    s0_map = np.exp(ln_s0).reshape(ny, nx)

    # Background mask: use the lowest-b image (averaged across directions)
    # as the SNR proxy, since b=0 is not guaranteed to be present.
    b0_idx = int(np.argmin(b_arr))
    snr_proxy = data[b0_idx].mean(axis=0)             # (ny, nx)
    mask = snr_proxy > (snr_proxy.max() * threshold_frac)

    md_map = np.where(mask & (md_map > 0) & (md_map < md_max), md_map, 0.0)
    fa_map = np.where(mask, fa_map, 0.0)
    s0_map = np.where(mask, s0_map, 0.0)

    return fa_map, md_map, eigvals_map, s0_map
