"""Tests for simulation/utils_diffusion.py and simulation/utils_relaxometry.py.

All tests use synthetic numpy arrays; no MRzeroCore or hardware is required.
The simulation/ directory is added to sys.path by conftest.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add simulation/ to path (mirrors the conftest pattern for pulseq_diffusion_mese/)
SIM_DIR = Path(__file__).resolve().parent.parent / "simulation"
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

from utils_diffusion import (
    create_adc_map,
    create_dti_maps,
    stejskal_tanner,
)
from utils_relaxometry import (
    _compare_estimators,
    create_t2_map,
    mono_exponential,
    t2_loglinear,
    t2_nlls,
)


# --------------------------------------------------------------------------
# Diffusion forward model
# --------------------------------------------------------------------------


def test_stejskal_tanner_at_b0():
    """b=0 -> signal equals S0 regardless of ADC."""
    assert stejskal_tanner(0.0, 1000.0, 1e-3) == pytest.approx(1000.0)


def test_stejskal_tanner_decays_with_b():
    """Signal must decrease monotonically as b increases."""
    b_values = np.array([0.0, 500.0, 1000.0, 2000.0])
    signals = stejskal_tanner(b_values, s0=1000.0, adc=1e-3)
    assert np.all(np.diff(signals) < 0)


# --------------------------------------------------------------------------
# ADC map fitting
# --------------------------------------------------------------------------


def _synthetic_dwi(true_adc=1.0e-3, s0=1000.0, ny=4, nx=4):
    """Create a noise-free (n_b, ny, nx) DWI stack with known ADC."""
    b_values = np.array([0.0, 500.0, 1000.0, 2000.0])
    data = s0 * np.exp(-b_values[:, None, None] * true_adc) * np.ones((1, ny, nx))
    return data, b_values


@pytest.mark.parametrize("method", ["nlls", "loglinear"])
def test_create_adc_map_noisefree_recovers_true_adc(method):
    """Both fitters must recover the ground-truth ADC within 1% on noise-free data."""
    true_adc = 1.0e-3
    data, b_values = _synthetic_dwi(true_adc)
    adc_map, s0_map = create_adc_map(data, b_values, method=method)

    foreground = adc_map[adc_map > 0]
    assert foreground.size > 0
    np.testing.assert_allclose(foreground, true_adc, rtol=0.01)


def test_create_adc_map_invalid_method_raises():
    data, b_values = _synthetic_dwi()
    with pytest.raises(ValueError, match="Unknown method"):
        create_adc_map(data, b_values, method="bad_method")


def test_create_adc_map_shape_mismatch_raises():
    data = np.ones((4, 4, 4))
    with pytest.raises(ValueError):
        create_adc_map(data, b_values=[0, 500], method="nlls")


def test_create_adc_map_wrong_ndim_raises():
    data = np.ones((4, 4))  # 2-D, not 3-D
    with pytest.raises(ValueError):
        create_adc_map(data, b_values=[0, 500, 1000, 2000], method="nlls")


def test_create_adc_map_background_masked():
    """Zero-signal voxels must be left as 0 (background mask)."""
    data, b_values = _synthetic_dwi()
    data[:, 0, 0] = 0.0
    adc_map, _ = create_adc_map(data, b_values, method="nlls", threshold_frac=0.1)
    assert adc_map[0, 0] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# T2 forward model
# --------------------------------------------------------------------------


def test_mono_exponential_at_te0():
    """te=0 -> signal equals S0."""
    assert mono_exponential(0.0, s0=1000.0, t2=80.0) == pytest.approx(1000.0)


def test_mono_exponential_decays_with_te():
    """Signal must decrease monotonically as TE increases."""
    te_values = np.array([20.0, 40.0, 80.0, 160.0])
    signals = mono_exponential(te_values, s0=1000.0, t2=80.0)
    assert np.all(np.diff(signals) < 0)


# --------------------------------------------------------------------------
# T2 map fitting
# --------------------------------------------------------------------------


def _synthetic_mese(true_t2=80.0, s0=1000.0, ny=4, nx=4):
    """Create a noise-free (n_te, ny, nx) multi-echo stack with known T2."""
    te_list = np.arange(20.0, 200.0, 20.0)  # 9 echoes, 20-180 ms
    data = s0 * np.exp(-te_list[:, None, None] / true_t2) * np.ones((1, ny, nx))
    return data, te_list


@pytest.mark.parametrize("method", ["nlls", "loglinear"])
def test_create_t2_map_noisefree_recovers_true_t2(method):
    """Both T2 fitters must recover ground-truth T2 within 1% on noise-free data."""
    true_t2 = 80.0
    data, te_list = _synthetic_mese(true_t2)
    t2_map, s0_map = create_t2_map(data, te_list, method=method)

    foreground = t2_map[t2_map > 0]
    assert foreground.size > 0
    np.testing.assert_allclose(foreground, true_t2, rtol=0.01)


def test_create_t2_map_invalid_method_raises():
    data, te_list = _synthetic_mese()
    with pytest.raises(ValueError):
        create_t2_map(data, te_list, method="not_a_method")


def test_t2_loglinear_non_decaying_signal_zeroed():
    """A constant (flat) signal has no T2 decay; the fit must return T2=0."""
    te_list = np.array([20.0, 60.0, 100.0, 140.0])
    data = np.ones((4, 4, 4)) * 1000.0
    t2_map, _ = t2_loglinear(data, te_list, mask_threshold_frac=0.01)
    assert np.all(t2_map == pytest.approx(0.0))


def test_t2_background_masked():
    """Zero-signal voxels must be excluded; T2 map returns 0 for background."""
    data, te_list = _synthetic_mese()
    data[:, 0, 0] = 0.0
    t2_map, _ = t2_loglinear(data, te_list, mask_threshold_frac=0.1)
    assert t2_map[0, 0] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Realistic in-vivo brain values (3 T)
#   T2:  WM ~70-80, GM ~80-100, CSF ~1800-2200 ms (Wansapura 1999; Stanisz 2005)
#   ADC: WM ~0.7-0.9e-3, GM ~0.8-1.0e-3, CSF ~3e-3 mm^2/s (Sener 2001; Helenius 2002)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tissue, true_t2", [("WM", 75.0), ("GM", 90.0), ("CSF", 2000.0)]
)
def test_t2_nlls_recovers_brain_tissue_t2(tissue, true_t2):
    """NLLS must recover physiological T2 (WM ~75, GM ~90, CSF ~2000 ms) with low noise."""
    rng = np.random.default_rng(0)
    te_list = np.arange(20.0, 320.0, 20.0)
    s0 = 1000.0
    clean = s0 * np.exp(-te_list[:, None, None] / true_t2) * np.ones((1, 6, 6))
    noisy = np.abs(clean + rng.standard_normal(clean.shape) * 2.0)
    t2_map, _ = t2_nlls(noisy, te_list, mask_threshold_frac=0.01, t2_bounds=(0.0, 4000.0))
    valid = t2_map[t2_map > 0]
    assert valid.size > 0
    np.testing.assert_allclose(valid.mean(), true_t2, rtol=0.10)


def test_t2_loglinear_biased_high_under_noise():
    """Log-linear estimator stays within 25% of truth under appreciable noise.

    Bias toward shorter T2 documented by Whittall & MacKay (JMR 1989).
    """
    rng = np.random.default_rng(1)
    te_list = np.arange(20.0, 320.0, 20.0)
    true_t2 = 80.0
    s0 = 1000.0
    clean = s0 * np.exp(-te_list[:, None, None] / true_t2) * np.ones((1, 8, 8))
    noisy = np.abs(clean + rng.standard_normal(clean.shape) * 30.0)
    t2_map, _ = t2_loglinear(noisy, te_list, mask_threshold_frac=0.01)
    valid = t2_map[t2_map > 0]
    assert valid.size > 0
    assert 0.75 * true_t2 < valid.mean() < 1.25 * true_t2


def test_t2_nlls_explicit_init_path():
    """t2_nlls accepts user-supplied (s0_init, t2_init) warm-start maps."""
    data, te_list = _synthetic_mese(true_t2=80.0)
    s0_init = np.full(data.shape[1:], 900.0)
    t2_init = np.full(data.shape[1:], 70.0)
    t2_map, s0_map = t2_nlls(data, te_list, init=(s0_init, t2_init))
    foreground = t2_map[t2_map > 0]
    np.testing.assert_allclose(foreground, 80.0, rtol=0.01)
    np.testing.assert_allclose(s0_map[t2_map > 0], 1000.0, rtol=0.01)


def test_create_t2_map_dispatches_to_nlls_by_default():
    data, te_list = _synthetic_mese(true_t2=100.0)
    t2_default, _ = create_t2_map(data, te_list)
    t2_nl, _ = create_t2_map(data, te_list, method="nlls")
    np.testing.assert_allclose(t2_default, t2_nl)


# --------------------------------------------------------------------------
# ADC fitting - noise behaviour
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tissue, true_adc",
    [("WM", 0.8e-3), ("GM", 0.9e-3), ("CSF", 3.0e-3)],
)
def test_create_adc_map_recovers_brain_tissue_adc(tissue, true_adc):
    """NLLS must recover physiological ADC values to within 10%."""
    rng = np.random.default_rng(0)
    b_values = np.array([0.0, 250.0, 500.0, 750.0, 1000.0, 1500.0])
    s0 = 1000.0
    clean = s0 * np.exp(-b_values[:, None, None] * true_adc) * np.ones((1, 6, 6))
    noisy = np.abs(clean + rng.standard_normal(clean.shape) * 2.0)
    adc_map, _ = create_adc_map(noisy, b_values, method="nlls", threshold_frac=0.01)
    valid = adc_map[adc_map > 0]
    assert valid.size > 0
    np.testing.assert_allclose(valid.mean(), true_adc, rtol=0.10)


def test_loglinear_handles_no_b0_in_list():
    """Falls back to smallest b-value when b=0 is missing."""
    b_values = np.array([100.0, 500.0, 1000.0])
    true_adc = 1.0e-3
    s0 = 1000.0
    data = s0 * np.exp(-b_values[:, None, None] * true_adc) * np.ones((1, 4, 4))
    adc_map, _ = create_adc_map(data, b_values, method="loglinear")
    valid = adc_map[adc_map > 0]
    np.testing.assert_allclose(valid, true_adc, rtol=0.01)


def test_loglinear_clips_unphysical_voxels():
    """ADC values below adc_min=0 must be zeroed (free water ~3e-3 at 37 C)."""
    b_values = np.array([0.0, 500.0, 1000.0])
    data = np.ones((3, 2, 2)) * 1000.0
    # Voxel (0,0): signal grows -> negative ADC -> zeroed by adc_min=0 default
    data[1, 0, 0] = 1200.0
    data[2, 0, 0] = 1500.0
    adc_map, _ = create_adc_map(data, b_values, method="loglinear")
    assert adc_map[0, 0] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# DTI: FA and MD
# --------------------------------------------------------------------------


def _six_dir_scheme():
    """Six non-collinear unit vectors that span 3D - minimum for DTI."""
    vecs = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]],
        dtype=float,
    )
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def _synthetic_dti(D, b_values, b_vectors, s0=1000.0, ny=4, nx=4):
    """Forward-simulate DWI signals from a known 3x3 tensor D."""
    n_b = len(b_values)
    n_dir = b_vectors.shape[0]
    data = np.empty((n_b, n_dir, ny, nx))
    for i, b in enumerate(b_values):
        for j, g in enumerate(b_vectors):
            data[i, j] = s0 * np.exp(-b * (g @ D @ g))
    return data


def test_dti_isotropic_tensor_yields_zero_fa():
    """Isotropic D = adc*I should produce FA ~0 and MD = adc everywhere."""
    adc = 0.8e-3
    D = adc * np.eye(3)
    b_values = np.array([0.0, 500.0, 1000.0])
    b_vectors = _six_dir_scheme()
    data = _synthetic_dti(D, b_values, b_vectors)

    fa_map, md_map, eigvals, s0_map = create_dti_maps(data, b_values, b_vectors)
    fg = md_map > 0
    np.testing.assert_allclose(md_map[fg], adc, rtol=0.02)
    np.testing.assert_allclose(fa_map[fg], 0.0, atol=1e-6)
    np.testing.assert_allclose(s0_map[fg], 1000.0, rtol=0.02)


def test_dti_stick_yields_fa_near_one():
    """A near-unidirectional tensor must yield FA > 0.95 (Basser 1994)."""
    lam = 1.7e-3
    D = np.diag([lam, 1e-6, 1e-6])
    b_values = np.array([0.0, 500.0, 1000.0])
    b_vectors = _six_dir_scheme()
    data = _synthetic_dti(D, b_values, b_vectors)

    fa_map, md_map, _, _ = create_dti_maps(data, b_values, b_vectors)
    fg = md_map > 0
    assert fa_map[fg].mean() > 0.95


def test_dti_wm_like_anisotropy():
    """Realistic WM tensor: AD~1.7e-3, RD~0.4e-3 -> FA~0.71 (Pierpaoli 1996)."""
    AD, RD = 1.7e-3, 0.4e-3
    D = np.diag([AD, RD, RD])
    b_values = np.array([0.0, 500.0, 1000.0])
    b_vectors = _six_dir_scheme()
    data = _synthetic_dti(D, b_values, b_vectors)

    fa_map, md_map, _, _ = create_dti_maps(data, b_values, b_vectors)
    fg = md_map > 0
    expected_md = (AD + 2 * RD) / 3
    np.testing.assert_allclose(md_map[fg].mean(), expected_md, rtol=0.02)
    assert 0.65 < fa_map[fg].mean() < 0.80


def test_dti_input_validation():
    b_vectors = _six_dir_scheme()
    b_values = np.array([0.0, 1000.0])
    good = _synthetic_dti(np.eye(3) * 1e-3, b_values, b_vectors)

    with pytest.raises(ValueError):
        create_dti_maps(good[0], b_values, b_vectors)  # wrong ndim

    with pytest.raises(ValueError):
        create_dti_maps(good, [0.0], b_vectors)  # mismatched b_values

    with pytest.raises(ValueError):
        create_dti_maps(good, b_values, b_vectors[:, :2])  # wrong b_vectors shape


# --------------------------------------------------------------------------
# _compare_estimators demo helper
# --------------------------------------------------------------------------


def test_compare_estimators_runs_without_error(capsys):
    """Smoke test: the demo helper should print a table for two noise levels."""
    _compare_estimators(noise_levels=(0.0, 5.0), seed=0)
    captured = capsys.readouterr()
    assert "loglinear" in captured.out
    assert "nlls" in captured.out


def test_compare_estimators_empty_roi_path(capsys):
    """When an ROI is completely masked the helper must not crash (nan branch)."""
    import utils_relaxometry as _rel
    _orig_create = _rel.create_t2_map

    def _always_zero(data, te_list, **kwargs):
        ny, nx = data.shape[1], data.shape[2]
        return np.zeros((ny, nx)), np.zeros((ny, nx))

    _rel.create_t2_map = _always_zero
    try:
        _compare_estimators(noise_levels=(0.0,), seed=0)
    finally:
        _rel.create_t2_map = _orig_create

    out = capsys.readouterr().out
    assert "nan" in out.lower()


# --------------------------------------------------------------------------
# Exception-handling fallback paths
# --------------------------------------------------------------------------


def test_nlls_adc_exception_fallback():
    """Pathological signal with impossibly tight bounds must not crash; voxel stays 0."""
    b_values = np.array([0.0, 500.0, 1000.0])
    data = np.ones((3, 2, 2)) * 500.0
    adc_map, _ = create_adc_map(
        data, b_values, method="nlls", threshold_frac=0.0,
        adc_init=1e-3, adc_max=1e-9,   # upper bound < init -> ValueError in curve_fit
    )
    assert np.all(adc_map == pytest.approx(0.0))


def test_nlls_adc_no_b0_fallback():
    """NLLS: when b=0 absent, falls back to smallest b-value for mask."""
    b_values = np.array([100.0, 500.0, 1000.0])
    true_adc = 1.0e-3
    s0 = 1000.0
    data = s0 * np.exp(-b_values[:, None, None] * true_adc) * np.ones((1, 4, 4))
    adc_map, _ = create_adc_map(data, b_values, method="nlls")
    valid = adc_map[adc_map > 0]
    np.testing.assert_allclose(valid, true_adc, rtol=0.01)


def test_t2_nlls_exception_fallback():
    """Constant signal + impossibly tight t2_bounds must not raise; result is finite."""
    te_list = np.array([20.0, 60.0, 100.0, 140.0])
    data = np.ones((4, 2, 2)) * 1000.0  # constant -> slope=0
    t2_map, _ = t2_nlls(data, te_list, mask_threshold_frac=0.01, t2_bounds=(0.0, 1.0))
    assert np.all(np.isfinite(t2_map))
