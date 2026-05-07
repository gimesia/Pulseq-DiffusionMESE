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
    stejskal_tanner,
)
from utils_relaxometry import (
    create_t2_map,
    mono_exponential,
    t2_loglinear,
)


# ──────────────────────────────────────────────────────────────────────────
# Diffusion forward model
# ──────────────────────────────────────────────────────────────────────────


def test_stejskal_tanner_at_b0():
    """b=0 → signal equals S0 regardless of ADC."""
    assert stejskal_tanner(0.0, 1000.0, 1e-3) == pytest.approx(1000.0)


def test_stejskal_tanner_decays_with_b():
    """Signal must decrease monotonically as b increases."""
    b_values = np.array([0.0, 500.0, 1000.0, 2000.0])
    signals = stejskal_tanner(b_values, s0=1000.0, adc=1e-3)
    assert np.all(np.diff(signals) < 0)


# ──────────────────────────────────────────────────────────────────────────
# ADC map fitting
# ──────────────────────────────────────────────────────────────────────────


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
    # Zero out one voxel across all b-values
    data[:, 0, 0] = 0.0
    adc_map, _ = create_adc_map(data, b_values, method="nlls", threshold_frac=0.1)
    assert adc_map[0, 0] == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────────────────────
# T2 forward model
# ──────────────────────────────────────────────────────────────────────────


def test_mono_exponential_at_te0():
    """te=0 → signal equals S0."""
    assert mono_exponential(0.0, s0=1000.0, t2=80.0) == pytest.approx(1000.0)


def test_mono_exponential_decays_with_te():
    """Signal must decrease monotonically as TE increases."""
    te_values = np.array([20.0, 40.0, 80.0, 160.0])
    signals = mono_exponential(te_values, s0=1000.0, t2=80.0)
    assert np.all(np.diff(signals) < 0)


# ──────────────────────────────────────────────────────────────────────────
# T2 map fitting
# ──────────────────────────────────────────────────────────────────────────


def _synthetic_mese(true_t2=80.0, s0=1000.0, ny=4, nx=4):
    """Create a noise-free (n_te, ny, nx) multi-echo stack with known T2."""
    te_list = np.arange(20.0, 200.0, 20.0)  # 9 echoes, 20–180 ms
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
    # Constant signal across all TEs (slope = 0, not negative)
    data = np.ones((4, 4, 4)) * 1000.0
    t2_map, _ = t2_loglinear(data, te_list, mask_threshold_frac=0.01)
    # All voxels should be 0 because slope >= 0
    assert np.all(t2_map == pytest.approx(0.0))


def test_t2_background_masked():
    """Zero-signal voxels must be excluded; T2 map returns 0 for background."""
    data, te_list = _synthetic_mese()
    data[:, 0, 0] = 0.0  # blank out one voxel
    t2_map, _ = t2_loglinear(data, te_list, mask_threshold_frac=0.1)
    assert t2_map[0, 0] == pytest.approx(0.0)
