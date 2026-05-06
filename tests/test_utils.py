"""Pure-function tests for utils.py (raster alignment, b-value math, etc.)."""

from __future__ import annotations

import numpy as np
import pypulseq as pp
import pytest

from utils import (
    SystemLimitType,
    align2rastertime_ceil,
    align2rastertime_floor,
    align2rastertime_nearest,
    bFactCalc,
    calc_area_preserving_trapezoid,
    calc_bval,
    calc_diffusion_gradient_amplitude,
    deg2rad,
    get_diffusion_directions,
    system_limit,
)


# ──────────────────────────────────────────────────────────────────────────
# Raster alignment
# ──────────────────────────────────────────────────────────────────────────

class TestRasterAlignment:
    @pytest.mark.parametrize(
        "x, rt, expected",
        [
            (0.0, 1e-5, 0.0),
            (1e-5, 1e-5, 1e-5),
            (1.5e-5, 1e-5, 2e-5),
            (1e-7, 1e-5, 1e-5),
        ],
    )
    def test_ceil(self, x, rt, expected):
        assert align2rastertime_ceil(x, rt) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "x, rt, expected",
        [
            (0.0, 1e-5, 0.0),
            (1e-5, 1e-5, 1e-5),
            (1.5e-5, 1e-5, 1e-5),
            (1.99e-5, 1e-5, 1e-5),
        ],
    )
    def test_floor(self, x, rt, expected):
        assert align2rastertime_floor(x, rt) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "x, rt, expected",
        [
            (1.4e-5, 1e-5, 1e-5),
            (1.5e-5, 1e-5, 2e-5),
            (1.6e-5, 1e-5, 2e-5),
        ],
    )
    def test_nearest(self, x, rt, expected):
        assert align2rastertime_nearest(x, rt) == pytest.approx(expected)


def test_deg2rad_quarter_turn():
    assert deg2rad(180) == pytest.approx(np.pi)
    assert deg2rad(90) == pytest.approx(np.pi / 2)


# ──────────────────────────────────────────────────────────────────────────
# Diffusion gradient calculations
# ──────────────────────────────────────────────────────────────────────────

class TestDiffusionMath:
    @pytest.mark.parametrize(
        "b, delta, DELTA",
        [
            (500, 0.012, 0.025),
            (1000, 0.018, 0.035),
            (2000, 0.020, 0.040),
        ],
    )
    def test_amplitude_inverse_of_bvalue(self, b, delta, DELTA):
        """G derived from b should reproduce b through bFactCalc."""
        G = calc_diffusion_gradient_amplitude(b, delta, DELTA)
        # bFactCalc returns b in s/m^2 — convert back to s/mm^2
        b_calc = bFactCalc(G, delta, DELTA) * 1e-6
        assert b_calc == pytest.approx(b, rel=1e-6)

    def test_b_value_zero_gives_zero_amplitude(self):
        assert calc_diffusion_gradient_amplitude(0, 0.018, 0.035) == pytest.approx(0.0)

    def test_calc_bval_matches_bFactCalc_with_zero_ramp(self):
        """When ramp_time=0, calc_bval (rectangular Stejskal–Tanner) == bFactCalc."""
        G = 5e3  # Hz/m
        delta = 0.018
        DELTA = 0.035
        # calc_bval and bFactCalc both use Hz/m gradients; calc_bval adds ramp correction
        b_rect = bFactCalc(G, delta, DELTA)
        b_with_zero_ramp = calc_bval(G, delta, DELTA, gdiff_rt=0.0)
        assert b_with_zero_ramp == pytest.approx(b_rect, rel=1e-6)

    def test_calc_bval_ramp_correction_lowers_b(self):
        """Adding ramp time reduces the rectangular b-value (Reese et al.)."""
        G = 5e3
        delta = 0.018
        DELTA = 0.035
        b_no_ramp = calc_bval(G, delta, DELTA, gdiff_rt=0.0)
        b_with_ramp = calc_bval(G, delta, DELTA, gdiff_rt=2e-4)
        assert b_with_ramp < b_no_ramp


class TestAreaPreservingTrapezoid:
    """The function operates on Hz/m units (gradient = γ·G_T_per_m), not mT/m."""

    @staticmethod
    def _safe_limits():
        opts = system_limit(SystemLimitType.SAFE)
        return opts.max_grad, opts.max_slew, opts.grad_raster_time

    def test_feasible_case(self):
        """Trapezoid duration ≤ small_delta, amplitude clamped to max_grad."""
        max_grad, max_slew, grad_raster = self._safe_limits()
        small_delta = 0.005  # 5 ms — comfortably > 2·rise (~0.5 ms)

        result = calc_area_preserving_trapezoid(
            G_req=max_grad * 2,  # impossible — forces clamp
            small_delta=small_delta,
            max_grad=max_grad,
            max_slew=max_slew,
            grad_raster_time=grad_raster,
        )
        assert result is not None
        assert result["amplitude"] == max_grad
        assert result["total_duration"] <= small_delta + 1e-12
        assert result["area"] > 0
        assert result["flat_time"] >= 0
        assert result["rise_time"] > 0

    def test_infeasible_returns_none(self):
        """When 2·rise alone exceeds small_delta, returns None."""
        max_grad, max_slew, grad_raster = self._safe_limits()
        # rise = max_grad/max_slew ≈ 240 µs; 2·rise ≈ 480 µs, so 100 µs is infeasible
        small_delta = 100e-6

        result = calc_area_preserving_trapezoid(
            G_req=max_grad * 2,
            small_delta=small_delta,
            max_grad=max_grad,
            max_slew=max_slew,
            grad_raster_time=grad_raster,
        )
        assert result is None


# ──────────────────────────────────────────────────────────────────────────
# Diffusion direction tables
# ──────────────────────────────────────────────────────────────────────────

class TestDiffusionDirections:
    @pytest.mark.parametrize("n", [1, 3, 6, 12, 60])
    def test_shape(self, n):
        g = get_diffusion_directions(n, insert_b0s_at=0)
        assert g.shape == (n, 3)

    def test_n_directions_3_is_orthogonal_basis(self):
        g = get_diffusion_directions(3, insert_b0s_at=0)
        np.testing.assert_array_equal(g, np.eye(3))

    @pytest.mark.parametrize("n", [6, 12, 60])
    def test_unit_norm_for_nonzero_rows(self, n):
        g = get_diffusion_directions(n, insert_b0s_at=0)
        norms = np.linalg.norm(g, axis=1)
        nonzero_norms = norms[norms > 0.1]
        np.testing.assert_allclose(nonzero_norms, 1.0, atol=1e-3)

    def test_b0_insertion_inserts_zero_rows(self):
        """With insert_b0s_at=3, every 4th row is [0,0,0]."""
        g = get_diffusion_directions(6, insert_b0s_at=3)
        # Insertion happens at positions 0 and 3 → 2 b0 rows added → shape (8, 3)
        assert g.shape[0] == 8
        # Every group of (insert_b0s_at + 1) starts with a zero row
        zero_rows_indices = [i for i, row in enumerate(g) if np.allclose(row, 0)]
        assert len(zero_rows_indices) >= 2

    def test_unsupported_n_falls_back_to_3(self):
        g = get_diffusion_directions(7, insert_b0s_at=0)
        np.testing.assert_array_equal(g, np.eye(3))


# ──────────────────────────────────────────────────────────────────────────
# System limits
# ──────────────────────────────────────────────────────────────────────────

class TestSystemLimits:
    @pytest.mark.parametrize(
        "system_type, expected_max_grad_mt, expected_max_slew_t",
        [
            (SystemLimitType.EXTRASAFE, 32, 130),
            (SystemLimitType.SAFE, 34, 140),
            (SystemLimitType.RISKY, 36, 160),
            (SystemLimitType.EXTREME, 38, 180),
        ],
    )
    def test_returns_pp_opts_with_documented_limits(
        self, system_type, expected_max_grad_mt, expected_max_slew_t
    ):
        opts = system_limit(system_type)
        assert isinstance(opts, pp.Opts)
        # max_grad is stored in Hz/m; gamma converts back to T/m, *1e3 for mT/m
        assert opts.max_grad / opts.gamma * 1e3 == pytest.approx(expected_max_grad_mt)
        assert opts.max_slew / opts.gamma == pytest.approx(expected_max_slew_t)

    def test_unknown_type_raises(self):
        with pytest.raises(NotImplementedError):
            system_limit("bogus")
