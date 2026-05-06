"""End-to-end tests for EPIDiffusionSEPulseqSeq.

For each system limit and several feature toggles we build the sequence and
exercise the class's own validation methods (`validate_sequence_properties`,
`seq.check_timing`) plus a recomputed b-value sanity check.
"""

from __future__ import annotations

import numpy as np
import pytest

from EPIDiffusionSEPulseqSeq import EPIDiffusionSEPulseqSeq
from utils import SystemLimitType, calc_bval


from tests.conftest import hard_failures  # type: ignore[import-not-found]


# ──────────────────────────────────────────────────────────────────────────
# Construction across all 4 system limits
# ──────────────────────────────────────────────────────────────────────────

def test_constructs_for_each_system_limit(system_type, se_kwargs_factory):
    seq = EPIDiffusionSEPulseqSeq(**se_kwargs_factory(system_type))
    assert seq.seq is not None
    assert seq.delayTR >= 0
    assert seq.epi.Ny_eff > 0
    assert seq.diffusion_gradient_amplitude > 0


# ──────────────────────────────────────────────────────────────────────────
# Class validators
# ──────────────────────────────────────────────────────────────────────────

def test_validate_sequence_properties(system_type, se_kwargs_factory):
    seq = EPIDiffusionSEPulseqSeq(**se_kwargs_factory(system_type))
    passed, fails = seq.validate_sequence_properties()
    assert passed, f"Validation failed: {fails}"
    # ⚠ warnings are acceptable; any non-warning failure is not
    assert hard_failures(fails) == []


def test_check_timing(system_type, se_kwargs_factory):
    seq = EPIDiffusionSEPulseqSeq(**se_kwargs_factory(system_type))
    ok, report = seq.seq.check_timing()
    assert ok, f"check_timing failed: {report}"


def test_b_value_within_tolerance(system_type, se_kwargs_factory):
    """Recompute b from stored G/δ/Δ/ramp; must be within ±max(10, 3% × b)."""
    kwargs = se_kwargs_factory(system_type)
    seq = EPIDiffusionSEPulseqSeq(**kwargs)

    b_calc = calc_bval(
        seq.diffusion_gradient_amplitude / 1000,
        seq.small_delta,
        seq.big_DELTA,
        seq.diffusion_gradient_rise_time,
    )
    tol = max(10.0, 0.03 * seq.b_value)
    assert abs(b_calc - seq.b_value) <= tol


# ──────────────────────────────────────────────────────────────────────────
# Dimension sweep
# ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "Nx, Ny, slice_thickness, N_slices",
    [
        (64, 64, 3e-3, 1),
        (96, 96, 3e-3, 1),
        (128, 96, 2e-3, 1),
        (64, 64, 4e-3, 3),
    ],
)
def test_dimensions(Nx, Ny, slice_thickness, N_slices, se_kwargs_factory):
    """A single-system slice across multiple matrix and slice configurations."""
    kwargs = se_kwargs_factory(
        SystemLimitType.SAFE,
        Nx=Nx,
        Ny=Ny,
        slice_thickness=slice_thickness,
        N_slices=N_slices,
    )
    seq = EPIDiffusionSEPulseqSeq(**kwargs)

    ok, _ = seq.seq.check_timing()
    assert ok
    passed, fails = seq.validate_sequence_properties()
    assert passed, fails
    assert hard_failures(fails) == []


# ──────────────────────────────────────────────────────────────────────────
# Feature toggles
# ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("pff", [0.75, 1.0])
def test_partial_fourier_factor(se_kwargs_factory, pff):
    kwargs = se_kwargs_factory(
        SystemLimitType.SAFE,
        partial_fourier_factor=pff,
        TE=120,  # extra headroom so fit_epi=False works at pff=1.0
        fit_epi=False,
    )
    seq = EPIDiffusionSEPulseqSeq(**kwargs)
    passed, _ = seq.validate_sequence_properties()
    assert passed


def test_calibration_readout_adds_blocks(se_kwargs_factory):
    base = EPIDiffusionSEPulseqSeq(
        **se_kwargs_factory(SystemLimitType.SAFE, calibration_readout=False)
    )
    nav = EPIDiffusionSEPulseqSeq(
        **se_kwargs_factory(SystemLimitType.SAFE, calibration_readout=True)
    )
    assert len(nav.seq.block_events) > len(base.seq.block_events)


@pytest.mark.parametrize("blip_down", [True, False])
def test_blip_polarity_both_directions(se_kwargs_factory, blip_down):
    seq = EPIDiffusionSEPulseqSeq(
        **se_kwargs_factory(SystemLimitType.SAFE, blip_down=blip_down)
    )
    passed, _ = seq.validate_sequence_properties()
    assert passed


@pytest.mark.parametrize("rf180_spoiler", [True, False])
def test_rf180_spoiler_toggle(se_kwargs_factory, rf180_spoiler):
    """rf180_spoiler=True consumes additional time inside TE/TR."""
    common = se_kwargs_factory(SystemLimitType.SAFE, rf180_spoiler=rf180_spoiler)
    seq = EPIDiffusionSEPulseqSeq(**common)
    ok, _ = seq.seq.check_timing()
    assert ok


@pytest.mark.parametrize("end_spoilers", [True, False])
def test_end_spoilers_toggle(se_kwargs_factory, end_spoilers):
    seq = EPIDiffusionSEPulseqSeq(
        **se_kwargs_factory(SystemLimitType.SAFE, end_spoilers=end_spoilers)
    )
    passed, _ = seq.validate_sequence_properties()
    assert passed


def test_b0_only_sequence(se_kwargs_factory):
    """b=0: no diffusion weighting, must still build cleanly."""
    seq = EPIDiffusionSEPulseqSeq(
        **se_kwargs_factory(SystemLimitType.SAFE, b_value=0, b_directions=1)
    )
    passed, _ = seq.validate_sequence_properties()
    assert passed
    ok, _ = seq.seq.check_timing()
    assert ok


@pytest.mark.parametrize("b_directions", [1, 3, 6])
def test_multiple_b_directions(se_kwargs_factory, b_directions):
    """Direction count flows through to the build loop."""
    seq = EPIDiffusionSEPulseqSeq(
        **se_kwargs_factory(SystemLimitType.SAFE, b_directions=b_directions)
    )
    assert seq.b_directions.shape[0] == b_directions
    passed, _ = seq.validate_sequence_properties()
    assert passed


def test_amplitude_clamp_path(se_kwargs_factory):
    """User-fixed small_delta + impossibly large b forces the amplitude clamp."""
    kwargs = se_kwargs_factory(
        SystemLimitType.SAFE,
        b_value=20000,           # very high
        small_delta=0.005,       # short — forces G > max_grad
        big_DELTA=0.025,
        TE=120,                  # widen so timing still fits
        fit_epi=False,
    )
    seq = EPIDiffusionSEPulseqSeq(**kwargs)
    assert seq._amplitude_clamped is True
    ok, _ = seq.seq.check_timing()
    assert ok


def test_fit_epi_disabled_raises(se_kwargs_factory):
    """TE too short with fit_epi=False must raise rather than silently retry."""
    kwargs = se_kwargs_factory(
        SystemLimitType.SAFE,
        TE=10,           # impossibly short
        fit_epi=False,
    )
    with pytest.raises(ValueError):
        EPIDiffusionSEPulseqSeq(**kwargs)
