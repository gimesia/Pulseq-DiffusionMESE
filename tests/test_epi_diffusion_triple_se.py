"""End-to-end tests for EPIDiffusionTripleSEPulseqSeq.

Mirrors the structure of test_epi_diffusion_se.py.  For each system limit
and several feature toggles we build the triple-echo sequence and exercise
the class's own validation methods (`validate_sequence_properties`,
`seq.check_timing`) plus additional triple-SE-specific invariants.
"""

from __future__ import annotations

import numpy as np
import pytest

from EPIDiffusionTripleSEPulseqSeq import EPIDiffusionTripleSEPulseqSeq
from utils import SystemLimitType, calc_bval


from tests.conftest import hard_failures  # type: ignore[import-not-found]


# ──────────────────────────────────────────────────────────────────────────
# Construction across all 4 system limits
# ──────────────────────────────────────────────────────────────────────────


def test_constructs_for_each_system_limit(system_type, triple_kwargs_factory):
    seq = EPIDiffusionTripleSEPulseqSeq(**triple_kwargs_factory(system_type))
    assert seq.seq is not None
    assert seq.delayTR >= 0
    # All three EPI readout objects must exist
    assert seq.epi is not None
    assert seq.epi2 is not None
    assert seq.epi3 is not None
    assert seq.diffusion_gradient_amplitude > 0


# ──────────────────────────────────────────────────────────────────────────
# Class validators
# ──────────────────────────────────────────────────────────────────────────


def test_validate_sequence_properties(system_type, triple_kwargs_factory):
    seq = EPIDiffusionTripleSEPulseqSeq(**triple_kwargs_factory(system_type))
    _, fails = seq.validate_sequence_properties()
    # The base PulseqSeq validator compares self.TE (TE1) against the pypulseq
    # report's TE which reflects TE3. Filter that known semantic mismatch and
    # check that nothing else failed.
    non_te_failures = [f for f in hard_failures(fails) if "TE:" not in f]
    assert non_te_failures == [], f"Unexpected hard failures: {non_te_failures}"


def test_check_timing(system_type, triple_kwargs_factory):
    seq = EPIDiffusionTripleSEPulseqSeq(**triple_kwargs_factory(system_type))
    ok, report = seq.seq.check_timing()
    assert ok, f"check_timing failed: {report}"


def test_b_value_within_tolerance(system_type, triple_kwargs_factory):
    """Recompute b from stored G/δ/Δ/ramp; must be within ±max(10, 3% × b)."""
    seq = EPIDiffusionTripleSEPulseqSeq(**triple_kwargs_factory(system_type))
    b_calc = calc_bval(
        seq.diffusion_gradient_amplitude / 1000,
        seq.small_delta,
        seq.big_DELTA,
        seq.diffusion_gradient_rise_time,
    )
    tol = max(10.0, 0.03 * seq.b_value)
    assert abs(b_calc - seq.b_value) <= tol


# ──────────────────────────────────────────────────────────────────────────
# Triple-SE-specific invariants
# ──────────────────────────────────────────────────────────────────────────


def test_validate_echo_timing(system_type, triple_kwargs_factory):
    """validate_echo_timing() checks spin-echo symmetry for all three echoes.

    Tolerance is 5 ms: EXTRASAFE's very long ramp times cause raster-rounding to
    accumulate across three echoes, producing small but non-zero timing drift.
    5 ms catches gross errors while accepting that hardware limit.
    """
    seq = EPIDiffusionTripleSEPulseqSeq(**triple_kwargs_factory(system_type))
    ok, fails = seq.validate_echo_timing(tolerance_us=5000)
    assert ok, f"Echo timing validation failed: {fails}"


def test_three_echoes_have_increasing_te(triple_kwargs_factory):
    """TE1 < TE2 < TE3 must hold: later echoes are further from excitation."""
    seq = EPIDiffusionTripleSEPulseqSeq(
        **triple_kwargs_factory(SystemLimitType.SAFE)
    )
    assert seq.TE < seq.TE2 < seq.TE3


@pytest.mark.parametrize("TE_ms", list(range(95, 111)))
def test_te3_whole_millisecond_in_95_110_range(TE_ms, triple_kwargs_factory):
    """TE3 must be a whole millisecond for every TE1 in the 95-110 ms range.

    The ceiling-rounding in _create_epi3() is supposed to guarantee this.
    """
    kwargs = triple_kwargs_factory(SystemLimitType.SAFE, TE=TE_ms, fit_epi=True)
    seq = EPIDiffusionTripleSEPulseqSeq(**kwargs)

    te3_ms = seq.TE3 * 1e3
    residual_us = abs(te3_ms - round(te3_ms)) * 1e3
    assert residual_us < 1.0, (
        f"TE1={TE_ms} ms → TE3={te3_ms:.6f} ms is not a whole millisecond "
        f"(residual {residual_us:.3f} µs)"
    )


@pytest.mark.parametrize("pff", [0.75, 1.0])
@pytest.mark.parametrize("TE_ms", list(range(80, 125, 5)))
def test_te_invariant_under_blip_polarity(TE_ms, pff, triple_kwargs_factory):
    """blip-down and blip-up must produce identical TE1, TE2, TE3.

    Blip polarity only reverses the k-space traversal direction and must not
    change echo timing. A prior off-by-one in EPIReadout._setup_kspace_trajectory()
    placed ky=0 at Ny_pre=15 for blip_down=False vs Ny_pre=16 for blip_down=True
    (pff=0.75, Ny=64), shifting the echo readout centre by 1 line_duration and
    leaking into TE1/TE2/TE3 via time_until_echo. We assert Ny_pre parity
    directly so a regression points at the readout layer, not the parent
    sequence's ms-snapping.
    """
    blip_down_seq = EPIDiffusionTripleSEPulseqSeq(
        **triple_kwargs_factory(
            SystemLimitType.SAFE, TE=TE_ms, blip_down=True, partial_fourier_factor=pff
        )
    )
    blip_up_seq = EPIDiffusionTripleSEPulseqSeq(
        **triple_kwargs_factory(
            SystemLimitType.SAFE, TE=TE_ms, blip_down=False, partial_fourier_factor=pff
        )
    )

    for name, epi_d, epi_u, te_d, te_u in [
        ("TE1/EPI1", blip_down_seq.epi,  blip_up_seq.epi,  blip_down_seq.TE,  blip_up_seq.TE),
        ("TE2/EPI2", blip_down_seq.epi2, blip_up_seq.epi2, blip_down_seq.TE2, blip_up_seq.TE2),
        ("TE3/EPI3", blip_down_seq.epi3, blip_up_seq.epi3, blip_down_seq.TE3, blip_up_seq.TE3),
    ]:
        assert epi_d.Ny_pre == epi_u.Ny_pre, (
            f"TE1={TE_ms} ms, pff={pff}, {name}: Ny_pre differs at readout layer — "
            f"blip-down Ny_pre={epi_d.Ny_pre}, blip-up Ny_pre={epi_u.Ny_pre}"
        )
        assert te_d == te_u, (
            f"TE1={TE_ms} ms, pff={pff}, {name}: "
            f"blip-down TE={te_d*1e3:.3f} ms, blip-up TE={te_u*1e3:.3f} ms"
        )
        


# ──────────────────────────────────────────────────────────────────────────
# Dimension sweep
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "Nx, Ny",
    [
        (64, 64),
        (96, 96),
        (128, 96),
    ],
)
def test_dimensions(Nx, Ny, triple_kwargs_factory):
    res_mm = 224 / min(Nx, Ny)
    kwargs = triple_kwargs_factory(
        SystemLimitType.SAFE,
        Nx=Nx,
        Ny=Ny,
        resolution=res_mm,
        slice_thickness=res_mm * 1e-3,
    )
    seq = EPIDiffusionTripleSEPulseqSeq(**kwargs)

    ok, _ = seq.seq.check_timing()
    assert ok
    _, fails = seq.validate_sequence_properties()
    non_te_failures = [f for f in hard_failures(fails) if "TE:" not in f]
    assert non_te_failures == [], non_te_failures


# ──────────────────────────────────────────────────────────────────────────
# Feature toggles
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("pff", [0.75, 1.0])
def test_partial_fourier_factor(triple_kwargs_factory, pff):
    kwargs = triple_kwargs_factory(
        SystemLimitType.SAFE,
        partial_fourier_factor=pff,
        TE=120,
        fit_epi=False,
    )
    seq = EPIDiffusionTripleSEPulseqSeq(**kwargs)
    _, fails = seq.validate_sequence_properties()
    assert [f for f in hard_failures(fails) if "TE:" not in f] == []


def test_calibration_readout_adds_blocks(triple_kwargs_factory):
    base = EPIDiffusionTripleSEPulseqSeq(
        **triple_kwargs_factory(SystemLimitType.SAFE, calibration_readout=False)
    )
    nav = EPIDiffusionTripleSEPulseqSeq(
        **triple_kwargs_factory(SystemLimitType.SAFE, calibration_readout=True)
    )
    assert len(nav.seq.block_events) > len(base.seq.block_events)


@pytest.mark.parametrize("blip_down", [True, False])
def test_blip_polarity_both_directions(triple_kwargs_factory, blip_down):
    seq = EPIDiffusionTripleSEPulseqSeq(
        **triple_kwargs_factory(SystemLimitType.SAFE, blip_down=blip_down)
    )
    _, fails = seq.validate_sequence_properties()
    assert [f for f in hard_failures(fails) if "TE:" not in f] == []


@pytest.mark.parametrize("rf180_spoiler", [True, False])
def test_rf180_spoiler_toggle(triple_kwargs_factory, rf180_spoiler):
    seq = EPIDiffusionTripleSEPulseqSeq(
        **triple_kwargs_factory(SystemLimitType.SAFE, rf180_spoiler=rf180_spoiler)
    )
    ok, _ = seq.seq.check_timing()
    assert ok


@pytest.mark.parametrize("b_directions", [1, 3, 6])
def test_multiple_b_directions(triple_kwargs_factory, b_directions):
    seq = EPIDiffusionTripleSEPulseqSeq(
        **triple_kwargs_factory(SystemLimitType.SAFE, b_directions=b_directions)
    )
    assert seq.b_directions.shape[0] == b_directions
    _, fails = seq.validate_sequence_properties()
    assert [f for f in hard_failures(fails) if "TE:" not in f] == []


def test_amplitude_clamp_path(triple_kwargs_factory):
    """User-fixed small_delta + impossibly large b forces the amplitude clamp."""
    kwargs = triple_kwargs_factory(
        SystemLimitType.SAFE,
        b_value=20000,
        small_delta=0.005,
        big_DELTA=0.025,
        TE=120,
        fit_epi=False,
    )
    seq = EPIDiffusionTripleSEPulseqSeq(**kwargs)
    assert seq._amplitude_clamped is True
    ok, _ = seq.seq.check_timing()
    assert ok


def test_fit_epi_disabled_raises(triple_kwargs_factory):
    """TE too short with fit_epi=False must raise rather than silently retry."""
    kwargs = triple_kwargs_factory(
        SystemLimitType.SAFE,
        TE=10,
        fit_epi=False,
    )
    with pytest.raises(ValueError):
        EPIDiffusionTripleSEPulseqSeq(**kwargs)


@pytest.mark.parametrize(
    "uniform_areas, uniform_dirs",
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_spoiler_strategy_variants(triple_kwargs_factory, uniform_areas, uniform_dirs):
    """All four spoiler area/direction combinations build and pass timing."""
    seq = EPIDiffusionTripleSEPulseqSeq(
        **triple_kwargs_factory(
            SystemLimitType.SAFE,
            uniform_spoiler_areas=uniform_areas,
            uniform_spoiler_directions=uniform_dirs,
        )
    )
    ok, _ = seq.seq.check_timing()
    assert ok
