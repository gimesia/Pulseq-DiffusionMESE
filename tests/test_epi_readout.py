"""Tests for the standalone EPIReadout class.

Covers k-space trajectory construction, gradient/blip/prephaser design, slew-rate
verification, and the duration-trimming helper. Parametrized over all 4 system limits.
"""

from __future__ import annotations

import numpy as np
import pypulseq as pp
import pytest

from EPIReadout import EPIReadout
from utils import SystemLimitType, system_limit


def _make_readout(system_type, **overrides):
    """Build an EPIReadout with sensible defaults plus per-test overrides."""
    base = dict(
        fov=0.224,
        Nx=64,
        Ny=64,
        dwell_time=5e-6,
        system=system_limit(system_type),
        partial_fourier_factor=1.0,
        blip_down=True,
        acceleration_factor=1,
        ramp_sampling="ramp_sampled",
        rephasers=False,
        simultan_rephasers=True,
        verbose=False,
        adc_dead_time_correction=True,
    )
    base.update(overrides)
    return EPIReadout(**base)


# ──────────────────────────────────────────────────────────────────────────
# Construction across all 4 system limits and matrix sizes
# ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("Nx, Ny", [(64, 64), (96, 96), (128, 96)])
def test_construction_succeeds(system_type, Nx, Ny):
    """Smoke: every system × dimension combination builds without error."""
    epi = _make_readout(system_type, Nx=Nx, Ny=Ny)

    assert epi.gx is not None
    assert epi.gx_ is not None
    assert epi.gy is not None
    assert epi.gy_composite is not None
    assert epi.adc is not None
    assert epi.gx_prephaser is not None
    assert epi.gy_prephaser is not None
    assert len(epi.ky_indices) > 0
    assert epi.duration > 0
    assert epi.time_until_echo > 0
    assert epi.time_after_echo > 0
    # gx and gx_ are mirror images
    assert epi.gx_.amplitude == pytest.approx(-epi.gx.amplitude)


# ──────────────────────────────────────────────────────────────────────────
# k-space trajectory
# ──────────────────────────────────────────────────────────────────────────

def test_kspace_full_sampling(system_type):
    """pff=1.0, R=1: ky covers the full [-Ny/2, Ny/2-1] range."""
    Ny = 64
    epi = _make_readout(system_type, Ny=Ny, partial_fourier_factor=1.0)

    assert epi.Ny_eff == Ny
    assert epi.echo_line_index == Ny // 2
    np.testing.assert_array_equal(epi.ky_indices, np.arange(-Ny // 2, Ny // 2))


@pytest.mark.parametrize("pff", [0.6, 0.75, 1.0])
def test_partial_fourier_reduces_lines(system_type, pff):
    """Acquired lines == round(pff·Ny) for blip_down=True with R=1."""
    Ny = 64
    epi = _make_readout(system_type, Ny=Ny, partial_fourier_factor=pff)
    assert epi.Ny_eff == int(round(pff * Ny))
    # post-echo half is always Ny/2
    assert epi.Ny_post == Ny // 2


def test_acceleration_factor_halves_lines_and_doubles_blip(system_type):
    """R=2: half the lines, doubled blip area."""
    base = _make_readout(system_type, acceleration_factor=1)
    accel = _make_readout(system_type, acceleration_factor=2)

    assert accel.Ny_eff == base.Ny_eff // 2
    assert abs(accel.blip_area) == pytest.approx(2 * abs(base.blip_area), rel=1e-6)


def test_blip_polarity_flips_gy_sign(system_type):
    """blip_down=True vs False inverts gy.area sign and the ky traversal direction."""
    down = _make_readout(system_type, blip_down=True)
    up = _make_readout(system_type, blip_down=False)

    # Phase-encoding step has opposite sign
    assert np.sign(down.gy.area) == -np.sign(up.gy.area)
    # blip-down starts at the most negative ky; blip-up starts at the most positive
    assert down.ky_indices[0] < 0 < up.ky_indices[0]


# ──────────────────────────────────────────────────────────────────────────
# Hardware compliance
# ──────────────────────────────────────────────────────────────────────────

def test_verify_slew_rates_passes(system_type):
    """Reconstructed gx waveform must respect the system slew limit."""
    epi = _make_readout(system_type, verbose=True)  # verbose=True enables verify
    # If __init__ didn't raise, slew rates verified internally; call again to be explicit
    epi.verify_slew_rates()


# ──────────────────────────────────────────────────────────────────────────
# Duration trimming
# ──────────────────────────────────────────────────────────────────────────

def test_fit_to_duration_trims_lines(system_type):
    """Setting max_duration well below the natural duration drops acquired lines."""
    full = _make_readout(system_type)
    target = full.duration * 0.5

    trimmed = _make_readout(system_type, max_duration=target)
    assert trimmed.Ny_eff < full.Ny_eff
    assert trimmed.duration <= target + 1e-9


def test_fit_to_duration_noop_when_within_budget(system_type):
    full = _make_readout(system_type)
    epi = _make_readout(system_type, max_duration=full.duration * 2)
    assert epi.Ny_eff == full.Ny_eff


# ──────────────────────────────────────────────────────────────────────────
# Rephasers
# ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("simultan", [True, False])
def test_rephasers_extend_duration(system_type, simultan):
    """rephasers=True increases the total readout duration vs no rephasers."""
    no_reph = _make_readout(system_type, rephasers=False)
    with_reph = _make_readout(
        system_type, rephasers=True, simultan_rephasers=simultan
    )

    assert with_reph.duration > no_reph.duration
    assert with_reph.rephaser_duration > 0
    assert with_reph.gx_rephaser is not None
    assert with_reph.gy_rephaser is not None


# ──────────────────────────────────────────────────────────────────────────
# Input validation
# ──────────────────────────────────────────────────────────────────────────

def test_invalid_partial_fourier_raises():
    """pff outside [0.5, 1.0] is rejected by the assert in __init__."""
    with pytest.raises(AssertionError):
        EPIReadout(
            fov=0.224,
            Nx=64,
            Ny=64,
            dwell_time=5e-6,
            system=system_limit(SystemLimitType.SAFE),
            partial_fourier_factor=0.4,
        )


def test_invalid_acceleration_factor_raises():
    with pytest.raises(AssertionError):
        EPIReadout(
            fov=0.224,
            Nx=64,
            Ny=64,
            dwell_time=5e-6,
            system=system_limit(SystemLimitType.SAFE),
            acceleration_factor=0,
        )


def test_invalid_ramp_sampling_raises():
    with pytest.raises((AssertionError, ValueError)):
        EPIReadout(
            fov=0.224,
            Nx=64,
            Ny=64,
            dwell_time=5e-6,
            system=system_limit(SystemLimitType.SAFE),
            ramp_sampling="not_a_mode",
        )


# ──────────────────────────────────────────────────────────────────────────
# Sequence-attachment helpers
# ──────────────────────────────────────────────────────────────────────────

def test_add_to_sequence_unlabeled_emits_one_block_per_line(system_type):
    """Unlabeled mode adds Ny_eff blocks (one per ky line)."""
    epi = _make_readout(system_type)
    seq = pp.Sequence(epi.system)

    blocks_before = len(seq.block_events)
    epi.add_to_sequence_unlabeled(seq)
    blocks_after = len(seq.block_events)

    assert blocks_after - blocks_before == epi.Ny_eff


def test_add_to_sequence_labeled_emits_at_least_lines(system_type):
    """Labeled mode adds at least Ny_eff blocks (labels live inside the block)."""
    epi = _make_readout(system_type)
    seq = pp.Sequence(epi.system)

    blocks_before = len(seq.block_events)
    epi.add_to_sequence(seq)
    blocks_after = len(seq.block_events)

    assert blocks_after - blocks_before >= epi.Ny_eff
