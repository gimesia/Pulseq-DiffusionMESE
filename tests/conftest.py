"""Shared pytest fixtures and import-path shim.

The package modules use bare imports (``from utils import *``) so they must
be on ``sys.path`` directly — adding the package directory here lets the
tests do ``from PulseqSeq import PulseqSeq`` etc. without modifying the
package itself.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest

# UTF-8 stdio so the validator's ⚠ character doesn't crash the Windows console
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PKG_DIR = Path(__file__).resolve().parent.parent / "pulseq_diffusion_mese"
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

# Silence the chatty class loggers — pytest captures the rest if we need them
logging.getLogger().setLevel(logging.CRITICAL)
for name in (
    "PulseqSeq",
    "EPIDiffusionSEPulseqSeqV4",
    "EPIDiffusionTripleSEPulseqSeq",
    "EPIReadout",
):
    logging.getLogger(name).setLevel(logging.CRITICAL)


from utils import SystemLimitType  # noqa: E402  (needs sys.path patch above)


ALL_SYSTEM_TYPES = [
    SystemLimitType.EXTRASAFE,
    SystemLimitType.SAFE,
    SystemLimitType.RISKY,
    SystemLimitType.EXTREME,
]


@pytest.fixture(params=ALL_SYSTEM_TYPES, ids=[s.value for s in ALL_SYSTEM_TYPES])
def system_type(request):
    """Parametrize across every supported hardware preset."""
    return request.param


def _se_kwargs(system_type, **overrides):
    """Minimal valid kwargs for EPIDiffusionSEPulseqSeq — TR/TE wide enough to fit."""
    base = dict(
        name="TestSE",
        fov=0.224,
        Nx=64,
        Ny=64,
        slice_thickness=3e-3,
        TR=3000,
        TE=80,
        b_value=500,
        b_directions=3,
        b_0_frequency=0,
        system_type=system_type,
        rf90_duration=3e-3,
        rf180_duration=3e-3,
        dwell_time=5e-6,
        partial_fourier_factor=0.75,
        ramp_sampling="ramp_sampled",
        fit_epi=True,
        labeled=False,
    )
    base.update(overrides)
    return base


def _triple_kwargs(system_type, **overrides):
    """Minimal valid kwargs for EPIDiffusionTripleSEPulseqSeq.

    Triple-SE's `_init_spoilers` reads `self.resolution` as mm, while
    `PulseqSeq._init_imaging_params` stores it in metres when derived from
    fov/min(Nx,Ny). To avoid that mm/m inconsistency, pass `resolution`
    explicitly in mm and derive `slice_thickness` from it.
    """
    res_mm = 3.5  # 224 mm / 64
    base = dict(
        name="TestTriple",
        fov=0.224,
        Nx=64,
        Ny=64,
        resolution=res_mm,
        slice_thickness=res_mm * 1e-3,
        TR=5000,
        TE=80,
        b_value=500,
        b_directions=3,
        b_0_frequency=0,
        system_type=system_type,
        rf90_duration=3e-3,
        rf180_duration=3e-3,
        dwell_time=5e-6,
        partial_fourier_factor=0.75,
        ramp_sampling="ramp_sampled",
        fit_epi=True,
        labeled=False,
        rephasers=True,
    )
    base.update(overrides)
    return base


@pytest.fixture
def se_kwargs_factory():
    """Factory that returns SE kwargs for a given system_type with optional overrides."""
    return _se_kwargs


@pytest.fixture
def triple_kwargs_factory():
    """Factory that returns Triple-SE kwargs for a given system_type with optional overrides."""
    return _triple_kwargs


def hard_failures(failed_tests):
    """Strip ⚠ warning lines (kept in failed_tests but don't flip all_passed)."""
    return [m for m in failed_tests if not m.lstrip().startswith("⚠")]
