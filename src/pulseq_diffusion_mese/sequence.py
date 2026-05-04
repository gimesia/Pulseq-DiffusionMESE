"""Core builder for the diffusion-weighted MESE Pulseq sequence.

The sequence follows the standard MESE (Multi-Echo Spin Echo) design:
  1. 90° slice-selective excitation pulse
  2. N refocusing 180° pulses separated by echo spacing (ESP)
  3. Diffusion-sensitising gradients (Stejskal–Tanner pair) around each
     refocusing pulse

Usage::

    seq = build_sequence(
        n_echoes=8,
        echo_spacing=10e-3,   # s
        b_value=1000,         # s/mm²
        fov=0.22,             # m
        n_slices=1,
    )
    seq.write("diffusion_mese.seq")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import pypulseq as pp
    _PYPULSEQ_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYPULSEQ_AVAILABLE = False


# ---------------------------------------------------------------------------
# Parameter dataclass
# ---------------------------------------------------------------------------

@dataclass
class SequenceParams:
    """Parameters controlling the diffusion-MESE sequence."""

    # Sequence timing
    n_echoes: int = 8
    echo_spacing: float = 10e-3          # s
    tr: float = 3.0                      # s

    # Diffusion
    b_value: float = 1000.0             # s/mm²
    diffusion_directions: list = field(
        default_factory=lambda: [[1, 0, 0]]
    )

    # Imaging
    fov: float = 0.22                    # m
    n_slices: int = 1
    slice_thickness: float = 3e-3        # m
    matrix_size: int = 64

    # System limits (Siemens Prisma-like defaults)
    max_grad: float = 80e-3             # T/m
    max_slew: float = 200.0             # T/m/s
    rf_ringdown_time: float = 20e-6     # s
    rf_dead_time: float = 100e-6        # s
    adc_dead_time: float = 10e-6        # s


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_sequence(
    n_echoes: int = 8,
    echo_spacing: float = 10e-3,
    b_value: float = 1000.0,
    fov: float = 0.22,
    n_slices: int = 1,
    slice_thickness: float = 3e-3,
    matrix_size: int = 64,
    tr: float = 3.0,
    diffusion_directions: Optional[list] = None,
    max_grad: float = 80e-3,
    max_slew: float = 200.0,
):
    """Build and return a :class:`pypulseq.Sequence` object.

    Parameters
    ----------
    n_echoes:
        Number of spin echoes per excitation.
    echo_spacing:
        Time between consecutive echo centres (s).
    b_value:
        Diffusion weighting (s/mm²). Set to 0 for a b=0 reference scan.
    fov:
        Field of view (m).
    n_slices:
        Number of slices to acquire.
    slice_thickness:
        Thickness of each slice (m).
    matrix_size:
        Number of readout/phase-encode samples.
    tr:
        Repetition time (s).
    diffusion_directions:
        List of unit-vector gradient directions, e.g. ``[[1,0,0]]``.
    max_grad:
        Maximum gradient amplitude (T/m).
    max_slew:
        Maximum gradient slew rate (T/m/s).

    Returns
    -------
    seq : pypulseq.Sequence | SequenceParams
        The assembled sequence (requires *pypulseq*).  When *pypulseq* is
        not installed a :class:`SequenceParams` placeholder is returned so
        the module can still be imported and tested structurally.
    """
    if diffusion_directions is None:
        diffusion_directions = [[1, 0, 0]]

    params = SequenceParams(
        n_echoes=n_echoes,
        echo_spacing=echo_spacing,
        b_value=b_value,
        fov=fov,
        n_slices=n_slices,
        slice_thickness=slice_thickness,
        matrix_size=matrix_size,
        tr=tr,
        diffusion_directions=diffusion_directions,
        max_grad=max_grad,
        max_slew=max_slew,
    )

    if not _PYPULSEQ_AVAILABLE:
        # Return params so downstream code can still inspect them.
        return params

    return _assemble(params)


# ---------------------------------------------------------------------------
# Internal assembly
# ---------------------------------------------------------------------------

_GAMMA_HZ = 42.577e6  # Hz/T (proton gyromagnetic ratio)


def _trap_duration(g: "pp.SimpleNamespace") -> float:
    """Return total duration of a trapezoid gradient (pypulseq ≥1.4 has no .duration)."""
    return g.rise_time + g.flat_time + g.fall_time


def _system(params: SequenceParams) -> "pp.Opts":
    return pp.Opts(
        max_grad=params.max_grad * 1e3,   # mT/m
        grad_unit="mT/m",
        max_slew=params.max_slew,
        slew_unit="T/m/s",
        rf_ringdown_time=params.rf_ringdown_time,
        rf_dead_time=params.rf_dead_time,
        adc_dead_time=params.adc_dead_time,
    )


def _assemble(params: SequenceParams) -> "pp.Sequence":
    """Low-level sequence assembly using pypulseq primitives."""
    system = _system(params)
    seq = pp.Sequence(system=system)

    half_esp = params.echo_spacing / 2.0

    # --- RF pulses ----------------------------------------------------------
    flip90 = pp.make_sinc_pulse(
        flip_angle=np.pi / 2,
        duration=3e-3,
        slice_thickness=params.slice_thickness,
        apodization=0.42,
        time_bw_product=4,
        system=system,
        return_gz=False,
    )
    flip180 = pp.make_sinc_pulse(
        flip_angle=np.pi,
        duration=3e-3,
        slice_thickness=params.slice_thickness,
        apodization=0.42,
        time_bw_product=4,
        use="refocusing",
        system=system,
        return_gz=False,
    )

    # --- Readout gradient ---------------------------------------------------
    gx = pp.make_trapezoid(
        channel="x",
        flat_area=params.matrix_size / params.fov,
        flat_time=6.4e-3,
        system=system,
    )
    gx_duration = _trap_duration(gx)
    adc = pp.make_adc(
        num_samples=params.matrix_size,
        duration=gx.flat_time,
        delay=gx.rise_time,
        system=system,
    )
    gx_pre = pp.make_trapezoid(
        channel="x",
        area=-gx.area / 2,
        duration=half_esp / 2,
        system=system,
    )
    gx_pre_duration = _trap_duration(gx_pre)

    # --- Diffusion gradient (simplified Stejskal–Tanner) -------------------
    # Amplitude in Hz/m (pypulseq internal unit); clamp to system max_grad.
    gd_amp_Tm = _diffusion_gradient_amplitude(params, gx)
    gd_amp_Hz = min(gd_amp_Tm * _GAMMA_HZ, system.max_grad)
    gd_dur = half_esp / 2

    gd = []
    for ch, d in zip(["x", "y", "z"], params.diffusion_directions[0]):
        if abs(d) > 1e-9 and abs(gd_amp_Hz * d) > 1e-3:
            gd.append(
                pp.make_trapezoid(
                    channel=ch,
                    amplitude=gd_amp_Hz * d,
                    duration=gd_dur,
                    system=system,
                )
            )

    # --- Build sequence blocks ----------------------------------------------
    seq.add_block(flip90)
    seq.add_block(pp.make_delay(half_esp))

    for _ in range(params.n_echoes):
        seq.add_block(flip180)
        if gd:
            seq.add_block(*gd)
        seq.add_block(gx_pre)
        seq.add_block(gx, adc)
        if gd:
            seq.add_block(*gd)
        remaining = half_esp - gx_duration - gx_pre_duration
        if remaining > 0:
            seq.add_block(pp.make_delay(remaining))

    tr_fill = params.tr - params.n_echoes * params.echo_spacing
    if tr_fill > 0:
        seq.add_block(pp.make_delay(tr_fill))

    return seq


def _diffusion_gradient_amplitude(
    params: SequenceParams, gx: "pp.SimpleNamespace"
) -> float:
    """Estimate the diffusion gradient amplitude for the target b-value.

    Uses the Stejskal–Tanner formula:
        b = γ² G² δ² (Δ − δ/3)
    where δ and Δ are set to ``echo_spacing/2``.
    """
    gamma = 2 * np.pi * 42.577e6  # rad/T/s
    delta_small = params.echo_spacing / 2
    Delta = params.echo_spacing / 2
    b_si = params.b_value * 1e6  # convert s/mm² → s/m²
    if b_si == 0:
        return 0.0
    denom = gamma**2 * delta_small**2 * (Delta - delta_small / 3)
    return float(np.sqrt(b_si / denom))
