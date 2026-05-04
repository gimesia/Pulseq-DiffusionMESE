"""Bloch-equation simulation of the diffusion-MESE sequence.

This script simulates the magnetisation evolution through the MESE echo train
and plots the resulting echo amplitudes as a function of echo number.

Usage::

    python simulation/simulate.py

Requirements: numpy, matplotlib (and optionally pypulseq for the full
sequence object).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Bloch simulation utilities
# ---------------------------------------------------------------------------

def _rot_x(angle: float) -> np.ndarray:
    """Rotation matrix around the x-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(angle: float) -> np.ndarray:
    """Rotation matrix around the y-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _relax(m: np.ndarray, dt: float, t1: float, t2: float) -> np.ndarray:
    """Apply T1/T2 relaxation over time *dt*."""
    e1 = np.exp(-dt / t1)
    e2 = np.exp(-dt / t2)
    return np.array([m[0] * e2, m[1] * e2, m[2] * e1 + (1 - e1)])


def simulate_mese(
    n_echoes: int = 8,
    echo_spacing: float = 10e-3,
    t1: float = 1.0,
    t2: float = 0.08,
    b_value: float = 1000.0,
    diffusivity: float = 0.8e-9,   # m²/s (grey matter ADC)
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate echo amplitudes for a MESE sequence using the Bloch equations.

    Parameters
    ----------
    n_echoes:
        Number of spin echoes.
    echo_spacing:
        Time between echo centres (s).
    t1:
        Longitudinal relaxation time (s).
    t2:
        Transverse relaxation time (s).
    b_value:
        Diffusion weighting (s/mm²).
    diffusivity:
        Apparent diffusion coefficient (m²/s).

    Returns
    -------
    echo_times : np.ndarray
        Time of each echo centre (s).
    echo_amplitudes : np.ndarray
        Simulated echo amplitude at each echo.
    """
    m = np.array([0.0, 0.0, 1.0])  # equilibrium magnetisation

    # 90° excitation
    m = _rot_x(np.pi / 2) @ m
    m = _relax(m, echo_spacing / 2, t1, t2)

    echo_times = np.zeros(n_echoes)
    echo_amplitudes = np.zeros(n_echoes)

    for n in range(n_echoes):
        # 180° refocusing
        m = _rot_y(np.pi) @ m
        m = _relax(m, echo_spacing / 2, t1, t2)

        # Record echo amplitude (Mxy)
        t_echo = (n + 1) * echo_spacing
        echo_times[n] = t_echo
        # Apply diffusion attenuation: exp(-b * ADC)  [b in s/mm², ADC in mm²/s]
        adc_mm2 = diffusivity * 1e6
        diff_atten = np.exp(-b_value * adc_mm2)
        echo_amplitudes[n] = np.sqrt(m[0] ** 2 + m[1] ** 2) * diff_atten

        if n < n_echoes - 1:
            m = _relax(m, echo_spacing / 2, t1, t2)

    return echo_times, echo_amplitudes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    n_echoes = 8
    echo_spacing = 10e-3  # s
    t1, t2 = 1.0, 0.08   # s  (grey matter)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Diffusion-MESE Bloch Simulation", fontsize=14)

    # --- Panel 1: echo train for different b-values -------------------------
    ax = axes[0]
    for bval in [0, 500, 1000, 2000]:
        times, amps = simulate_mese(
            n_echoes=n_echoes,
            echo_spacing=echo_spacing,
            t1=t1,
            t2=t2,
            b_value=bval,
        )
        ax.plot(times * 1e3, amps, "o-", label=f"b={bval} s/mm²")
    ax.set_xlabel("Echo time (ms)")
    ax.set_ylabel("Signal amplitude (a.u.)")
    ax.set_title("Echo train vs. b-value")
    ax.legend()
    ax.grid(True)

    # --- Panel 2: T2 decay curve for b=1000 ---------------------------------
    ax = axes[1]
    t2_values = [0.04, 0.08, 0.12]
    for t2_val in t2_values:
        times, amps = simulate_mese(
            n_echoes=n_echoes,
            echo_spacing=echo_spacing,
            t1=t1,
            t2=t2_val,
            b_value=1000,
        )
        ax.plot(times * 1e3, amps, "s-", label=f"T2={int(t2_val * 1e3)} ms")
    ax.set_xlabel("Echo time (ms)")
    ax.set_ylabel("Signal amplitude (a.u.)")
    ax.set_title("Echo train vs. T2 (b=1000 s/mm²)")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig("simulation/bloch_simulation.png", dpi=150)
    plt.show()
    print("Simulation complete. Figure saved to simulation/bloch_simulation.png")


if __name__ == "__main__":
    main()
