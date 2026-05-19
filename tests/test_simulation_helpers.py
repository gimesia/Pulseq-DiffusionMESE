"""Tests for simulation/utils_simulation.py.

Exercises add_tumor_to_phantom, epi_ghost_correction, fft_reconstruct_image,
pad_to_cube, and visualize_kspace_trajectory.
No MRzeroCore or real Sequence objects are needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SIM_DIR = Path(__file__).resolve().parent.parent / "simulation"
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

torch = pytest.importorskip("torch")

from utils_simulation import (  # noqa: E402
    add_tumor_to_phantom,
    epi_ghost_correction,
    fft_reconstruct_image,
    pad_to_cube,
    visualize_kspace_trajectory,
)


# --------------------------------------------------------------------------
# add_tumor_to_phantom
# --------------------------------------------------------------------------


class _Phantom:
    """Minimal stand-in for an MR0 VoxelGridPhantom."""
    def __init__(self, shape, fill=1.0e-3):
        self.D = torch.full(shape, fill, dtype=torch.float32)


@pytest.mark.parametrize("loc", ["tl", "tr", "bl", "br"])
def test_tumor_placement_quarters_2d(loc):
    """Each named quadrant must place the tumour inside the phantom."""
    p = _Phantom((40, 40))
    p = add_tumor_to_phantom(
        p, tumor_size=(6, 6, 1), tumor_location=loc,
        adc_tumor_core=1.5, adc_tumor_border=2.5,
    )
    D = p.D.numpy()
    assert np.any(np.isclose(D, 1.5))   # core present
    assert np.any(np.isclose(D, 2.5))   # border present


def test_tumor_explicit_center_3d():
    """3-D phantom with explicit (cx, cy, cz) - interior gets core or border ADC."""
    p = _Phantom((20, 20, 20))
    p = add_tumor_to_phantom(
        p,
        tumor_size=(6, 6, 6),
        tumor_location=(10, 10, 10),
        adc_tumor_core=1.5,
        adc_tumor_border=2.5,
    )
    center = p.D[10, 10, 10].item()
    assert abs(center - 1.5) < 1e-4 or abs(center - 2.5) < 1e-4


def test_tumor_invalid_string_location():
    p = _Phantom((20, 20))
    with pytest.raises(ValueError):
        add_tumor_to_phantom(p, tumor_size=(4, 4, 1), tumor_location="nowhere")


def test_tumor_invalid_dim():
    """1-D D tensor must be rejected."""
    class Bad:
        D = torch.zeros(10)
    with pytest.raises(ValueError):
        add_tumor_to_phantom(Bad(), (2, 2, 2), "tl")


def test_tumor_invalid_location_type():
    p = _Phantom((20, 20))
    with pytest.raises(ValueError):
        add_tumor_to_phantom(p, tumor_size=(4, 4, 1), tumor_location=(1, 2))  # 2-tuple


# --------------------------------------------------------------------------
# epi_ghost_correction
# --------------------------------------------------------------------------


def _uniform_kspace(N=32, n_lines=8):
    """k-space with no even/odd phase mismatch."""
    x = np.linspace(-1, 1, N)
    img = np.exp(-(x**2) * 8).astype(np.complex64)
    k = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(img)))
    kspace = np.empty((n_lines, N), dtype=np.complex64)
    for i in range(n_lines):
        kspace[i] = k[::-1] if i % 2 else k
    return kspace


def _kspace_with_odd_phase_ramp(N=32, n_lines=8, slope=0.15):
    """k-space where odd lines carry a systematic image-domain phase ramp."""
    x = np.linspace(-1, 1, N)
    img = np.exp(-(x**2) * 8).astype(np.complex64)
    ghost_phase = slope * np.arange(N, dtype=float)

    even_img = img
    odd_img = img * np.exp(1j * ghost_phase)

    def img_to_k(row, is_odd):
        k = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(row)))
        return k[::-1] if is_odd else k

    kspace = np.empty((n_lines, N), dtype=np.complex64)
    for i in range(n_lines):
        row = odd_img if i % 2 else even_img
        kspace[i] = img_to_k(row, bool(i % 2))
    return kspace


def test_epi_ghost_correction_identity_when_no_ghost():
    """No ghost -> correction is near-identity (within FFT float32 noise floor)."""
    kspace = _uniform_kspace()
    corrected = epi_ghost_correction(kspace, kspace)
    peak = np.max(np.abs(kspace))
    np.testing.assert_allclose(
        np.abs(corrected), np.abs(kspace), atol=peak * 1e-3,
        err_msg="ghost correction should be near-identity when there is no ghost",
    )


def test_epi_ghost_correction_output_shape_and_dtype():
    """Output must be complex64 numpy with the same shape as the EPI input."""
    nav = _uniform_kspace(N=32, n_lines=4)
    epi = _uniform_kspace(N=32, n_lines=16)
    out = epi_ghost_correction(nav, epi)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.complex64
    assert out.shape == epi.shape


def test_epi_ghost_correction_modifies_data_when_ghost_present():
    """When a phase ramp is present the corrected k-space must differ from input."""
    kspace_ghost = _kspace_with_odd_phase_ramp(slope=0.3)
    corrected = epi_ghost_correction(kspace_ghost, kspace_ghost)
    assert not np.allclose(corrected, kspace_ghost)


def test_epi_ghost_correction_reduces_phase_error():
    """After correction the even/odd phase difference should decrease.

    Ghost model: odd lines carry a linear phase ramp (eddy-current artefact).
    The navigator-based symmetric correction (Bernstein et al. 2004) is designed
    to halve the per-pixel phase error. This test encodes that physical
    expectation.
    """
    N, n_lines, slope = 64, 8, 0.4
    kspace = _kspace_with_odd_phase_ramp(N=N, n_lines=n_lines, slope=slope)
    corrected = epi_ghost_correction(kspace, kspace)

    def img_from_even(stack):
        imgs = []
        for i in range(0, len(stack), 2):
            imgs.append(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(stack[i]))))
        return np.mean(imgs, axis=0)

    def img_from_odd(stack):
        imgs = []
        for i in range(1, len(stack), 2):
            imgs.append(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(stack[i][::-1].copy()))))
        return np.mean(imgs, axis=0)

    diff_before = np.mean(np.abs(np.angle(img_from_even(kspace) * np.conj(img_from_odd(kspace)))))
    diff_after = np.mean(np.abs(np.angle(img_from_even(corrected) * np.conj(img_from_odd(corrected)))))

    assert diff_after < diff_before, (
        f"phase error should decrease: before={diff_before:.3f} rad, after={diff_after:.3f} rad"
    )


def test_epi_ghost_correction_torch_input():
    """Accepts torch tensors and returns numpy with matching shape."""
    kspace = _uniform_kspace(N=32, n_lines=8)
    out = epi_ghost_correction(torch.from_numpy(kspace), torch.from_numpy(kspace))
    assert isinstance(out, np.ndarray)
    assert out.shape == kspace.shape


# --------------------------------------------------------------------------
# fft_reconstruct_image
# --------------------------------------------------------------------------


def test_fft_reconstruct_round_trip_delta():
    """A centred k-space delta reconstructs to a uniform-magnitude image."""
    N = 16
    k = np.zeros((N, N), dtype=np.complex64)
    k[N // 2, N // 2] = 1.0
    mag, img = fft_reconstruct_image(k)
    assert mag.shape == (N, N)
    assert np.allclose(mag, mag[0, 0])
    assert np.iscomplexobj(img)


def test_fft_reconstruct_round_trip_known_image():
    """k = FFT(image) -> ifft(k) == image up to centring convention."""
    rng = np.random.default_rng(0)
    img_true = rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16))
    k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img_true)))
    mag, recon = fft_reconstruct_image(k)
    np.testing.assert_allclose(recon, img_true, atol=1e-5)
    np.testing.assert_allclose(mag, np.abs(img_true), atol=1e-5)


def test_fft_reconstruct_accepts_real_input():
    """Real k-space is promoted to complex64 internally."""
    real_k = np.zeros((8, 8), dtype=np.float32)
    real_k[4, 4] = 1.0
    mag, img = fft_reconstruct_image(real_k)
    assert img.dtype == np.complex64


# --------------------------------------------------------------------------
# pad_to_cube  (this repo: pad_depth=False by default)
# --------------------------------------------------------------------------


def test_pad_to_cube_3d_no_depth_pad():
    """Default pad_depth=False: only H and W are padded, D stays the same."""
    t = torch.ones(2, 3, 4)   # H=2, W=3, D=4
    out = pad_to_cube(t, 6)
    assert out.shape == (6, 6, 4)   # D unchanged
    assert int(out.sum().item()) == 2 * 3 * 4


def test_pad_to_cube_3d_with_depth_pad():
    """With pad_depth=True all three spatial dims are padded to target_size."""
    t = torch.ones(2, 3, 4)
    out = pad_to_cube(t, 6, pad_depth=True)
    assert out.shape == (6, 6, 6)
    assert int(out.sum().item()) == 2 * 3 * 4


def test_pad_to_cube_4d_preserves_channel_dim():
    """Leading channel dim is not touched; only the 3 spatial dims are padded."""
    t = torch.ones(5, 2, 3, 4)
    out = pad_to_cube(t, 6)
    # channel=5 unchanged; H/W padded to 6; D unchanged (pad_depth=False)
    assert out.shape == (5, 6, 6, 4)


def test_pad_to_cube_already_target_size_is_identity():
    t = torch.ones(4, 4, 4)
    out = pad_to_cube(t, 4)
    assert out.shape == (4, 4, 4)
    assert torch.equal(out, t)


# --------------------------------------------------------------------------
# visualize_kspace_trajectory (mocked plt + Sequence)
# --------------------------------------------------------------------------


def test_visualize_kspace_trajectory_calls_show(monkeypatch):
    """plt.show() must be called exactly once; no real display window opened."""
    import utils_simulation as _us

    N = 16
    ktraj_adc = np.zeros((3, N))
    ktraj_adc[0] = np.linspace(-128, 128, N)
    ktraj = np.zeros((3, N * 4))
    ktraj[0] = np.linspace(-128, 128, N * 4)
    t_adc = np.linspace(0, 1e-3, N)

    class _MockSeq:
        grad_raster_time = 10e-6

        def calculate_kspace(self):
            return ktraj_adc, ktraj, None, None, t_adc

    show_calls = []

    class _FakePlt:
        @staticmethod
        def figure(): pass
        @staticmethod
        def plot(*a, **kw): pass
        @staticmethod
        def title(*a, **kw): pass
        @staticmethod
        def xlabel(*a, **kw): pass
        @staticmethod
        def ylabel(*a, **kw): pass
        @staticmethod
        def axis(*a, **kw): pass
        @staticmethod
        def show(): show_calls.append(1)

    monkeypatch.setattr(_us, "plt", _FakePlt)
    visualize_kspace_trajectory(_MockSeq())
    assert len(show_calls) == 1
