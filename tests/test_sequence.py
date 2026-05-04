"""Tests for the pulseq_diffusion_mese package."""

import pytest
from pulseq_diffusion_mese import __version__, build_sequence
from pulseq_diffusion_mese.sequence import SequenceParams, _diffusion_gradient_amplitude


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------


def test_version_is_string():
    assert isinstance(__version__, str)
    assert len(__version__) > 0


# ---------------------------------------------------------------------------
# SequenceParams defaults
# ---------------------------------------------------------------------------


class TestSequenceParams:
    def test_default_n_echoes(self):
        p = SequenceParams()
        assert p.n_echoes == 8

    def test_default_b_value(self):
        p = SequenceParams()
        assert p.b_value == 1000.0

    def test_default_diffusion_directions(self):
        p = SequenceParams()
        assert p.diffusion_directions == [[1, 0, 0]]

    def test_custom_params(self):
        p = SequenceParams(n_echoes=4, echo_spacing=12e-3, b_value=500)
        assert p.n_echoes == 4
        assert p.echo_spacing == pytest.approx(12e-3)
        assert p.b_value == 500


# ---------------------------------------------------------------------------
# build_sequence (without pypulseq installed returns SequenceParams)
# ---------------------------------------------------------------------------


class TestBuildSequence:
    """These tests are designed to run with or without pypulseq installed."""

    def _result(self, **kwargs):
        """Call build_sequence and return the result."""
        return build_sequence(**kwargs)

    def test_returns_something(self):
        result = self._result()
        assert result is not None

    def test_default_call_succeeds(self):
        """build_sequence() must not raise with default arguments."""
        self._result()

    def test_b0_reference(self):
        result = self._result(b_value=0)
        assert result is not None

    def test_custom_n_echoes(self):
        result = self._result(n_echoes=4)
        # When pypulseq is absent the result is a SequenceParams placeholder.
        if isinstance(result, SequenceParams):
            assert result.n_echoes == 4

    def test_custom_fov(self):
        result = self._result(fov=0.30)
        if isinstance(result, SequenceParams):
            assert result.fov == pytest.approx(0.30)

    def test_multi_direction(self):
        directions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        result = self._result(diffusion_directions=directions)
        if isinstance(result, SequenceParams):
            assert result.diffusion_directions == directions


# ---------------------------------------------------------------------------
# Diffusion gradient amplitude helper
# ---------------------------------------------------------------------------


class TestDiffusionGradientAmplitude:
    """Unit tests for the Stejskal–Tanner amplitude calculation."""

    def _make_gx_stub(self):
        """Minimal stub for the gx gradient object."""
        from types import SimpleNamespace

        return SimpleNamespace(
            area=1.0, flat_time=6.4e-3, rise_time=1e-4, duration=7e-3
        )

    def test_b0_gives_zero_amplitude(self):
        params = SequenceParams(b_value=0)
        amp = _diffusion_gradient_amplitude(params, self._make_gx_stub())
        assert amp == pytest.approx(0.0)

    def test_nonzero_b_gives_positive_amplitude(self):
        params = SequenceParams(b_value=1000)
        amp = _diffusion_gradient_amplitude(params, self._make_gx_stub())
        assert amp > 0

    def test_higher_b_gives_higher_amplitude(self):
        gx = self._make_gx_stub()
        amp_low = _diffusion_gradient_amplitude(SequenceParams(b_value=500), gx)
        amp_high = _diffusion_gradient_amplitude(SequenceParams(b_value=1000), gx)
        assert amp_high > amp_low

    def test_amplitude_within_system_limits(self):
        params = SequenceParams(b_value=1000, max_grad=80e-3)
        amp = _diffusion_gradient_amplitude(params, self._make_gx_stub())
        # Result is in T/m; must be positive and below an extreme upper bound (1 T/m).
        assert 0 < amp < 1.0
