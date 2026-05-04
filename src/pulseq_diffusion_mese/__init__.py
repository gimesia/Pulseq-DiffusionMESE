"""Pulseq-DiffusionMESE – diffusion-weighted multi-echo spin-echo pulse sequence."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pulseq-diffusion-mese")
except PackageNotFoundError:
    __version__ = "unknown"

from .sequence import build_sequence

__all__ = ["build_sequence", "__version__"]
