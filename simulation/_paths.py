# IQ-BRAIN is funded by the European Union (MSCA Doctoral Network,
# December 2024–November 2028, Grant Agreement No. 101169519).
"""Repo-relative path resolution.

Centralises the filesystem locations that used to be hardcoded to one
machine (``C:\\Users\\...\\Pulseq-DiffusionMESE``) so the simulation
scripts work for anyone who clones the repository, regardless of where it
lives on disk. Every path is derived from this file's own location, the
same approach already used in ``tests/conftest.py``.

Usage
-----
    import _paths

    seq_path = str(_paths.PACKAGE_DIR)
    PHANTOMS_DIR_PATH = _paths.PHANTOMS_DIR_PATH
"""

from pathlib import Path

# simulation/_paths.py -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent

SIMULATION_DIR = REPO_ROOT / "simulation"
PACKAGE_DIR = REPO_ROOT / "pulseq_diffusion_mese"

# Git-ignored external data (not shipped in the repo) — expected to sit at
# the repo root, e.g. after downloading/extracting the BrainWeb phantoms.
PHANTOMS_DIR_PATH = REPO_ROOT / "brainweb_phantoms"
