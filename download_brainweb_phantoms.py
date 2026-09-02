# IQ-BRAIN is funded by the European Union (MSCA Doctoral Network,
# December 2024–November 2028, Grant Agreement No. 101169519).
"""Download and build the BrainWeb phantoms used by the simulation scripts.

``brainweb_phantoms/`` is git-ignored (it's large, external data), so a
fresh clone is missing it - every ``ipy_sim_*.py`` / ``sim_*.py`` script
under ``simulation/`` fails on ``os.listdir(PHANTOMS_DIR_PATH)`` until this
directory exists. This script recreates it by wrapping MRzeroCore's own
BrainWeb downloader (``MRzeroCore.phantom.brainweb``), which:

    1. Downloads the raw segmented tissue-probability maps from the
       BrainWeb Simulated Brain Database
       (https://brainweb.bic.mni.mcgill.ca/brainweb/), caching the raw
       files so re-runs don't re-download.
    2. Assigns literature T1/T2/T2'/ADC values per tissue at 3T and 7T.
    3. Saves each subject as ``brainweb-subj<NN>.nii.gz`` plus a
       ``brainweb-subj<NN>-3T.json`` / ``-7T.json`` descriptor — the exact
       format ``simulation/phantom_loader.py`` reads via
       ``mr0.TissueDict.load()``.

BrainWeb provides 20 subjects; by default only the first one (subject 04,
the one referenced throughout ``simulation/``) is downloaded to keep this
quick. Pass ``--count`` or ``--all`` for more.

Usage
-----
    python download_brainweb_phantoms.py              # subject 04 only (default)
    python download_brainweb_phantoms.py --count 3     # first 3 subjects (04, 05, 06)
    python download_brainweb_phantoms.py --all         # all 20 subjects

Requires MRzeroCore (already a dependency of the simulation scripts) and an
internet connection - the BrainWeb server can be slow, so downloads may
take a while.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "brainweb_phantoms"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of BrainWeb subjects to download, in order "
        "[04, 05, 06, 18, 20, ...] (default: 1, i.e. subject 04 only).",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Download all 20 available BrainWeb subjects instead of --count.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to store the generated phantoms (default: {DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    try:
        import MRzeroCore as mr0
    except ImportError as exc:
        raise SystemExit(
            "MRzeroCore is required to download the BrainWeb phantoms. "
            "Activate the project's environment (with MRzeroCore installed) and retry."
        ) from exc

    subject_count = None if args.all else args.count
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading BrainWeb phantom(s) into {args.output_dir} ...")
    mr0.generate_brainweb_phantoms(str(args.output_dir), subject_count=subject_count)
    print("Done.")


if __name__ == "__main__":
    main()
