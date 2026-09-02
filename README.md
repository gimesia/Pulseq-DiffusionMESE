# Pulseq-DiffusionMESE

A [Pulseq](https://pulseq.github.io/) pulse-sequence library for **diffusion-weighted multi-echo spin-echo (MESE)** MRI, implemented in Python.  
Developed as part of the ESMRMB 2026 showcase for the [IQ-BRAIN](https://iq-brain.eu/) MSCA Doctoral Network.

---

## Sequences

| Class | Topology | Primary use |
|---|---|---|
| `EPIDiffusionSEPulseqSeq` | Single spin-echo EPI | Reference DWI |
| `EPIDiffusionTripleSEPulseqSeq` | Triple spin-echo EPI | Simultaneous DWI + T2 mapping |
| `DiffusionSEMultishotPulseqSeq` | Multi-shot FSE Cartesian | Low-distortion DWI |

All sequences share the abstract base `PulseqSeq` and use the standalone `EPIReadout` block.

---

## Repository structure

```
Pulseq-DiffusionMESE/
├── pulseq_diffusion_mese/          # pulse-sequence package
│   ├── PulseqSeq.py                # abstract base class
│   ├── EPIReadout.py               # EPI readout block
│   ├── EPIDiffusionSEPulseqSeq.py  # single spin-echo EPI
│   ├── EPIDiffusionTripleSEPulseqSeq.py  # triple spin-echo EPI
│   ├── SEMultishotPulseqSeq.py     # multi-shot FSE base
│   ├── DiffusionSEMultishotPulseqSeq.py  # multi-shot FSE + diffusion
│   └── utils.py
├── simulation/                     # Bloch-equation simulation pipeline
│   ├── _paths.py                   # repo-relative path resolution (package dir, phantoms dir, ...)
│   ├── run_sim.py                  # run all 6 qMRI pipelines
│   ├── run_sim_volume.py           # volume-level runner
│   ├── sim_adc_{sse,triple,multishot}.py     # callable pipelines, used by run_sim.py
│   ├── sim_t2_{sse,triple,multishot}.py
│   ├── ipy_sim_adc_{sse,triple,multishot}.py # interactive (`# %%` cell) equivalents
│   ├── ipy_sim_t2_{sse,triple,multishot}.py
│   ├── utils_{simulation,diffusion,relaxometry,sim_lib}.py
│   ├── phantom_loader.py
│   ├── pipeline_showcase.py
│   └── simulated/                  # post-processing scripts + pipeline outputs (git-ignored)
│       ├── process_dist_corrected_{diff,t2}.py
│       ├── compare{Diff,T2}{,_nifti,_nifti_noise,_nifti_combined}.py
│       └── ...
├── download_brainweb_phantoms.py   # fetches brainweb_phantoms/ (see "Phantom data" below)
├── brainweb_phantoms/              # BrainWeb phantom data — git-ignored, built locally
├── tests/                          # pytest test suite
├── pyproject.toml
├── requirements.txt                # full simulation-pipeline dependencies
├── LICENSE
└── README.md
```

---

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.9 and [pypulseq](https://github.com/imr-framework/pypulseq) ≥ 1.4. This installs enough to build and write `.seq` files (the "Quick start" below).

The Bloch-equation **simulation pipeline** (everything under `simulation/`) needs a larger stack — MRzeroCore, mri-nufft, PyTorch, nibabel, etc. — pinned in `requirements.txt`:

```bash
pip install -r requirements.txt
```

`requirements.txt` pins CUDA 12.1 PyTorch wheels; edit the `--index-url` line (or drop it) if you're on CPU-only or a different CUDA version.

---

## Quick start

```python
import sys
sys.path.insert(0, "pulseq_diffusion_mese")

from EPIDiffusionTripleSEPulseqSeq import EPIDiffusionTripleSEPulseqSeq

seq = EPIDiffusionTripleSEPulseqSeq(
    fov=0.22,           # m
    n_slices=1,
    b_value=500,        # s/mm²
    n_dirs=6,
)
seq.build_seq()
seq.write("triple_se_diffusion.seq")
```

---

## Phantom data

The simulation pipeline loads a BrainWeb phantom via `simulation/phantom_loader.py`. `brainweb_phantoms/` is git-ignored (it's large, external data), so a fresh clone doesn't have it yet — build it with:

```bash
python download_brainweb_phantoms.py
```

This downloads subject 04 from the [BrainWeb Simulated Brain Database](https://brainweb.bic.mni.mcgill.ca/brainweb/) and writes `brainweb_phantoms/brainweb-subj04/brainweb-subj04-3T.json` (+ NIfTI maps) — the phantom every `sim_*`/`ipy_sim_*` script uses by default. Pass `--count N` for more of the 20 available subjects, or `--all` for all of them; see `--help` for details. Re-running the script re-uses its local cache, so it won't re-download.

---

## Running the simulation

All paths — the `pulseq_diffusion_mese` package, `brainweb_phantoms/`, and the `simulation/simulated/...` output folders — are resolved relative to the repository (see `simulation/_paths.py`), so nothing needs editing after cloning and scripts can be run from any working directory.

To run all six qMRI pipelines (ADC and T2 for each of the three sequence types):

```bash
cd simulation
python run_sim.py
```

Individual pipelines can be run directly, e.g.:

```bash
python sim_adc_triple.py
python sim_t2_triple.py
```

Each `sim_*.py` also has an `ipy_sim_*.py` counterpart with the same pipeline split into `# %%` cells (with plots and intermediate inspection) for step-through in VS Code's Python Interactive window or Jupyter — run it as a plain script the same way, or cell-by-cell. Output folders (`simulated/seq`, `simulated/brainmaps`, `simulated/masks`, ...) are created automatically if missing.

Post-processing and figure generation scripts live under `simulation/simulated/`.

---

## Running tests

```bash
pytest
```

---

## Funding

This work is part of **IQ-BRAIN**, funded by the European Union  
(MSCA Doctoral Network, December 2024 – November 2028, Grant Agreement No. 101169519).

---

## Author

Aron Gimesi — [aron.gimesi@tecnico.ulisboa.pt](mailto:aron.gimesi@tecnico.ulisboa.pt)  
Instituto Superior Técnico | MSCA-DN IQ-BRAIN
