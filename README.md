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
│   ├── run_sim.py                  # run all 6 qMRI pipelines
│   ├── run_sim_volume.py           # volume-level runner
│   ├── sim_adc_{sse,triple,multishot}.py
│   ├── sim_t2_{sse,triple,multishot}.py
│   ├── utils_{simulation,diffusion,relaxometry,sim_lib}.py
│   ├── phantom_loader.py
│   ├── pipeline_showcase.py
│   └── simulated/                  # post-processing & comparison scripts
│       ├── process_dist_corrected_{diff,t2}.py
│       ├── compare{Diff,T2}{,_nifti,_nifti_noise,_nifti_combined}.py
│       └── ...
├── brainweb_phantoms/              # BrainWeb phantom generation
│   ├── gen_simulation.py
│   └── gen_simulation_nifti.py
├── tests/                          # pytest test suite
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.9, [pypulseq](https://github.com/imr-framework/pypulseq) ≥ 1.4, and PyTorch (for simulation utilities).

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

## Running the simulation

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
