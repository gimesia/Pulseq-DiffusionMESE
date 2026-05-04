# Pulseq-DiffusionMESE

A [Pulseq](https://pulseq.github.io/) pulse sequence for **diffusion-weighted
multi-echo spin echo (MESE)** MRI, implemented in Python.

---

## Repository structure

```
Pulseq-DiffusionMESE/
├── src/
│   └── pulseq_diffusion_mese/   # installable Python package
│       ├── __init__.py
│       └── sequence.py          # sequence builder
├── tests/                       # pytest test suite
│   ├── __init__.py
│   └── test_sequence.py
├── simulation/                  # Bloch-equation simulation scripts
│   └── simulate.py
├── pyproject.toml               # project metadata and build config
├── LICENSE
└── README.md
```

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.9 and [pypulseq](https://github.com/imr-framework/pypulseq) ≥ 1.4.

## Quick start

```python
from pulseq_diffusion_mese import build_sequence

seq = build_sequence(
    n_echoes=8,
    echo_spacing=10e-3,   # s
    b_value=1000,         # s/mm²
    fov=0.22,             # m
    n_slices=1,
)
seq.write("diffusion_mese.seq")
```

## Running the simulation

```bash
python simulation/simulate.py
```

## Running tests

```bash
pytest
```