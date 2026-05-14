# Pulseq-DiffusionMESE: Repository Overview

## Purpose

The repository implements a **Pulseq-based pulse sequence design and simulation framework** for quantitative MRI (qMRI). The central goal is to design, simulate, and validate a **triple spin-echo diffusion EPI sequence** capable of simultaneous ADC and T2 mapping from a single acquisition — and to verify its quantitative accuracy via Bloch-equation simulation on digital phantoms.

---

## Architecture

The codebase splits cleanly into two layers.

### 1. Sequence Design (`pulseq_diffusion_mese/`)

A hierarchy of Python classes generates vendor-neutral `.seq` files (Pulseq format):

| Class | Role |
|---|---|
| `PulseqSeq` | Abstract base — RF pulses, spoilers, system limits, timing alignment |
| `SEMultishotPulseqSeq` | Fast spin-echo (FSE/RARE) with configurable echo train length |
| `DiffusionSEMultishotPulseqSeq` | Diffusion-weighted FSE (PGSE prep on first refocusing pulse) |
| `EPIReadout` | Standalone EPI readout module (ramp sampling, blips, partial Fourier, GRAPPA, N/2 ghost navigator labels) |
| `EPIDiffusionSEPulseqSeq` | Single spin-echo diffusion EPI |
| `EPIDiffusionTripleSEPulseqSeq` | **Triple spin-echo diffusion EPI** — the primary deliverable |

The triple-echo sequence fires one RF90, then three successive RF180 pulses each followed by an independent EPI readout. Diffusion encoding (PGSE) is applied only once, on the first RF180. All three echoes share the same diffusion weighting but are acquired at different TEs, giving simultaneous:

- Multi-echo T2 contrast (three TEs per b-value)
- Diffusion-weighted images (b > 0) for ADC
- Optional blip-down/blip-up polarity alternation ([down, up, down]) across the three echoes for B0-distortion field mapping

Hardware safety is enforced throughout: gradient amplitudes, slew rates, ADC dwell times, and inter-echo timing are all constrained to configurable system presets (SAFE, EXTRASAFE, RISKY, EXTREME), with automatic clamping and achieved-b-value reporting when limits are hit.

---

### 2. Simulation & qMRI Analysis (`simulation/`)

Each simulation script runs a complete end-to-end pipeline:

```
BrainWeb phantom (NPZ) → Sequence generation → Bloch simulation (MRzeroCore, GPU)
  → K-space extraction → NUFFT reconstruction (mrinufft/finufft)
  → Quantitative fitting → Validation vs. ground truth
```

Key simulation scripts:

| Script | Sequence used | Metric fitted |
|---|---|---|
| `qMRI_adc.py` | Single-echo EPI | ADC, DTI (FA, MD) |
| `qMRI_t2relax.py` | Single-echo EPI (b=0) | T2 |
| `qMRI_adc_triple_se.py` | Triple-echo EPI | ADC |
| `qMRI_t2relax_triple_se.py` | Triple-echo EPI | T2 |
| `qMRI_adc_diff_multishot_se.py` | Diffusion FSE | ADC |
| `qMRI_t2relax_multishot_se.py` | Multi-echo FSE | T2 |

**Diffusion encoding** uses monopolar Stejskal–Tanner PGSE. Gradients are designed to hit target b-values (0–2300 s/mm²) across multiple directions (3, 6, 12, or 60-direction electrostatic schemes). The trace-DWI (geometric mean across directions) is the input to fitting.

**Fitting** is implemented with two methods for robustness:

- **Log-linear** (OLS on ln(S)): fast, vectorized, biased at low SNR
- **NLLS** (`scipy.optimize.curve_fit` per voxel): unbiased, warm-started from log-linear, with fallback on non-convergence

**EPI reconstruction** handles ramp-sampled k-space (non-Cartesian during trapezoid ramps) via NUFFT, and includes navigator-based Nyquist ghost correction.

**Validation** compares fitted ADC and T2 maps against BrainWeb ground-truth tissue parameters, with optional synthetic tumour insertion (different ADC for core and border) to test sensitivity.

---

## qMRI Quantities Computed

| Quantity | Model | Method |
|---|---|---|
| ADC | S(b) = S₀ · exp(−b · ADC) | NLLS + log-linear |
| T2 | S(TE) = S₀ · exp(−TE/T2) | NLLS + log-linear |
| Diffusion tensor D | ln S = ln S₀ − b·(gᵀDg) | OLS / eigendecomposition |
| Mean Diffusivity (MD) | trace(D)/3 | From tensor |
| Fractional Anisotropy (FA) | Normalised tensor variance | From eigenvalues |

---

## Key Design Decisions

- **Ramp sampling** throughout EPI — maximises temporal efficiency; requires NUFFT rather than FFT.
- **Modular `EPIReadout`** as a composition object (not a base class), allowing it to be plugged into any spin-echo topology.
- **Hardware-raster-aligned timing** — dwell times and gradient durations computed to satisfy both ADC-raster (100 ns) and gradient-raster (10 μs) constraints simultaneously.
- **Pypulseq slew-rate tolerance**: validation applies a 1.75× tolerance factor to account for gradient pre-emphasis, matching pypulseq community practice.

---

## Summary

The repository is a self-contained pipeline from **sequence design → Bloch simulation → qMRI reconstruction → quantitative validation**, built around the triple spin-echo EPI as its flagship sequence. The triple-echo design is the novel contribution: it collapses what would normally require separate T2 and DWI acquisitions into a single TR, while preserving the accuracy of both ADC and T2 estimates — verified in simulation against BrainWeb ground truth.
