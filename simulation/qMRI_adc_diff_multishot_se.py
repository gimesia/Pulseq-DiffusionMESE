# IQ-BRAIN is funded by the European Union (MSCA Doctoral Network,
# December 2024–November 2028, Grant Agreement No. 101169519).

# %% ================================================================================
#  Imports
# ===================================================================================
import logging
import warnings
import os
import sys
import numpy as np
import MRzeroCore as mr0
import numpy as np
import torch
import matplotlib.pyplot as plt
import nibabel as nib

from pypulseq import Sequence

import phantom_loader


# The path to the pulseq-diffusion-mese directory.
# TODO: It is advisable to replace this with a more robust method for path management,
# such as using environment variables or a configuration file.
seq_path = r"C:\Users\User\OneDrive\PhD\Sumbission\ESMRMB26\Pulseq-DiffusionMESE\pulseq_diffusion_mese"
if seq_path not in sys.path:
    sys.path.append(seq_path)

# %%
from DiffusionSEMultishotPulseqSeq import DiffusionSEMultishotPulseqSeq
from utils import SystemLimitType
from utils_simulation import *
from mrinufft import get_operator

logger = logging.getLogger()
logger.setLevel(logging.FATAL)  # Suppress INFO and WARNING
warnings.filterwarnings("ignore", category=UserWarning, module="mrinufft")

np.int = int
np.float = float
np.complex = complex

use_GPU = torch.cuda.is_available()
PHANTOM_IDX = 0

# ================================================================================
#   Paths
# ================================================================================
SEQUENCES_DIR_PATH = r".\simulated\seq"
VOLUMES_DIR_PATH = r".\simulated\brainmaps"
PHANTOMS_DIR_PATH = r"C:\Users\User\OneDrive\PhD\Sumbission\ESMRMB26\Pulseq-DiffusionMESE\brainweb_phantoms"
ECHO_IMAGES_DIR_PATH = r"C:\Users\User\OneDrive\PhD\Sumbission\ESMRMB26\Pulseq-DiffusionMESE\simulation\simulated\TE"

# %% ==============================================================================
#   Simulation parameters
# ==============================================================================
fov = 224e-3
res = 2.33333333
slice_thickness = res * 1e-3

# Multiple TEs: ADC fit averages across TEs (T2 cancels in S(b)/S(0) ratio).
# DTI fit uses ONLY the shortest TE for highest SNR. The DTI log-linear model
# ln S = ln S0 - b * g^T D g assumes a single TE; mixing TEs makes S0 a function
# of b-index (because high-b acquisitions weight differently across TEs via SNR),
# which biases the tensor fit and inflates spurious FA.
TE_VALUES = [
    100,
]  # 150, 200]         # [ms] — three echo times
TE_FOR_DTI = TE_VALUES[0]  # use shortest TE for DTI (best SNR)
TR = 5000  # [ms]
ETL = 1  # echo train length (1 = conventional SE per shot)

b_values = np.arange(0, 2001, 100, dtype=int)  # [s/mm²]
B_DIRS = 12  # 12 directions → full-rank, well-conditioned tensor fit
small_delta = 0.018  # [s]
big_DELTA = 0.03  # [s]

Nx = Ny = int(fov / slice_thickness)
N_shots = Ny // ETL

print(
    f"Matrix: {Ny}*{Nx}  ETL={ETL}  N_shots={N_shots}  "
    f"TEs: {TE_VALUES} ms (DTI uses {TE_FOR_DTI} ms)  "
    f"b-values: {b_values}  directions: {B_DIRS}"
)

# %% ==============================================================================
#   Load phantom
# ==============================================================================
phantoms = [f for f in os.listdir(PHANTOMS_DIR_PATH) if "brainweb" in f]
print("Available phantoms:", phantoms)
phantom_path = os.path.join(
    PHANTOMS_DIR_PATH, phantoms[PHANTOM_IDX], f"{phantoms[PHANTOM_IDX]}-3T.json"
)
print(f"Loading phantom from {phantom_path} ...")

phantom, phantom_data = phantom_loader.load_phantom(
    json_path=phantom_path,
    resolution_mm=res,
    slice_idx=None,
)
D = phantom.D
T2 = phantom.T2

# %% ================================================================================
#   Main simulation loop: sweep TEs × b-values
# ================================================================================
# Storage layout: images_per_te[te_idx] = (n_b, n_dirs, Ny, Nx)
# This lets us:
#   - Average across TEs for ADC (mag_images_adc)
#   - Pick only the shortest TE for DTI (mag_images_dti)
# without re-simulating anything.
# ================================================================================
images_per_te = {te: [] for te in TE_VALUES}
n_dirs = None
last_seq = None  # keep reference for b_directions in DTI fit

for te in TE_VALUES:
    for b_value in b_values:
        # ============================================================================
        #   Build sequence
        # ============================================================================
        name = f"DiffSEMultishot-b{int(b_value)}-te{int(te)}"
        seq = DiffusionSEMultishotPulseqSeq(
            name=name,
            fov=fov,
            Nx=Nx,
            Ny=Ny,
            slice_thickness=slice_thickness,
            TR=TR,
            TE=te,
            ETL=ETL,
            b_value=b_value,
            b_directions=B_DIRS,
            small_delta=small_delta,
            big_DELTA=big_DELTA,
            save_dir=SEQUENCES_DIR_PATH,
            v141_compat=True,
            system_type=SystemLimitType.EXTRASAFE,
            logger=logger,
        )
        seq.build_seq()
        seq.write()
        last_seq = seq

        if n_dirs is None:
            n_dirs = len(seq.b_directions)
            print(f"n_dirs={n_dirs}")

        seq_filename = seq.get_save_filename()
        print(f"[TE={te}ms, b={b_value}]  file: {seq_filename}")

        # ============================================================================
        #   Simulate
        # ============================================================================
        seq0 = mr0.Sequence.import_file(rf"{SEQUENCES_DIR_PATH}\{seq_filename}")
        seq0_gpu = None
        phantom_data_gpu = None
        if use_GPU:
            seq0_gpu = seq0.cuda()
            phantom_data_gpu = phantom_data.cuda()
            graph = mr0.compute_graph(seq0_gpu, phantom_data_gpu, 50000, 1e-6)
            signal = mr0.execute_graph(
                graph, seq0_gpu, phantom_data_gpu, print_progress=True
            ).cpu()
        else:
            phantom_data_cpu = phantom_data.cpu()
            graph = mr0.compute_graph(seq0, phantom_data_cpu, 5000, 1e-5)
            signal = mr0.execute_graph(
                graph, seq0, phantom_data_cpu, print_progress=False
            )

        # Clean up GPU memory
        if seq0_gpu is not None:
            del seq0_gpu
        if phantom_data_gpu is not None:
            del phantom_data_gpu
        if use_GPU:
            torch.cuda.empty_cache()

        assert (
            signal.shape[0] == n_dirs * Ny * Nx
        ), f"Signal length mismatch: expected {n_dirs * Ny * Nx}, got {signal.shape[0]}"

        # ============================================================================
        #   Reconstruct per direction (Cartesian iFFT)
        # ============================================================================
        dir_images = []
        for d in range(n_dirs):
            kspace = signal[d * Ny * Nx : (d + 1) * Ny * Nx].numpy().reshape(Ny, Nx)
            img_mag, _ = fft_reconstruct_image(kspace, use_gpu=use_GPU)
            dir_images.append(img_mag.squeeze())
        images_per_te[te].append(np.stack(dir_images, axis=0))  # (n_dirs, Ny, Nx)

    print(f"  Completed TE={te} ms: {len(images_per_te[te])} b-values × {n_dirs} dirs")

print("Simulation loop complete.")

# %% ==============================================================================
#   Assemble image stacks
# ==============================================================================
# Stack into (n_te, n_b, n_dirs, Ny, Nx) then derive the two views we need.
stacked = np.stack(
    [np.stack(images_per_te[te], axis=0) for te in TE_VALUES],
    axis=0,
)  # (n_te, n_b, n_dirs, Ny, Nx)

# ADC: average across TEs (T2 cancels in S(b)/S(0) ratio)
mag_images_adc = np.abs(stacked.mean(axis=0))  # (n_b, n_dirs, Ny, Nx)

# DTI: use only the shortest TE (single-TE log-linear model, highest SNR)
te_idx_dti = TE_VALUES.index(TE_FOR_DTI)
mag_images_dti = np.abs(stacked[te_idx_dti])  # (n_b, n_dirs, Ny, Nx)

print(f"mag_images_adc shape (TE-averaged): {mag_images_adc.shape}")
print(f"mag_images_dti shape (TE={TE_FOR_DTI} ms only): {mag_images_dti.shape}")

# %% ==============================================================================
#   Diagnostic: b=0 direction consistency check
# ==============================================================================
# At b=0 the diffusion gradients have zero amplitude → all directions should produce
# IDENTICAL images. Any direction-to-direction variation at b=0 indicates a sequence
# artefact (e.g., residual eddy currents, timing mismatch between Gdiff axes) that
# will corrupt the tensor fit at b > 0.
b0_idx = int(np.argmin(b_values))
b0_per_dir = mag_images_dti[b0_idx]  # (n_dirs, Ny, Nx)
b0_mean = b0_per_dir.mean(axis=0)
brain_mask_b0 = b0_mean > 0.1 * b0_mean.max()

print("\n=== b=0 direction consistency check ===")
for d in range(n_dirs):
    rel_diff = (b0_per_dir[d] - b0_mean) / (b0_mean + 1e-9)
    rms = np.sqrt(np.mean(rel_diff[brain_mask_b0] ** 2))
    print(
        f"  dir {d} ({last_seq.b_directions[d]}): RMS relative diff vs mean = {rms*100:.3f}%"
    )
print("If any RMS > ~1% the sequence has direction-dependent artefacts at b=0.\n")

# %% ==============================================================================
#   Visualize — show a subset of b-values for direction 0 (TE-averaged ADC stack)
# ==============================================================================
SHOW_DIR = 0
b_subset = np.linspace(0, len(b_values) - 1, min(6, len(b_values)), dtype=int)

fig, axs = plt.subplots(1, len(b_subset), figsize=(3 * len(b_subset), 3))
fig.suptitle(f"Diffusion SE Multishot DWI (direction {SHOW_DIR}, TE-averaged)")
for col, b_idx in enumerate(b_subset):
    axs[col].imshow(np.rot90(mag_images_adc[b_idx, SHOW_DIR], -1), cmap="gray")
    axs[col].set_title(f"b={b_values[b_idx]:.0f}")
    axs[col].set_axis_off()
plt.tight_layout()
plt.show()

# %% ==============================================================================
#   Per-direction comparison at mid b-value (sanity check for striping)
# ==============================================================================
b_mid_idx = len(b_values) // 2
fig, axs = plt.subplots(1, min(n_dirs, 6), figsize=(3 * min(n_dirs, 6), 3))
fig.suptitle(f"Per-direction DWI at b={b_values[b_mid_idx]:.0f} (TE={TE_FOR_DTI} ms)")
for d in range(min(n_dirs, 6)):
    axs[d].imshow(np.rot90(mag_images_dti[b_mid_idx, d], -1), cmap="gray")
    axs[d].set_title(f"dir {d}\n{np.round(last_seq.b_directions[d], 2)}")
    axs[d].set_axis_off()
plt.tight_layout()
plt.show()

# %% ==============================================================================
#   Trace DWI — geometric mean across directions (TE-averaged stack)
# ==============================================================================
eps = 1e-12


def compute_trace_dwi(mag):
    """Geometric mean across diffusion directions. mag: (n_b, n_dirs, Ny, Nx)."""
    return np.exp(np.mean(np.log(mag + eps), axis=1))


trace_dwi = compute_trace_dwi(mag_images_adc)  # (n_b, Ny, Nx)
print(f"Trace DWI shape: {trace_dwi.shape}")

# %% ==============================================================================
#   ADC maps (uses TE-averaged trace DWI)
# ==============================================================================
from utils_diffusion import create_adc_map  # noqa: E402

adc_nlls, _ = create_adc_map(trace_dwi, b_values, method="nlls")
adc_ll, _ = create_adc_map(trace_dwi, b_values, method="loglinear")

mask = adc_nlls > 0
print(
    f"ADC NLLS:       range [{adc_nlls.min()*1e3:.3f}, {adc_nlls.max()*1e3:.3f}] "
    f"x10⁻³ mm²/s, median (brain) = {np.median(adc_nlls[mask])*1e3:.3f}"
)
print(
    f"ADC log-linear: range [{adc_ll.min()*1e3:.3f}, {adc_ll.max()*1e3:.3f}] "
    f"x10⁻³ mm²/s, median (brain) = {np.median(adc_ll[mask])*1e3:.3f}"
)

# %% ==============================================================================
#   ADC comparison plot (NLLS vs log-linear)
# ==============================================================================
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
titles = ["NLLS Fit", "Log-Linear Fit"]
for i, adc in enumerate([adc_nlls, adc_ll]):
    im = axs[i].imshow(np.fliplr(adc) * 1e3, cmap="viridis")
    axs[i].set_title(titles[i])
    axs[i].set_axis_off()
    fig.colorbar(im, ax=axs[i], label="ADC (x10⁻³ mm²/s)")
plt.suptitle(
    f"Diffusion SE Multishot ADC (TEs={TE_VALUES} ms, ETL={ETL}, "
    f"{B_DIRS} directions, TE-averaged)"
)
plt.tight_layout()
plt.show()

# %% ==============================================================================
#   DTI maps — FA and MD (uses single-TE stack, NOT TE-averaged)
# ==============================================================================
from utils_diffusion import create_dti_maps  # noqa: E402

fa_map, md_map, eigvals_map, dti_s0_map = create_dti_maps(
    mag_images_dti,  # (n_b, n_dirs, Ny, Nx) — single TE only
    b_values,
    last_seq.b_directions,  # (n_dirs, 3) unit vectors
)

# Report distribution rather than just min/max — easier to spot noise-floor FA
brain_mask_md = md_map > 0
if brain_mask_md.any():
    fa_brain = fa_map[brain_mask_md]
    print(
        f"FA in brain: median={np.median(fa_brain):.3f}, "
        f"p95={np.percentile(fa_brain, 95):.3f}, "
        f"max={fa_brain.max():.3f}"
    )
    print(
        f"MD in brain: median={np.median(md_map[brain_mask_md])*1e3:.3f}, "
        f"range=[{md_map[brain_mask_md].min()*1e3:.3f}, "
        f"{md_map[brain_mask_md].max()*1e3:.3f}] x10⁻³ mm²/s"
    )
print(
    "NOTE: BrainWeb phantom has isotropic D (scalar). True FA = 0 everywhere; "
    "any non-zero FA is fit noise. With 12 directions the noise floor should be "
    "much flatter and lower than with 3."
)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
im = axs[0].imshow(np.fliplr(md_map) * 1e3, cmap="viridis")
axs[0].set_title("Mean Diffusivity (MD)")
axs[0].set_axis_off()
fig.colorbar(im, ax=axs[0], label="MD (x10⁻³ mm²/s)")

# Auto-scale FA vmax to the 99th percentile of brain voxels — avoids hot-spot
# pixels from dominating the colormap, and matches what a clinical viewer does.
fa_vmax = np.percentile(fa_brain, 99) if brain_mask_md.any() else 0.1
im = axs[1].imshow(np.fliplr(fa_map), cmap="inferno", vmin=0, vmax=fa_vmax)
axs[1].set_title(f"Fractional Anisotropy (FA)  [vmax={fa_vmax:.3f}]")
axs[1].set_axis_off()
fig.colorbar(im, ax=axs[1], label="FA")
plt.suptitle(
    f"Diffusion SE Multishot DTI (TE={TE_FOR_DTI} ms, ETL={ETL}, "
    f"{B_DIRS} directions, single-TE fit)"
)
plt.tight_layout()
plt.show()

# %% ==============================================================================
#   Reference phantom comparison (ADC)
# ==============================================================================
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Diffusion SE Multishot ADC vs Reference")
ims_data = [
    (np.fliplr(adc_nlls) * 1e3, "NLLS ADC", "viridis"),
    (np.fliplr(adc_ll) * 1e3, "Log-Linear ADC", "viridis"),
    (np.rot90(D, 1), "Reference D map", "viridis"),
]
for ax, (im_data, title, cmap) in zip(axs, ims_data):
    im = ax.imshow(im_data, cmap=cmap)
    ax.set_title(title)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, label="ADC (x10⁻³ mm²/s)")
plt.tight_layout()
plt.show()

# %%
try:
    np.save(rf"{VOLUMES_DIR_PATH}\ADC_multishot_se.npy", adc_nlls)
    print("Saved ADC/DTI maps.")
except Exception as e:
    print(f"Could not save: {e}")
# %%
