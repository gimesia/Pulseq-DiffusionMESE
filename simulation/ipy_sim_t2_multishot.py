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
from matplotlib.colors import LogNorm
from cmcrameri import cm
import nibabel as nib

from pypulseq import Sequence

import phantom_loader
import _paths

# The path to the pulseq-diffusion-mese directory, resolved relative to
# this repo so the script works after any clone.
seq_path = str(_paths.PACKAGE_DIR)
if seq_path not in sys.path:
    sys.path.append(seq_path)

# %%
from SEMultishotPulseqSeq import SEMultishotPulseqSeq
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
SEQUENCES_DIR_PATH = _paths.SIMULATION_DIR / "simulated" / "seq"
VOLUMES_DIR_PATH = _paths.SIMULATION_DIR / "simulated" / "brainmaps"
PHANTOMS_DIR_PATH = _paths.PHANTOMS_DIR_PATH
ECHO_IMAGES_DIR_PATH = _paths.SIMULATION_DIR / "simulated" / "t2_img"
os.makedirs(ECHO_IMAGES_DIR_PATH, exist_ok=True)

# %% ==============================================================================
#   Simulation parameters
# ==============================================================================
fov = 224e-3
res = 2.33333333
slice_thickness = res * 1e-3
# TEs = np.arange(65, 200, 5)  # ms — 12 TE values
TEs = np.array([ 65,  70,  75,  80,  85,  90,  95, 100, 105, 110, 115,120, 123, 125, 128, 130, 133, 135, 138, 140, 
                143, 145,148, 150, 153, 158, 163, 168, 173, 178, 181, 183, 186, 188, 191, 193, 196, 198, 201, 203, 
                206, 208, 211, 216, 221, 226, 231, 236, 241, 246, 251, 256, 261, 266], dtype=int) # Match triple SE TEs
TR = 5000
ETL = 1  # echo train length
Nx = Ny = int(fov / slice_thickness)

N_shots = Ny // ETL
print(
    f"Matrix: {Ny}x{Nx}  ETL={ETL}  N_shots per TE={N_shots}  "
    f"Total TRs for all TEs: {len(TEs) * N_shots}  "
    f"(vs {len(TEs) * Ny} for ETL=1 — {ETL}× faster)"
)

# %% ==============================================================================
#   Load phantom
# =================================================================================
phantoms = [f for f in os.listdir(PHANTOMS_DIR_PATH) if "brainweb" in f]
print("Available phantoms:", phantoms)
phantom_path = os.path.join(
    PHANTOMS_DIR_PATH, phantoms[PHANTOM_IDX], f"{phantoms[PHANTOM_IDX]}-3T.json"
)
print(f"Loading phantom from {phantom_path} ...")

phantom, phantom_data, tissue_masks = phantom_loader.load_phantom(
    json_path=phantom_path,
    resolution_mm=res,
    slice_idx=None,
)
D = phantom.D
T2 = phantom.T2

# %% ================================================================================
#   Main simulation loop
# ================================================================================
# For each TE: build the multishot SE sequence, simulate all N_shots TRs, reshape
# the Cartesian k-space, and reconstruct with a plain inverse FFT.
# No NUFFT is needed because the readout is a conventional Cartesian gradient echo.
# ================================================================================
reconstructed_images = []  # (n_TEs,) list of (Ny, Nx) magnitude images

for te in TEs:
    print(f"Simulating for TE={te} ms ...")
    # ============================================================================
    #   Build sequence
    # ============================================================================
    seq = SEMultishotPulseqSeq(
        name="SEMultishot",
        fov=fov,
        Nx=Nx,
        Ny=Ny,
        slice_thickness=slice_thickness,
        TR=TR,
        TE=int(te),
        ETL=ETL,
        save_dir=SEQUENCES_DIR_PATH,
        v141_compat=True,
        system_type=SystemLimitType.EXTREME,
        logger=logger,
    )
    seq.build_seq()
    seq.write()

    seq_filename = seq.get_save_filename()
    print(f"[TE={te} ms]  N_shots={N_shots}  file: {seq_filename}")

    # ============================================================================
    #   Simulate
    # ============================================================================
    seq0 = mr0.Sequence.import_file(rf"{SEQUENCES_DIR_PATH}\{seq_filename}")
    if use_GPU:
        seq0_gpu = seq0.cuda()
        phantom_data_gpu = phantom_data.cuda()
        graph = mr0.compute_graph(seq0_gpu, phantom_data_gpu, 20000, 1e-5)
        signal = mr0.execute_graph(
            graph, seq0_gpu, phantom_data_gpu, print_progress=True
        ).cpu()
        del seq0_gpu, phantom_data_gpu
        torch.cuda.empty_cache()
    else:
        phantom_data_cpu = phantom_data.cpu()
        graph = mr0.compute_graph(seq0, phantom_data_cpu, 2000, 1e-4)
        signal = mr0.execute_graph(graph, seq0, phantom_data_cpu, print_progress=False)
    try:
        del seq0_gpu, phantom_data_gpu
    except Exception:
        pass
    torch.cuda.empty_cache()

    # ============================================================================
    #   Assemble k-space and reconstruct
    # ============================================================================
    # Signal layout: [shot0_echo0…echo(ETL-1), shot1_echo0…, …, shot(N_shots-1)_echo(ETL-1)]
    # Each echo contributes Nx ADC samples → total = Ny * Nx (linear ky ordering).
    # Reshape directly to (Ny, Nx); the ky ordering matches phase_encoding_gradients.
    kspace = signal.numpy().reshape(Ny, Nx)
    img_mag, _ = fft_reconstruct_image(kspace, use_gpu=use_GPU)
    reconstructed_images.append(img_mag.squeeze())
    print(f"  Reconstructed image shape: {img_mag.squeeze().shape}")

print(f"\nDone. {len(reconstructed_images)} images collected.")

# %% ==============================================================================
#   Visualize reconstructed images
# ==============================================================================
n_cols = 6
n_rows = int(np.ceil(len(TEs) / n_cols))

fig, axs = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
axs = axs.flatten()
fig.suptitle(
    f"Multishot SE T2 relaxometry  (ETL={ETL}, {N_shots} shots/TE, "
    f"total TRs={len(TEs)*N_shots} vs {len(TEs)*Ny} for ETL=1 — {ETL}× faster)"
)
for i, img in enumerate(reconstructed_images):
    axs[i].imshow(np.rot90(img, -1), cmap="gray")
    axs[i].set_title(f"TE={TEs[i]} ms")
    axs[i].set_axis_off()
for ax in axs[len(TEs) :]:
    ax.set_visible(False)
plt.tight_layout()
plt.show()

# %% ==============================================================================
#   T2 fitting
# ==============================================================================
from utils_relaxometry import create_t2_map

images_stack = np.array(reconstructed_images)  # (n_TEs, Ny, Nx)
print(f"Images stack: {images_stack.shape}  TEs: {TEs}")

ims = []
for method in ["nlls", "loglinear"]:
    t2_result = create_t2_map(images_stack, TEs, method=method)
    ims.append(t2_result[0])

# T2 colour convention (ISMRM QMR Study Group Color Recommendation
# Committee, Fuderer/Wichtmann/Crameri et al., MRM 2025;93:490-506):
# perceptually-uniform `navia` on a log scale. Background/rejected
# voxels (T2 <= 0) are masked and rendered black rather than folded
# into the log scale, since log(0) is undefined. vmin=10 ms is a
# placeholder floor (below in-vivo brain WM/GM at 3T, above the
# noise-floor voxels) — flagged for review rather than a silent
# choice. vmax is tied to the reference T2 map's own max, shared with
# the comparison plot below.
T2_LOG_VMIN_MS = 10.0
navia_bad = cm.navia.copy()
navia_bad.set_bad("black")
vmax_shared = float(np.asarray(T2).max())

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
titles = ["NLLS Fit", "Log-Linear Fit"]
for i, ax in enumerate(axs):
    im_data_ms = np.rot90(ims[i], -1)
    im_data_ms = np.ma.masked_where(im_data_ms <= 0, im_data_ms)
    im = ax.imshow(
        im_data_ms, cmap=navia_bad,
        norm=LogNorm(vmin=T2_LOG_VMIN_MS, vmax=vmax_shared * 1000),
    )
    ax.set_title(titles[i])
    fig.colorbar(im, ax=ax, label="T2 (ms)")
plt.suptitle(
    f"Multishot SE T2 maps (ETL={ETL}, {len(TEs)} TEs, {len(TEs)*N_shots} TRs)"
)
plt.tight_layout()
plt.show()

# %% ==============================================================================
#   Final comparison plot
# ==============================================================================
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    f"Multishot SE T2 mapping — ETL={ETL}  "
    f"{len(TEs)*N_shots} TRs ({len(TEs)*N_shots*TR/1000:.0f} s)  "
    f"vs single SE: {len(TEs)*Ny} TRs ({len(TEs)*Ny*TR/1000:.0f} s)  "
    f"[{ETL}× fewer shots]"
)
titles = ["NLLS Fit", "Log-Linear Fit", "Reference T2 map"]
ims2 = [*ims, T2]

# Shared colour scale across all three panels, tied to the reference
# map's own T2 range (rather than a hardcoded constant, which would
# silently go stale for a different phantom/slice). The pixels this
# clips are inspected in the next cell.
for i, ax in enumerate(axs):
    if i < 2:
        im_data = np.rot90(ims2[i], -1) / 1000
    else:
        im_data = np.rot90(ims2[i], -1)
    ims2[i] = im_data  # save for later (unmasked — QC cell below needs raw values)
    im = ax.imshow(
        np.ma.masked_where(im_data <= 0, im_data),
        cmap=navia_bad,
        norm=LogNorm(vmin=T2_LOG_VMIN_MS / 1000, vmax=vmax_shared),
    )
    ax.set_title(titles[i])
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, label="T2 (s)")
plt.tight_layout()
plt.show()

# %% ==============================================================================
#   QC: pixels clipped by the shared colour scale above
# =================================================================================
# Distinct from the implausible-fit rejection already done inside
# create_t2_map (see utils_relaxometry.py): those voxels are zeroed at
# the source and don't show up here. This is the remainder —
# legitimate, non-zero fitted T2 values that simply exceed the
# reference map's max and get visually saturated to the top colour of
# vmax_shared above.
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for ax, im_data, title in zip(axs, ims2[:2], titles[:2]):
    clipped = im_data > vmax_shared
    n_clipped = int(clipped.sum())
    n_fitted = int((im_data > 0).sum())
    pct = 100 * n_clipped / n_fitted if n_fitted else 0.0
    ax.imshow(clipped, cmap="Reds", vmin=0, vmax=1)
    ax.set_title(f"{title}: {n_clipped}/{n_fitted} px clipped ({pct:.1f}%)")
    ax.set_axis_off()
plt.tight_layout()
plt.show()

# %%
try:
    np.save(rf"{VOLUMES_DIR_PATH}\{phantoms[PHANTOM_IDX]}-T2_multishot_se.npy", ims2[0])
    np.save(rf"{VOLUMES_DIR_PATH}\{phantoms[PHANTOM_IDX]}-T2_ref.npy", ims2[2])
    print("Saved T2 maps.")
except Exception as e:
    print(f"Could not save: {e}")
# %%
