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
import _paths

# The path to the pulseq-diffusion-mese directory, resolved relative to
# this repo so the script works after any clone.
seq_path = str(_paths.PACKAGE_DIR)
if seq_path not in sys.path:
    sys.path.append(seq_path)

# %%
from EPIDiffusionTripleSEPulseqSeq import EPIDiffusionTripleSEPulseqSeq
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
BLIP_DOWN = True  # Whether to use blip-down or blip-up EPI readout (affects distortion direction)
PHANTOM_IDX = 0

# =================================================================================
#   Paths
# =================================================================================
SEQUENCES_DIR_PATH = _paths.SIMULATION_DIR / "simulated" / "seq"
VOLUMES_DIR_PATH = _paths.SIMULATION_DIR / "simulated" / "brainmaps"
PHANTOMS_DIR_PATH = _paths.PHANTOMS_DIR_PATH
ECHO_IMAGES_DIR_PATH = _paths.SIMULATION_DIR / "simulated" / "t2_img"
os.makedirs(ECHO_IMAGES_DIR_PATH, exist_ok=True)

# %% ==============================================================================
#   Simulation parameters
# =================================================================================
fov = 224e-3
res = 2.33333333
slice_thickness = res * 1e-3

# Triple SE: sweep ~15 TE1 values instead of 45.
# Each TR yields 3 echoes (TE1, TE2, TE3 auto-computed), giving ~45 (TE, image)
# pairs with only ~15 TRs — a 3× scan-time reduction vs the single-echo approach.
TEs_TE1 = np.arange(65, 155, 5, dtype=int)  # 15 TE1 values [ms]
TR = 5000
b_value = 0
Nx = Ny = int(fov / slice_thickness)


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
#   Main simulation loop: generate triple SE sequence, simulate, separate
#   k-spaces for all three echoes, reconstruct images.
# =================================================================================
# Collect magnitude images and echo times from all TRs and all 3 echoes.
all_echo_images = []  # each entry: (Ny, Nx) magnitude image
all_echo_tes = []  # corresponding echo time in ms

for te1 in TEs_TE1:
    print(f"Simulating for TE1={te1} ms ...")
    # =================================================================================
    #   Generate sequence
    # =================================================================================
    name = f"DiffTripleSE-TE1-{te1}"
    seq = EPIDiffusionTripleSEPulseqSeq(
        name=name,
        resolution=res,
        Nx=Nx,
        Ny=Ny,
        fov=fov,
        slice_thickness=slice_thickness,
        TE=te1,
        TR=TR,
        b_value=b_value,
        b_directions=1,
        b_0_frequency=0,
        save_dir=SEQUENCES_DIR_PATH,
        save_name=name,
        v141_compat=True,
        small_delta=0.018,
        big_DELTA=0.03,
        system_type=SystemLimitType.EXTREME,
        calibration_readout=True,
        uniform_spoiler_directions=False,
        uniform_spoiler_areas=True,
        phase_cycling=False,
        blip_down=BLIP_DOWN,
        logger=logger,
        fit_epi=True,
    )
    seq.write()

    # Read back the actual echo times (stored in seconds internally)
    te1_ms = seq.TE * 1e3
    te2_ms = seq.TE2 * 1e3
    te3_ms = seq.TE3 * 1e3
    print(f"[TE1={te1} ms] actual TEs: {te1_ms:.1f} / {te2_ms:.1f} / {te3_ms:.1f} ms")

    # =================================================================================
    #   Simulate sequence
    # =================================================================================
    seq0 = mr0.Sequence.import_file(rf"{SEQUENCES_DIR_PATH}\{name}.seq")
    if use_GPU:
        seq0_gpu = seq0.cuda()
        phantom_data_gpu = phantom_data.cuda()
        graph = mr0.compute_graph(seq0_gpu, phantom_data_gpu, 20000, 1e-5)
        signal = mr0.execute_graph(
            graph, seq0_gpu, phantom_data_gpu, print_progress=True
        ).cpu()
        del seq0_gpu
        del phantom_data_gpu
        torch.cuda.empty_cache()
    else:
        phantom_data_cpu = phantom_data.cpu()
        graph = mr0.compute_graph(seq0, phantom_data_cpu, 2000, 1e-4)
        signal = mr0.execute_graph(graph, seq0, phantom_data_cpu, print_progress=False)
    try:
        del seq0_gpu
        del phantom_data_gpu
    except Exception:
        pass
    torch.cuda.empty_cache()

    # ==============================================================================
    #   Separate k-spaces
    # =================================================================================
    # Signal layout per TR:
    #   [calibration: 3×Nx] | [dir0: EPI1|EPI2|EPI3] | [dir1: ...] | [dir2: ...]
    # EPI1 may use partial Fourier; EPI2 and EPI3 always use pff=1.0.
    samples_per_cal = int(3 * seq.adc.num_samples)
    samples_epi1 = int(seq.Ny * seq.partial_fourier_factor) * seq.adc.num_samples
    samples_epi2 = seq.Ny * seq.adc.num_samples
    samples_epi3 = seq.Ny * seq.adc.num_samples
    samples_per_dir = samples_epi1 + samples_epi2 + samples_epi3

    epi_signal = signal[samples_per_cal:].squeeze()

    # Use direction 0 for T2 fitting (b=0 → all directions are identical;
    # this matches qMRI_t2relax.py which also uses the first direction)
    echo1_ksp = epi_signal[:samples_epi1].reshape(
        int(seq.Ny * seq.partial_fourier_factor), seq.adc.num_samples
    )
    echo2_ksp = epi_signal[samples_epi1 : samples_epi1 + samples_epi2].reshape(
        seq.Ny, seq.adc.num_samples
    )
    echo3_ksp = epi_signal[samples_epi1 + samples_epi2 : samples_per_dir].reshape(
        seq.Ny, seq.adc.num_samples
    )

    # print(f"  Echo1 ksp: {echo1_ksp.shape}, Echo2 ksp: {echo2_ksp.shape}, Echo3 ksp: {echo3_ksp.shape}")

    # ==============================================================================
    #   Calculate k-space trajectory
    # ==============================================================================
    k_traj_adc, k_traj, _, _, t_adc = seq.seq.calculate_kspace()
    kx_norm = k_traj_adc[0] * fov / Nx
    ky_norm = k_traj_adc[1] * fov / Ny
    # print(
    #     f"  [k-traj] kx range: [{kx_norm.min():.4f}, {kx_norm.max():.4f}] "
    #     f"(expected Nyquist ≈ ±0.5, ramp overshoot OK)"
    # )
    traj = np.stack([kx_norm, ky_norm], axis=-1)

    # Split trajectory to match the three echo k-spaces of direction 0
    traj_epi1 = traj[samples_per_cal : samples_per_cal + samples_epi1]
    traj_epi2 = traj[
        samples_per_cal + samples_epi1 : samples_per_cal + samples_epi1 + samples_epi2
    ]
    traj_epi3 = traj[
        samples_per_cal
        + samples_epi1
        + samples_epi2 : samples_per_cal
        + samples_per_dir
    ]

    # =================================================================================
    #   NUFFT reconstruction for each echo
    # =================================================================================
    img_size = (Ny, Nx)
    echo_imgs = []
    for ksp, traj_echo in [
        (echo1_ksp, traj_epi1),
        (echo2_ksp, traj_epi2),
        (echo3_ksp, traj_epi3),
    ]:
        nufft_op = get_operator(
            backend_name="cufinufft" if False else "finufft",
            samples=traj_echo,
            shape=img_size,
            n_coils=1,
            density=True,
        )
        sig_t = ksp.to(torch.complex64)
        img_complex = nufft_op.adj_op(sig_t.flatten()).squeeze()
        echo_imgs.append(img_complex.cpu().numpy())

    for img, te_ms in zip(echo_imgs, [te1_ms, te2_ms, te3_ms]):
        mag_img = np.abs(img)
        all_echo_images.append(mag_img)
        all_echo_tes.append(te_ms)

        te_name = f"TE-{int(te_ms)}-{'blipdown' if BLIP_DOWN else 'blipup'}"
        nii_path = os.path.join(
            ECHO_IMAGES_DIR_PATH, f"{name.replace(f'TE1-{te1}', '')}{te_name}.nii.gz"
        )
        affine = np.array(
            [[res, 0, 0, 0], [0, res, 0, 0], [0, 0, res, 0], [0, 0, 0, 1]]
        )
        nib.save(
            nib.Nifti1Image(
                np.asarray(mag_img[:, :, np.newaxis], dtype=np.float32), affine=affine
            ),
            nii_path,
        )
    print(
        "=================================================================================="
    )
print(f"\nCollected {len(all_echo_images)} echo images across {len(TEs_TE1)} TRs.")

# %% ==============================================================================
#   Visualize echo images (rows = TRs, columns = 3 echoes)
# =================================================================================
n_trs = len(TEs_TE1)
fig, axs = plt.subplots(n_trs, 3, figsize=(9, 3 * n_trs))
fig.suptitle(
    f"Triple SE echo images — {n_trs} TRs  "
    f"(single SE equivalent: 45 TRs)\n"
    f"Columns: Echo 1 (TE1) | Echo 2 (TE2) | Echo 3 (TE3)"
)
for tr_idx in range(n_trs):
    for echo_idx in range(3):
        flat_idx = tr_idx * 3 + echo_idx
        img = np.rot90(all_echo_images[flat_idx], -1)
        axs[tr_idx, echo_idx].imshow(img, cmap="gray")
        axs[tr_idx, echo_idx].set_title(f"TE={all_echo_tes[flat_idx]:.1f} ms")
        axs[tr_idx, echo_idx].set_axis_off()
plt.tight_layout()
plt.show()

# %% ==============================================================================
#   Sort by TE and stack for T2 fitting
# =================================================================================
order = np.argsort(all_echo_tes)
images_stack = np.array(all_echo_images)[order]  # (n_echoes, Ny, Nx)
te_sorted = np.array(all_echo_tes)[order]

print(f"Images stack shape: {images_stack.shape}")
print(
    f"TE range: [{te_sorted.min():.1f}, {te_sorted.max():.1f}] ms  "
    f"({len(te_sorted)} data points)"
)

# %% ==============================================================================
#   T2 fitting
# =================================================================================
from utils_relaxometry import create_t2_map

ims = []
for method in ["nlls", "loglinear"]:
    t2_result = create_t2_map(images_stack, te_sorted, method=method)
    ims.append(t2_result[0])

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
titles = ["NLLS Fit", "Log-Linear Fit"]
for i, ax in enumerate(axs):
    im = ax.imshow(np.rot90(ims[i], -1 if i < 2 else 1), cmap="viridis")
    ax.set_title(titles[i])
    fig.colorbar(im, ax=ax, label="T2 (ms)")
plt.tight_layout()
plt.show()


# %% ==============================================================================
#   Comparison: Triple SE T2 maps vs reference
# =================================================================================
n_trs  # qMRI_t2relax.py uses np.arange(65, 290, 5) = 45 TEs

fig, axs = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    f"Triple SE T2 mapping — {n_trs} TRs ({n_trs * TR / 1000:.0f} s)  "
    f"vs single SE: {n_trs} TRs ({n_trs * TR / 1000:.0f} s)  "
    f"[{n_trs// n_trs}x fewer TRs]"
)
titles = ["NLLS Fit", "Log-Linear Fit", "Reference T2 map"]
ims2 = [*ims, T2]
# Save ims2[0] to file for later use

# Shared colour scale across all three panels, tied to the reference
# map's own (physically plausible) T2 range. Without this each panel
# auto-scales to its own data max, so panels aren't actually
# comparable — a handful of outlier voxels in one fit would silently
# rescale that panel's colours relative to the other two. The pixels
# this clips are inspected in the next cell.
vmax_shared = float(np.asarray(T2).max())

for i, ax in enumerate(axs):
    if i < 2:
        im_data = np.rot90(ims2[i], -1) / 1000
    else:
        im_data = np.rot90(ims2[i], -1)
    ims2[i] = im_data  # save for later
    im = ax.imshow(im_data, cmap="viridis", vmin=0, vmax=vmax_shared)
    ax.set_title(titles[i])
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, label="T2 (s)")
plt.tight_layout()
plt.show()

# %% ==============================================================================
#   QC: pixels clipped by the shared colour scale above
# =================================================================================
# Note this is distinct from the implausible-fit rejection already
# done inside create_t2_map (see utils_relaxometry.py): those voxels
# are zeroed at the source and don't show up here at all. What's
# shown below is the remainder — legitimate, non-zero fitted T2
# values that simply exceed the reference map's max and therefore get
# visually saturated to the top colour of vmax_shared above.
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
    np.save(
        f"simulated/brainmaps/{phantoms[PHANTOM_IDX]}-T2_MSE_{'blipdown' if BLIP_DOWN else 'blipup'}.npy", ims2[0]
    )
except Exception:
    pass
# %%
