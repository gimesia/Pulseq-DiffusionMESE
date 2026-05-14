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
from EPIDiffusionSEPulseqSeq import EPIDiffusionSEPulseqSeq
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
SEQUENCES_DIR_PATH = rf".\simulated\seq"
VOLUMES_DIR_PATH = rf".\simulated\brainmaps"
PHANTOMS_DIR_PATH = rf"C:\Users\User\OneDrive\PhD\Sumbission\ESMRMB26\Pulseq-DiffusionMESE\brainweb_phantoms"
ECHO_IMAGES_DIR_PATH = r"C:\Users\User\OneDrive\PhD\Sumbission\ESMRMB26\Pulseq-DiffusionMESE\simulation\simulated\t2_img"


# %% ==============================================================================
#   Simulation parameters
# =================================================================================
fov = 224e-3
res = 2.33333333
slice_thickness = res * 1e-3
# TEs = np.arange(65, 290, 5)  # From 70 to 410 with a step of 5
TEs = np.array(
    [
        65,
        70,
        75,
        80,
        85,
        90,
        95,
        100,
        105,
        110,
        115,
        120,
        125,
        127,
        130,
        132,
        135,
        137,
        140,
        142,
        145,
        147,
        150,
        152,
        157,
        162,
        167,
        172,
        177,
        182,
        187,
        189,
        192,
        194,
        197,
        199,
        202,
        204,
        207,
        209,
        212,
        214,
        219,
        224,
        229,
        234,
        239,
        244,
        249,
        254,
        259,
        264,
        269,
        274,
    ], dtype=int
)
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

phantom, phantom_data = phantom_loader.load_phantom(
    json_path=phantom_path,
    resolution_mm=res,
    slice_idx=None,
)
D = phantom.D
T2 = phantom.T2


# %% ================================================================================
#   Main simulation loop: Generate sequence, simulate signal, separate k-spaces, calculate trajectories, and reconstruct images
# =================================================================================
nu_kspaces = []
trajectories = []
reconstructed_images = []
for te in TEs:
    print(f"Simulating for TE={te} ms ...")
    # =================================================================================
    #   Generate sequence
    # =================================================================================
    name = f"DiffSE-TE{te}-{'blipdown' if BLIP_DOWN else 'blipup'}"
    seq = EPIDiffusionSEPulseqSeq(
        name=name,
        resolution=res,
        Nx=Nx,
        Ny=Ny,
        fov=fov,
        slice_thickness=slice_thickness,
        TE=te,
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
        blip_down=BLIP_DOWN,
        logger=logger,
        fit_epi=True,
    )

    # Generate and write the sequence to a file
    # The filename will be based on the sequence parameters
    seq.write()

    # =================================================================================
    #   Simulate sequence
    # =================================================================================
    seq0 = mr0.Sequence.import_file(rf"{SEQUENCES_DIR_PATH}\{name}.seq")
    if use_GPU:
        seq0_gpu = seq0.cuda()
        phantom_data_gpu = (
            phantom_data.cuda()
        )  # Use only one slice for GPU computation to save memory
        # print(f"Using GPU for computation. Shape: {phantom_data_gpu.shape}")
        graph = mr0.compute_graph(seq0_gpu, phantom_data_gpu, 20000, 1e-5)
        signal = mr0.execute_graph(
            graph, seq0_gpu, phantom_data_gpu, print_progress=True
        ).cpu()

        del seq0_gpu
        del phantom_data_gpu
        torch.cuda.empty_cache()
    else:
        phantom_data_cpu = (
            phantom_data.cpu()
        )  # Use only one slice for GPU computation to save memory
        # print(f"Using CPU for computation. Shape: {phantom_data_cpu.shape}")

        ## =======================================================================================================================
        graph = mr0.compute_graph(seq0, phantom_data_cpu, 2000, 1e-4)
        signal = mr0.execute_graph(graph, seq0, phantom_data_cpu, print_progress=False)
    try:
        del seq0_gpu
        del phantom_data_gpu
    except Exception:
        pass
    torch.cuda.empty_cache()

    # ==============================================================================
    # Separate K-spaces
    # =================================================================================
    samples_per_cal = int(3 * seq.adc.num_samples)
    samples_per_dir = int(seq.Ny * seq.partial_fourier_factor) * seq.adc.num_samples

    calib_signal = signal[:samples_per_cal,].reshape((3, seq.adc.num_samples))
    epi_signal = signal[samples_per_cal:,].squeeze()
    dir_signal = []
    for i in range(len(seq.b_directions)):
        start_idx = i * samples_per_dir
        end_idx = (i + 1) * samples_per_dir
        dir_signal.append(
            epi_signal[start_idx:end_idx,].reshape(
                (int(seq.Ny * seq.partial_fourier_factor), seq.adc.num_samples)
            )
        )
    dir_signal = np.array(dir_signal)

    # print(f"Calibration signal shape: {calib_signal.shape}")
    # print(f"EPI signal shape: {epi_signal.shape}")
    # print(f"Directional signal shapes: {[e.shape for e in dir_signal]}")

    nu_kspaces.append(dir_signal)

    # ==============================================================================
    #   Calculate k-space trajectory
    # =================================================================================
    k_traj_adc, k_traj, _, _, t_adc = seq.seq.calculate_kspace()

    # k_traj_adc shape: (3, n_adc_samples) → rows are kx, ky, kz
    kx = k_traj_adc[0]  # shape: (n_total_adc_samples,)
    ky = k_traj_adc[1]

    # Normalize to [-0.5, 0.5]: Cartesian Nyquist NX/(2*fov) → ±0.5
    # PyPulseq returns k in cycles/m; ramp samples may slightly exceed ±0.5 (expected)
    kx_norm = kx * fov / Nx
    ky_norm = ky * fov / Ny
    # print(
    #     f"[k-traj] kx range: [{kx_norm.min():.4f}, {kx_norm.max():.4f}] "
    #     f"(expected Nyquist ≈ ±0.5, ramp overshoot OK)"
    # )

    traj = np.stack([kx_norm, ky_norm], axis=-1)  # shape: (n_samples, 2)
    calibration_trajectory = traj[:samples_per_cal,]
    direction_trajectory = traj[samples_per_cal:,]
    direction_trajectories = []
    for i in range(len(seq.b_directions)):
        start_idx = i * samples_per_dir
        end_idx = (i + 1) * samples_per_dir
        direction_trajectories.append(direction_trajectory[start_idx:end_idx,])
    direction_trajectories = np.array(direction_trajectories)
    # print(f"Calibration trajectory shape: {calibration_trajectory.shape}")
    # print(f"Directional trajectory shapes: {[t.shape for t in direction_trajectories]}")

    trajectories.append(direction_trajectories)

    # =================================================================================
    # Non-uniform FFT reconstruction (NUFFT) - for the ramp sampling pattern in the EPI readout
    # =================================================================================

    img_size = (Ny, Nx)  # your Cartesian reconstruction grid size

    recon = []
    # Build one NUFFT operator shared across all directions (same EPI k-space trajectory)
    nufft_op = get_operator(
        backend_name="cufinufft" if False else "finufft",
        samples=direction_trajectories[i],
        shape=img_size,
        n_coils=1,
        density=True,
    )
    sig = dir_signal[i]
    sig = torch.from_numpy(sig).to(torch.complex64)  # Convert to PyTorch tensor
    # print(f"Signal shape for direction {i+1}: {sig.shape}, dtype: {sig.dtype}")
    img_complex = nufft_op.adj_op(sig.flatten())  # shape: (1, Ny, Nx) if n_coils=1
    img_complex = img_complex.squeeze().cpu().numpy()  # (Ny, Nx)

    mag_img = np.abs(img_complex)
    # print(
    #     f"Reconstructed image shape for direction {i+1}: {img_complex.shape}, dtype: {img_complex.dtype}"
    # )

    nii_path = os.path.join(ECHO_IMAGES_DIR_PATH, f"{name}.nii.gz")
    affine = np.array([[res, 0, 0, 0], [0, res, 0, 0], [0, 0, res, 0], [0, 0, 0, 1]])
    nib.save(
        nib.Nifti1Image(np.asarray(mag_img, dtype=np.float32), affine=affine), nii_path
    )

    recon.append(img_complex)  # Move to CPU and convert to NumPy
    reconstructed_images.append(recon)
    print(
        "=================================================================================="
    )
print("Sequence generation complete.")
# %%
for dir_idx, dir in enumerate(seq.b_directions):
    rows = int(len(TEs) / 5) if len(TEs) % 5 == 0 else int(len(TEs) / 5) + 1

    fig, axs = plt.subplots(rows, 5, figsize=(15, 3 * rows))
    axs = axs.flatten()
    fig.suptitle(
        f"Reconstructed images (dir {seq.b_directions[dir_idx]})"
    )  # Add a title for the entire figure
    for i, recon in enumerate(reconstructed_images):
        b0 = np.rot90(recon[dir_idx], -1)
        axs[i].imshow(np.abs(b0), cmap="gray")
        axs[i].set_title(f"TE={TEs[i]} ms")
        axs[i].set_axis_off()
    plt.tight_layout()  # Adjust layout to make room for the suptitle
    plt.show()

reconstructed_images = np.array(
    reconstructed_images
)  # shape: (n_TEs, n_directions, Ny, Nx)
print(f"Reconstructed images array shape: {reconstructed_images.shape}")
reconstructed_b0_images = reconstructed_images[:, 0, :, :]  # shape: (n_TEs, Ny, Nx)
print(f"Extracted b0 images shape: {reconstructed_b0_images.shape}")
reconstructed_b0_magnitude_images = np.abs(
    reconstructed_b0_images
)  # Take magnitude for T2 fitting

# %% ==============================================================================
#   T2 fitting
# =================================================================================
from utils_relaxometry import create_t2_map

ims = []
for i in ["nlls", "loglinear"]:
    t2_result = create_t2_map(reconstructed_b0_magnitude_images, TEs, method=i)
    ims.append(t2_result[0])
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
titles = ["NLLS Fit", "Log-Linear Fit"]

for i, ax in enumerate(axs):
    im = ax.imshow(np.rot90(ims[i], -1 if i < 2 else 1), cmap="viridis")
    ax.set_title(titles[i])
    fig.colorbar(im, ax=ax, label="T2 (ms)")

# %% ==============================================================================
#   Comparison: Single SE T2 maps vs reference
# =================================================================================
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
titles = ["NLLS Fit", "Log-Linear Fit", "Reference T2 Map"]

ims2 = [*ims, T2]
for i, ax in enumerate(axs):
    if i < 2:
        im = np.rot90(ims2[i], -1)
        im = im / 1000
    else:
        im = np.rot90(ims2[i], 1)
    ims2[i] = im  # save for later
    im = ax.imshow(im, cmap="viridis", vmax=1.6)
    ax.set_title(titles[i])
    fig.colorbar(im, ax=ax, label="T2 (s)")


# %%
try:
    np.save(
        f"simulated/vol/T2_SSE_{'blipdown' if BLIP_DOWN else 'blipup'}.npy", ims2[0]
    )
except Exception:
    pass
# %%
