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
seq_path = r"C:\Users\gimes\OneDrive\PhD\Sumbission\ESMRMB26\Pulseq-DiffusionMESE\pulseq_diffusion_mese"
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
B_DIRS = 6
BLIP_DOWN = False  # Whether to use blip-down or blip-up EPI readout (affects distortion direction)
PHANTOM_IDX = 0

# =================================================================================
#   Paths
# =================================================================================
SEQUENCES_DIR_PATH = rf".\simulated\seq"
VOLUMES_DIR_PATH = rf".\simulated\brainmaps"
PHANTOMS_DIR_PATH = rf"C:\Users\gimes\OneDrive\PhD\Sumbission\ESMRMB26\Pulseq-DiffusionMESE\brainweb_phantoms"
ECHO_IMAGES_DIR_PATH = r"C:\Users\gimes\OneDrive\PhD\Sumbission\ESMRMB26\Pulseq-DiffusionMESE\simulation\simulated\diff_img"


# %% ==============================================================================
#   Simulation parameters
# =================================================================================
fov = 224e-3
res = 2.33333333
slice_thickness = res * 1e-3

# Vary b-values instead of TEs. b=0 is required as the reference (S0).
# A typical clinical DWI protocol uses b=0 and b=1000; for ADC fitting any
# 2+ b-values work, but spreading them improves the fit conditioning.
b_values = np.arange(0, 2001, 100, dtype=int)  # [s/mm^2]
TR = 5000
Nx = Ny = int(fov / slice_thickness)
# Fixed TE for ADC: choose a single TE long enough to accommodate the largest b-value.
# All b-values must use the same TE so that T2 weighting is identical across the
# series and only the diffusion-weighting differs (Stejskal-Tanner assumption).
TE = 100  # [ms] - must be feasible for the highest b-value below


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
#   Main simulation loop: Generate sequence, simulate signal, separate k-spaces, calculate trajectories, and reconstruct images
# =================================================================================
# Containers indexed by b-value (outermost loop replaces the TEs loop)
nu_kspaces = []
trajectories = []
reconstructed_images = []

TE_list = [100]
all_echo_images = []  # will become (n_b, n_dirs, n_echoes, Ny, Nx)

for b_value in b_values:
    echo_images_per_b = []  # will become (n_echoes, n_dirs, Ny, Nx)
    for TE_val in TE_list:
        print(f"Simulating sequence | b={b_value} s/mm² | TE={TE_val} ms")
        # =================================================================================
        #   Generate sequence
        # =================================================================================
        name = f"DiffSE-b{int(b_value)}-TE{TE_val}"  # unique name per TE too
        seq = EPIDiffusionSEPulseqSeq(
            name=name,
            resolution=res,
            Nx=Nx,
            Ny=Ny,
            fov=fov,
            slice_thickness=slice_thickness,
            TE=TE_val,  # <-- was incorrectly `TE`
            TR=TR,
            b_value=b_value,
            b_directions=B_DIRS,
            b_0_frequency=0,
            save_dir=SEQUENCES_DIR_PATH,
            save_name=name,
            v141_compat=True,
            small_delta=0.018,
            big_DELTA=0.03,
            system_type=SystemLimitType.EXTRASAFE,
            calibration_readout=True,
            blip_down=BLIP_DOWN,
            logger=logger,
        )
        seq.write()

        # =================================================================================
        #   Simulate sequence
        # =================================================================================
        seq0 = mr0.Sequence.import_file(rf"{SEQUENCES_DIR_PATH}\{name}.seq")

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
        epi_signal = signal[samples_per_cal:].squeeze()
        dir_signal = []
        for i in range(len(seq.b_directions)):
            start_idx = i * samples_per_dir
            end_idx = (i + 1) * samples_per_dir
            dir_signal.append(
                epi_signal[start_idx:end_idx].reshape(
                    int(seq.Ny * seq.partial_fourier_factor), seq.adc.num_samples
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
        kx_norm = k_traj_adc[0] * fov / Nx
        ky_norm = k_traj_adc[1] * fov / Ny
        traj = np.stack([kx_norm, ky_norm], axis=-1)
        direction_trajectory = traj[samples_per_cal:]
        direction_trajectories = np.array(
            [
                direction_trajectory[i * samples_per_dir : (i + 1) * samples_per_dir]
                for i in range(len(seq.b_directions))
            ]
        )
        # print(f"Calibration trajectory shape: {calibration_trajectory.shape}")
        # print(f"Directional trajectory shapes: {[t.shape for t in direction_trajectories]}")
        trajectories.append(direction_trajectories)

        # =================================================================================
        # Non-uniform FFT reconstruction (NUFFT) - for the ramp sampling pattern in the EPI readout
        # =================================================================================
        img_size = (Ny, Nx)
        recon = []  # (n_dirs, Ny, Nx) for this (b, TE)
        for i in range(len(seq.b_directions)):
            nufft_op = get_operator(
                backend_name="finufft",
                samples=direction_trajectories[i],
                shape=img_size,
                n_coils=1,
                density=True,
            )
            sig = torch.from_numpy(dir_signal[i]).to(torch.complex64)
            img_complex = nufft_op.adj_op(sig.flatten()).squeeze().cpu().numpy()
            recon.append(img_complex)
            print(f"Reconstructed image shape for direction {i+1}: {img_complex.shape}")

        echo_images_per_b.append(np.stack(recon, axis=0))  # (n_dirs, Ny, Nx)

        affine = np.array(
            [[res, 0, 0, 0], [0, res, 0, 0], [0, 0, res, 0], [0, 0, 0, 1]]
        )
        for d, img_cx in enumerate(recon):
            mag_img = np.abs(img_cx)
            nii_name = (
                f"DiffSE-b{int(b_value)}-dir{d}-TE{int(TE_val)}"
                f"-{'blipdown' if BLIP_DOWN else 'blipup'}.nii.gz"
            )
            nib.save(
                nib.Nifti1Image(
                    np.asarray(mag_img[:, :, np.newaxis], dtype=np.float32), affine
                ),
                os.path.join(ECHO_IMAGES_DIR_PATH, nii_name),
            )

    # echo_images_per_b is (n_echoes, n_dirs, Ny, Nx) → transpose to (n_dirs, n_echoes, Ny, Nx)
    all_echo_images.append(np.stack(echo_images_per_b, axis=0).transpose(1, 0, 2, 3))
    print(
        "=================================================================================="
    )  
all_echo_images = np.array(all_echo_images)  # (n_b, n_dirs, n_echoes, Ny, Nx)

assert all_echo_images.shape == (len(b_values), B_DIRS, len(TE_list), Ny, Nx)

# Then combine exactly like Triple SE:
mag_images = np.abs(all_echo_images)  # (n_b, n_dirs, n_echoes, Ny, Nx)
mag_images_combined = mag_images.mean(axis=2)  # (n_b, n_dirs, Ny, Nx)

print("Sequence generation complete.")

# %% ==============================================================================
#   Post-loop: assemble arrays
# =================================================================================
# all_echo_images already assembled in loop: (n_b, n_dirs, n_echoes, Ny, Nx)
print(f"all_echo_images shape: {all_echo_images.shape}")
assert all_echo_images.shape == (len(b_values), B_DIRS, len(TE_list), Ny, Nx)

mag_images = np.abs(all_echo_images)  # (n_b, n_dirs, n_echoes, Ny, Nx)
mag_echo1 = mag_images[:, :, 0, :, :]  # (n_b, n_dirs, Ny, Nx)
# mag_echo2 = mag_images[:, :, 1, :, :]
# mag_echo3 = mag_images[:, :, 2, :, :]
mag_images_combined = mag_images.mean(axis=2)  # (n_b, n_dirs, Ny, Nx)

# %% ==============================================================================
#   Visualize reconstructed images — rows: TEs, columns: subset of b-values
# =================================================================================
b_subset_indices = np.linspace(0, len(b_values) - 1, len(b_values) // 2, dtype=int)
fig, axs = plt.subplots(
    B_DIRS, len(b_subset_indices), figsize=(len(b_subset_indices) * 3, B_DIRS * 3)
)

for i in range(B_DIRS):
    for j, b_idx in enumerate(b_subset_indices):
        im = axs[i, j].imshow(
            np.rot90(mag_images_combined[b_idx, i], -1),
            cmap="gray",
        )
        if i == 0:
            axs[i, j].set_title(f"b={b_values[b_idx]} s/mm²")

        axs[i, j].set_axis_off()
    axs[i, 0].set_ylabel(f"Direction {i+1}", rotation=0, labelpad=40, fontsize=12)
fig.suptitle("Reconstructed magnitude images (all echoes combined)", fontsize=16)
plt.tight_layout()
plt.show()

# %% ==============================================================================
#   Trace-DWI (geometric mean across directions) — combined across echoes
# =================================================================================
def compute_trace_dwi(mag):
    """Geometric mean across diffusion directions. mag: (n_b, n_dirs, Ny, Nx).

    Floor is data-scaled (not 1e-12) so that a single noise-floor voxel in
    one direction at high b doesn't drag the geometric mean to ~0 and create
    a fake fast-decay (= falsely high ADC).
    """
    eps_local = 1e-3 * float(mag[0].max())  # ~1e-3 of the b=0 peak
    return np.exp(np.mean(np.log(np.maximum(mag, eps_local)), axis=1))


trace_dwi = compute_trace_dwi(mag_images_combined)  # (n_b, Ny, Nx)
print(f"Trace DWI shape (all echoes combined): {trace_dwi.shape}")
# %% ==============================================================================
#   ADC map — fitted from all echoes combined
# =================================================================================
from utils_diffusion import create_adc_map  # noqa: E402

adc_nlls, _ = create_adc_map(trace_dwi, b_values, method="nlls")
adc_ll, _ = create_adc_map(trace_dwi, b_values, method="loglinear")
mask = adc_nlls > 0
print(
    f"ADC combined (NLLS): range [{adc_nlls.min()*1e3:.3f}, {adc_nlls.max()*1e3:.3f}] "
    f"x10⁻³ mm²/s, median (brain) = {np.median(adc_nlls[mask])*1e3:.3f}"
)

# %% ==============================================================================
#   ADC vs reference comparison (all echoes combined)
# =================================================================================
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Single SE ADC (all echoes combined) vs Reference")

adc_nlls= np.rot90(adc_nlls, -1) * 1e3
adc_ll = np.rot90(adc_ll, -1) * 1e3
adc_ref = np.rot90(D, 1)

ims_data = [
    (adc_nlls, "NLLS (all echoes)", "ADC (x10⁻³ mm²/s)", "viridis"),
    (
        adc_ll,
        "Log-Linear (all echoes)",
        "ADC (x10⁻³ mm²/s)",
        "viridis",
    ),
    (adc_ref, "Reference D map", "D (x10⁻³ mm²/s)", "viridis"),
]
for ax, (im_data, title, label, cmap) in zip(axs, ims_data):
    im = ax.imshow(im_data, cmap=cmap, vmax=3.1)  # ADC values are typically in the range of 0-3 x10⁻³ mm²/s
    ax.set_title(title)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, label=label)
plt.tight_layout()
plt.show()

# %% ==============================================================================
#   Diffusion tensor fit → FA and MD (all echoes combined)
# =================================================================================
from utils_diffusion import create_dti_maps  # noqa: E402

fa_map, md_map, eigvals_map, dti_s0_map = create_dti_maps(
    mag_images_combined,  # (n_b, n_dirs, Ny, Nx)
    b_values,
    seq.b_directions,  # (n_dirs, 3) unit vectors from the last sequence iteration
)
print(
    f"FA range: [{fa_map.min():.3f}, {fa_map.max():.3f}]; "
    f"MD range: [{md_map.min()*1e3:.3f}, {md_map.max()*1e3:.3f}] *10⁻³ mm²/s"
)

# %% ==============================================================================
#   DTI maps
# =================================================================================
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
im = axs[0].imshow(
    np.rot90(md_map, -1) * 1e3,
    cmap="viridis",
)
axs[0].set_title("Mean Diffusivity (MD)")
axs[0].set_axis_off()
fig.colorbar(im, ax=axs[0], label="MD (*10⁻³ mm²/s)")

im = axs[1].imshow(
    np.rot90(fa_map, -1),
    cmap="inferno",
    vmax=0.1
)
axs[1].set_title("Fractional Anisotropy (FA)")
axs[1].set_axis_off()
fig.colorbar(im, ax=axs[1], label="FA")
fig.suptitle("DTI maps from Single SE (all echoes combined)")
plt.tight_layout()
plt.show()

# %% ==============================================================================
#   Save outputs
# =================================================================================
try:
    np.save(f"{VOLUMES_DIR_PATH}/ADC_SSE{'blipdown' if BLIP_DOWN else 'blipup'}.npy", adc_nlls)
    print(f"Saved all maps to {VOLUMES_DIR_PATH}")
except Exception as e:
    print(f"Could not save volumes: {e}")

# %%
