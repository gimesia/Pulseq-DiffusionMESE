# IQ-BRAIN is funded by the European Union (MSCA Doctoral Network,
# December 2024–November 2028, Grant Agreement No. 101169519).
# Single SE diffusion simulation and ADC fitting

# %%
import logging
import os
import sys
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mrinufft")

logger = logging.getLogger()
logger.setLevel(logging.ERROR)  # Suppress INFO and WARNING

# The path to the pulseq-diffusion-mese directory.
# TODO: It is advisable to replace this with a more robust method for path management,
# such as using environment variables or a configuration file.
seq_path = r"C:\Users\User\OneDrive\PhD\Sumbission\ESMRMB26\Pulseq-DiffusionMESE\pulseq_diffusion_mese"
if seq_path not in sys.path:
    sys.path.append(seq_path)

# %% ================================================================================
#  Imports
# =================================================================================
import MRzeroCore as mr0
import numpy as np
import torch
import matplotlib.pyplot as plt
from pypulseq import Sequence
from EPIDiffusionSEPulseqSeq import EPIDiffusionSEPulseqSeq
from utils import SystemLimitType
from utils_simulation import *
from mrinufft import get_operator
import torch
import torch.nn.functional as F

np.int = int
np.float = float
np.complex = complex

use_GPU = torch.cuda.is_available()
B_DIRS = 3
BLIP_DOWN = True
# =================================================================================
#   Paths
# =================================================================================
SEQUENCES_DIR_PATH = rf".\simulated\seq"
VOLUMES_DIR_PATH = rf".\simulated\vol"
PHANTOMS_DIR_PATH = rf".\phantoms\brainweb"

# %% ==============================================================================
#   Simulation parameters
# =================================================================================
fov = 224e-3
res = 2.33333333
slice_thickness = res * 1e-3

# Fixed TE for ADC: choose a single TE long enough to accommodate the largest b-value.
# All b-values must use the same TE so that T2 weighting is identical across the
# series and only the diffusion-weighting differs (Stejskal-Tanner assumption).
TE = 100  # [ms] - must be feasible for the highest b-value below

# Vary b-values instead of TEs. b=0 is required as the reference (S0).
# A typical clinical DWI protocol uses b=0 and b=1000; for ADC fitting any
# 2+ b-values work, but spreading them improves the fit conditioning.
b_values = np.arange(0, 2301, 150)  # [s/mm^2]

TR = 5000
Nx = Ny = int(fov / slice_thickness)

# %% ==============================================================================
#   Load phantom
# =================================================================================
PHANTOM_IDX = 4  # Select phantom index
NZ = 10
SLICE_IDX = 4  # Select slice index
add_tumor = True
tumor_size = (20, 20, 20)  # size of the tumor in voxels (x, y, z)

phantoms = [f for f in os.listdir(PHANTOMS_DIR_PATH) if f.endswith(".npz")]
print("Available phantoms:", phantoms)
phantom_path = os.path.join(PHANTOMS_DIR_PATH, phantoms[PHANTOM_IDX])

if os.path.isfile(phantom_path) and phantom_path.endswith(".npz"):
    phantom = mr0.VoxelGridPhantom.load(phantom_path)  # Load phantom
    if add_tumor:
        # typical brain tumor ADC values are around ~1.5 * 10^-3 mm^2/s,
        # which lies between GM/WM and CSF (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3000221)
        phantom = add_tumor_to_phantom(
            phantom,
            tumor_size=tumor_size,
            tumor_location="br",
            adc_tumor_core=1.5,
            adc_tumor_border=2.5,  # 'tl', 'bl', 'tr', 'br' or (cx, cy, cz)
        )
    print(f"Loaded phantom {os.path.split(phantom_path)[-1]}. Shape: {phantom.D.shape}")
    phantom.plot()  # Plot phantom
    phantom = phantom.interpolate(Nx, Ny, NZ)  # Resize phantom, select slice
    print(f"Resized phantom. Shape: {phantom.D.shape}")
    phantom = phantom.slices([SLICE_IDX])  # Resize phantom, select slice
    print(f"Selected slice. Shape: {phantom.D.shape}")
else:
    raise FileNotFoundError(f"Error: Invalid phantom file at {phantom_path}")

# Visualize and build phantom
PD = phantom.PD
T1 = phantom.T1
T2 = phantom.T2
D = phantom.D
B0 = phantom.B0
B1 = phantom.B1
phantom_data = phantom.build()  # Build phantom with specified voxel size (in mm)

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
        name = f"DiffSE-b{int(b_value)}-TE{TE_val}"  # unique name per TE too
        # =================================================================================
        #   Generate sequence
        # =================================================================================
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
            signal = mr0.execute_graph(graph, seq0_gpu, phantom_data_gpu, print_progress=True).cpu()
        else:
            phantom_data_cpu = phantom_data.cpu()
            graph = mr0.compute_graph(seq0, phantom_data_cpu, 5000, 1e-5)
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
        # nu_kspaces.append(dir_signal)

        # ==============================================================================
        #   Calculate k-space trajectory
        # =================================================================================
        k_traj_adc, k_traj, _, _, t_adc = seq.seq.calculate_kspace()
        kx_norm = k_traj_adc[0] * fov / Nx
        ky_norm = k_traj_adc[1] * fov / Ny
        traj = np.stack([kx_norm, ky_norm], axis=-1)
        direction_trajectory = traj[samples_per_cal:]
        direction_trajectories = np.array([
            direction_trajectory[i * samples_per_dir:(i + 1) * samples_per_dir]
            for i in range(len(seq.b_directions))
        ])
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
        print(f"Reconstructed image shape for direction {i+1}: {np.array(recon).shape}")
    # echo_images_per_b is (n_echoes, n_dirs, Ny, Nx) → transpose to (n_dirs, n_echoes, Ny, Nx)
    all_echo_images.append(
        np.stack(echo_images_per_b, axis=0).transpose(1, 0, 2, 3)
    )
all_echo_images = np.array(all_echo_images)  # (n_b, n_dirs, n_echoes, Ny, Nx)
print(f"all_echo_images shape: {all_echo_images.shape}")
assert all_echo_images.shape == (len(b_values), B_DIRS, len(TE_list), Ny, Nx)
# Then combine exactly like Triple SE:
mag_images = np.abs(all_echo_images)  # (n_b, n_dirs, n_echoes, Ny, Nx)
mag_images_combined = mag_images.mean(axis=2)  # (n_b, n_dirs, Ny, Nx)
print(f"all_echo_images shape: {all_echo_images.shape}")
print(f"mag_images_combined shape: {mag_images_combined.shape}")

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
b_subset_indices = np.linspace(0, len(b_values) - 1, len(b_values)//2, dtype=int)
fig, axs = plt.subplots(B_DIRS, len(b_subset_indices), figsize=(len(b_subset_indices)*3, B_DIRS*3))

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
# Collapse the direction axis via geometric mean to obtain a rotation-invariant
# signal: (n_b, Ny, Nx).
eps = 1e-12


def compute_trace_dwi(mag):
    """Geometric mean across diffusion directions. mag: (n_b, n_dirs, Ny, Nx)."""
    return np.exp(np.mean(np.log(mag + eps), axis=1))


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
#   Log-linear vs NLLS comparison (all echoes combined)
# =================================================================================
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
for ax, adc_map, title in zip(
    axs,
    [adc_nlls, adc_ll],
    ["NLLS Fit (all echoes)", "Log-Linear Fit (all echoes)"],
):
    im = ax.imshow(np.rot90(adc_map, -1) * 1e3, cmap="viridis")
    ax.set_title(title)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, label="ADC (x10⁻³ mm²/s)")
plt.tight_layout()
plt.show()

# %% ==============================================================================
#   Reference phantom
# =================================================================================
ref = mr0.VoxelGridPhantom.load(rf"{PHANTOMS_DIR_PATH}\{phantoms[PHANTOM_IDX]}")

ref = add_tumor_to_phantom(
    ref,
    tumor_size=tumor_size,
    tumor_location="br",
    adc_tumor_core=1.5,
    adc_tumor_border=2.5,
)

max_dim = max(ref.D.shape)
ref.D = pad_to_cube(ref.D, max_dim)
ref.T2 = pad_to_cube(ref.T2, max_dim)
ref.T2dash = pad_to_cube(ref.T2dash, max_dim)
ref.T1 = pad_to_cube(ref.T1, max_dim)
ref.PD = pad_to_cube(ref.PD, max_dim)
ref.B0 = pad_to_cube(ref.B0, max_dim)
ref.B1 = pad_to_cube(ref.B1, max_dim)

ref = ref.interpolate(Nx, Ny, NZ)

ref = ref.slices([SLICE_IDX])
ref.plot()

# %% ==============================================================================
#   ADC vs reference comparison (all echoes combined)
# =================================================================================
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Single SE ADC (all echoes combined) vs Reference")
ims_data = [
    (np.rot90(adc_nlls, -1) * 1e3, "NLLS (all echoes)", "ADC (x10⁻³ mm²/s)", "viridis"),
    (
        np.rot90(adc_ll, -1) * 1e3,
        "Log-Linear (all echoes)",
        "ADC (x10⁻³ mm²/s)",
        "viridis",
    ),
    (np.rot90(ref.D, 1), "Reference D map", "D (x10⁻³ mm²/s)", "viridis"),
]
for ax, (im_data, title, label, cmap) in zip(axs, ims_data):
    im = ax.imshow(im_data, cmap=cmap)
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
    np.save(f"{VOLUMES_DIR_PATH}/ADC_single_se.npy", adc_nlls)
    print(f"Saved all maps to {VOLUMES_DIR_PATH}")
except Exception as e:
    print(f"Could not save volumes: {e}")

# %%
