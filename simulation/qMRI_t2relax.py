# %%
import os
import sys
import numpy as np

# The path to the pulseq-diffusion-mese directory.
# TODO: It is advisable to replace this with a more robust method for path management,
# such as using environment variables or a configuration file.
seq_path = r'C:\Users\User\OneDrive\PhD\Sumbission\ESMRMB26\Pulseq-DiffusionMESE\pulseq_diffusion_mese'
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
from simulation_utils import *
from mrinufft import get_operator

np.int = int
np.float = float
np.complex = complex
use_GPU = torch.cuda.is_available()

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
slice_thickness = res*1e-3
TEs = np.arange(65, 255, 5)  # From 70 to 410 with a step of 5
TR = 5000
b_value = 1000
Nx = Ny = int(fov / slice_thickness)

# %% ==============================================================================
#   Load phantom
# =================================================================================
PHANTOM_IDX = 4  # Select phantom index

NZ = 10
SLICE_IDX = 4  # Select slice index

add_tumor = True
tumor_size = (10, 10, 10)  # size of the tumor in voxels (x, y, z)


phantoms = [f for f in os.listdir(PHANTOMS_DIR_PATH) if f.endswith(".npz")]
print("Available phantoms:", phantoms)
phantom_path = os.path.join(PHANTOMS_DIR_PATH, phantoms[PHANTOM_IDX])


if os.path.isfile(phantom_path) and phantom_path.endswith(".npz"):
    phantom = mr0.VoxelGridPhantom.load(phantom_path)  # Load phantom
    print(f"Loaded phantom {os.path.split(phantom_path)[-1]}. Shape: {phantom.D.shape}")
    phantom = phantom.interpolate(Nx, Ny, NZ)  # Resize phantom, select slice
    print(f"Resized phantom. Shape: {phantom.D.shape}")

    phantom.voxel_size = torch.Tensor(
        [0.00233, 0.00233, 0.00233]
    )  # Set voxel size (in mm)

    vox = phantom.voxel_size

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

phantom.plot()  # Plot phantom
phantom_data = phantom.build()  # Build phantom with specified voxel size (in mm)

nu_kspaces = []
trajectories = []
reconstructed_images = []
for te in TEs:
    # =================================================================================
    #   Generate sequence
    # =================================================================================
    name = f"DiffSE-TE{te}"
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
        b_directions=3,
        b_0_frequency=3,
        save_dir=SEQUENCES_DIR_PATH,
        save_name=name,
        v141_compat=True,
        small_delta=0.018,
        big_DELTA=0.03,
        system_type=SystemLimitType.EXTREME,
        calibration_readout=True,
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
            epi_signal[start_idx:end_idx,].reshape((int(seq.Ny * seq.partial_fourier_factor), seq.adc.num_samples))
        )
    dir_signal = np.array(dir_signal)

    print(f"Calibration signal shape: {calib_signal.shape}")
    print(f"EPI signal shape: {epi_signal.shape}")
    print(f"Directional signal shapes: {[e.shape for e in dir_signal]}")

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
    print(
        f"[k-traj] kx range: [{kx_norm.min():.4f}, {kx_norm.max():.4f}] "
        f"(expected Nyquist ≈ ±0.5, ramp overshoot OK)"
    )

    traj = np.stack([kx_norm, ky_norm], axis=-1)  # shape: (n_samples, 2)
    calibration_trajectory = traj[:samples_per_cal,]
    direction_trajectory = traj[samples_per_cal:,]
    direction_trajectories = []
    for i in range(len(seq.b_directions)):
        start_idx = i * samples_per_dir
        end_idx = (i + 1) * samples_per_dir
        direction_trajectories.append(direction_trajectory[start_idx:end_idx,])
    direction_trajectories = np.array(direction_trajectories)
    print(f"Calibration trajectory shape: {calibration_trajectory.shape}")
    print(f"Directional trajectory shapes: {[t.shape for t in direction_trajectories]}")

    trajectories.append(direction_trajectories)


    # =================================================================================
    # Non-uniform FFT reconstruction (NUFFT) - for the ramp sampling pattern in the EPI readout
    # =================================================================================

    img_size = (Ny, Nx)  # your Cartesian reconstruction grid size

    recon = []
    for i in range(len(seq.b_directions)):
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
        print(f"Signal shape for direction {i+1}: {sig.shape}, dtype: {sig.dtype}")
        img_complex = nufft_op.adj_op(sig.flatten())  # shape: (1, Ny, Nx) if n_coils=1
        img_complex = img_complex.squeeze()  # (Ny, Nx)
        print(
            f"Reconstructed image shape for direction {i+1}: {img_complex.shape}, dtype: {img_complex.dtype}"
        )
        recon.append(img_complex.cpu().numpy())  # Move to CPU and convert to NumPy
    reconstructed_images.append(recon)

print("Sequence generation complete.")
# %%
for dir_idx, dir in enumerate(seq.b_directions):
    rows = int(len(TEs)/5) if len(TEs) % 5 == 0 else int(len(TEs)/5) + 1
    
    fig, axs = plt.subplots(rows, 5, figsize=(15, 3*rows))
    axs = axs.flatten()
    fig.suptitle(f"Reconstructed images (dir {seq.b_directions[dir_idx]})")  # Add a title for the entire figure
    for i, recon in enumerate(reconstructed_images):
        b0 = np.rot90(recon[dir_idx], -1)
        axs[i].imshow(np.abs(b0), cmap="gray")
        axs[i].set_title(f"TE={TEs[i]} ms")
        axs[i].set_axis_off()
    plt.tight_layout()  # Adjust layout to make room for the suptitle
    plt.show()

reconstructed_images = np.array(reconstructed_images)  # shape: (n_TEs, n_directions, Ny, Nx)
print(f"Reconstructed images array shape: {reconstructed_images.shape}")
reconstructed_b0_images = reconstructed_images[:, 0, :, :]  # shape: (n_TEs, Ny, Nx)
print(f"Extracted b0 images shape: {reconstructed_b0_images.shape}")
reconstructed_b0_magnitude_images = np.abs(reconstructed_b0_images)  # Take magnitude for T2 fitting

# %%
from scipy.optimize import curve_fit

def mono_exponential(te, s0, t2):
    return s0 * np.exp(-te / t2)

def create_t2_map_scipy(data, te_list):
    """
    data: (n_te, 96, 96) array
    te_list: array of echo times
    """
    n_te, ny, nx = data.shape
    t2_map = np.zeros((ny, nx))
    s0_map = np.zeros((ny, nx))

    # Threshold to ignore background noise (tweak 0.1 as needed)
    threshold = np.max(data) * 0.1

    for y in range(ny):
        for x in range(nx):
            pixel_series = data[:, y, x]
            
            # Only fit if signal is above noise floor
            if pixel_series[0] > threshold:
                try:
                    # Initial guesses are crucial for NLLS:
                    # s0_guess = first echo, t2_guess = 50ms
                    popt, _ = curve_fit(
                        mono_exponential, 
                        te_list, 
                        pixel_series, 
                        p0=[pixel_series[0], 50],
                        bounds=(0, [np.inf, 1600]) # Constrain T2 to realistic values
                    )
                    s0_map[y, x], t2_map[y, x] = popt
                except:
                    # If the fit fails (e.g. non-convergent), leave as 0
                    continue
                    
    return t2_map, s0_map


# Assuming 't2_result' is your 96x96 map
t2_result, s0_result = create_t2_map_scipy(reconstructed_b0_magnitude_images, TEs)
plt.imshow(np.rot90(t2_result, -1), cmap='viridis',)
plt.colorbar(label='T2 (ms)')
plt.title('Quantitative T2 Map')

# %%
def mono_exponential(te, s0, t2):
    return s0 * np.exp(-te / t2)
 
 
def create_t2_map_scipy(data, te_list):
    n_te, ny, nx = data.shape
    t2_map = np.zeros((ny, nx))
    s0_map = np.zeros((ny, nx))
 
    # Background mask: skip pixels whose first-echo signal is below
    # this fraction of the global maximum. NB: 0.5 is aggressive -
    # see the docstring's "Threshold" note.
    threshold = np.max(data) * 0.5
 
    # Per-pixel fit. The outer loops are over image rows/columns;
    # the inner curve_fit fits one decay curve at a time.
    for y in range(ny):
        for x in range(nx):
            pixel_series = data[:, y, x]
 
            # Reject background / low-SNR voxels using the first-echo
            # intensity as a proxy for tissue signal.
            if pixel_series[0] > threshold:
                try:
                    # p0: initial guess for [S0, T2]. Using the first
                    #   echo as S0_guess is robust because S(TE_min)
                    #   is the closest direct estimate of S0 we have.
                    # bounds: keep both parameters non-negative and
                    #   clip T2 to a physiologically plausible range.
                    popt, _ = curve_fit(
                        mono_exponential,
                        te_list,
                        pixel_series,
                        p0=[pixel_series[0], 50],
                        bounds=(0, [np.inf, 1600]),
                    )
                    s0_map[y, x], t2_map[y, x] = popt
                except Exception:
                    # Non-convergent fit - leave this voxel as 0.
                    # Consider logging or counting these for QC.
                    continue
 
    return t2_map, s0_map

# Assuming 't2_result' is your 96x96 map
t2_result, s0_result = create_t2_map_scipy(reconstructed_b0_magnitude_images, TEs)
plt.imshow(np.rot90(t2_result, -1), cmap='viridis',)
plt.colorbar(label='T2 (ms)')
plt.title('Quantitative T2 Map')

# %%
from relaxometry_utils import create_t2_map

ims = []
for i in ['nlls', 'loglinear']:
    t2_result = create_t2_map(reconstructed_b0_magnitude_images, TEs, method=i)
    ims.append(t2_result[0])

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
titles = ['NLLS Fit', 'Log-Linear Fit']
for i, ax in enumerate(axs):
    im = ax.imshow(np.rot90(ims[i], -1), cmap='viridis')
    ax.set_title(titles[i])
    fig.colorbar(im, ax=ax, label='T2 (ms)')
# %%
