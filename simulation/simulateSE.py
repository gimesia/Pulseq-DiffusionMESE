"""Bloch-equation simulation of the diffusion-(ME)-SE sequence.

This script simulates the magnetisation evolution through the (ME)-SE echo train
and plots the resulting echo amplitudes as a function of echo number.
"""
# %%
import os
import ast
import re

import MRzeroCore as mr0
import numpy as np
import torch
import matplotlib.pyplot as plt

from pypulseq import Sequence

from simulation_utils import *

np.int = int
np.float = float
np.complex = complex

# =================================================================================
#   Paths
# =================================================================================
SEQUENCES_DIR_PATH = rf"..\pulseq_diffusion_mese\seq_files"
PHANTOMS_DIR_PATH = rf".\phantoms\brainweb"

# %% ==============================================================================
#   Simulation parameters
# =================================================================================
NX = NY = 96
MULTI_ECHO = False  # Whether the sequence is multi-echo (MESE) or single-echo (SE)

use_GPU = torch.cuda.is_available()

# %% ==============================================================================
#   Phantom parameters
# =================================================================================
PHANTOM_IDX =  4  # Select phantom index

NZ = 10
SLICE_IDX = 4  # Select slice index

add_tumor = True
tumor_size = (10, 10, 10)  # size of the tumor in voxels (x, y, z)


# %% ==============================================================================
#   Load phantom
# =================================================================================
phantoms = [f for f in os.listdir(PHANTOMS_DIR_PATH) if f.endswith(".npz")]
print("Available phantoms:", phantoms)
phantom_path = os.path.join(PHANTOMS_DIR_PATH, phantoms[PHANTOM_IDX])


if os.path.isfile(phantom_path) and phantom_path.endswith(".npz"):
    ## =======================================================================================================================
    phantom = mr0.VoxelGridPhantom.load(phantom_path)  # Load phantom
    print(f"Loaded phantom {os.path.split(phantom_path)[-1]}. Shape: {phantom.D.shape}")
    phantom = phantom.interpolate(NX, NY, NZ)  # Resize phantom, select slice
    print(f"Resized phantom. Shape: {phantom.D.shape}")
    
    phantom.voxel_size = torch.Tensor([0.00233, 0.00233, 0.00233])  # Set voxel size (in mm)
    
    vox = phantom.voxel_size

    if add_tumor:
        # typical brain tumor ADC values are around ~1.5 * 10^-3 mm^2/s,
        # which lies between GM/WM and CSF (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3000221)
        phantom = add_tumor_to_phantom(
            phantom, tumor_size=tumor_size, tumor_location="br", adc_tumor_core=1.5, adc_tumor_border=2.5  # 'tl', 'bl', 'tr', 'br' or (cx, cy, cz)
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

# %% ==============================================================================
#   Load sequence
# =================================================================================
SEQUENCE_IDX = 1  # Select sequence index

seq_files = [f for f in os.listdir(SEQUENCES_DIR_PATH) if f.endswith(".seq")]
sequences = [f for f in os.listdir(SEQUENCES_DIR_PATH) if f.endswith(".seq") and 'v14' in f]  # Filter for sequences containing 'v14' in the filename
if MULTI_ECHO:
    sequences = [f for f in sequences if "DiffSE" not in f]
else:
    sequences = [f for f in sequences if "DiffMESE" not in f]



print("Available sequences:", sequences)

sequence_path = os.path.join(SEQUENCES_DIR_PATH, sequences[SEQUENCE_IDX])

# sequence_path = r"C:\Users\User\OneDrive\PhD\_CODE\IQ-BRAIN_DC3_research\src\sequences\files\Diffusion-SE_v14_safe_fov224mm_96x96x1_TR5000ms_dw4.0us_TE121ms_b1000_dirs1_b0s1_rf-spoiler_delta20.00ms_DELTA40.00ms.seq"
seq0 = mr0.Sequence.import_file(sequence_path)
seq = Sequence()
seq.read(sequence_path)
print(f"Loaded sequence {os.path.split(sequence_path)[-1]}")  # with {len(seq0.blocks)} blocks.")


TR = seq.definitions.get('TR')
AdcNumSamples = int(seq.definitions.get('AdcNumSamples'))
N_navigator_lines = int(seq.definitions.get('NNavigatorLines'))
PFF = seq.definitions.get('PartialFourierFactor')
BVAL = seq.definitions.get('bValue')

NY_acq = int(NY * PFF)
fov = float(seq.definitions.get('FOV')[0])   # in-plane FOV in metres


N_directions_raw = seq.definitions.get('DiffusionDirections')
if isinstance(N_directions_raw, str):
    # Example input: "[0, 0, 0] [1, 0, 0] [0, 0, 0] [0, 1, 0] [0, 0, 0] [0, 0, 1]"
    groups = re.findall(r"\[[^\[\]]*\]", N_directions_raw)
    directions = [list(ast.literal_eval(g)) for g in groups] if groups else []
    N_directions = len(directions)
elif isinstance(N_directions_raw, (list, tuple, np.ndarray)):
    directions = [list(x) for x in N_directions_raw]
    N_directions = len(directions)
else:
    directions = []
    N_directions = 0

# %% ==============================================================================
#   Simulate sequence
# =================================================================================
if use_GPU:
    seq0_gpu = seq0.cuda()
    phantom_data_gpu = phantom_data.cuda()  # Use only one slice for GPU computation to save memory
    # print(f"Using GPU for computation. Shape: {phantom_data_gpu.shape}")
    graph = mr0.compute_graph(seq0_gpu, phantom_data_gpu, 20000, 1e-5)
    signal = mr0.execute_graph(graph, seq0_gpu, phantom_data_gpu, print_progress=True).cpu()
    

    del seq0_gpu
    del phantom_data_gpu
    torch.cuda.empty_cache()
else:
    phantom_data_cpu = phantom_data.cpu()  # Use only one slice for GPU computation to save memory
    # print(f"Using CPU for computation. Shape: {phantom_data_cpu.shape}")
    

    ## =======================================================================================================================
    graph = mr0.compute_graph(seq0, phantom_data_cpu, 2000, 1e-4)
    signal = mr0.execute_graph(graph, seq0, phantom_data_cpu, print_progress=False)

try:
    del seq0_gpu
    del phantom_data_gpu
except:
    pass
torch.cuda.empty_cache()


print(f"Signal shape: {signal.shape}")
print(f"Signal dtype: {signal.dtype}")

TR_idx = 1

seq.plot(plot_now=False, time_range=(TR * TR_idx, (TR * TR_idx) + 0.25),)
mr0.util.insert_signal_plot(seq=seq, signal=signal.numpy())
plt.show()

# %% ==============================================================================
# Separate K-spaces
# =================================================================================
if not MULTI_ECHO:
    samples_per_cal = int(N_navigator_lines * AdcNumSamples)
    samples_per_dir = int(NY_acq * AdcNumSamples)
    
    calib_signal = signal[:samples_per_cal,].reshape((N_navigator_lines, AdcNumSamples))
    epi_signal = signal[samples_per_cal:,].squeeze()
    dir_signal = []
    for i in range(N_directions):
        start_idx = i * samples_per_dir
        end_idx = (i + 1) * samples_per_dir
        dir_signal.append(epi_signal[start_idx:end_idx,].reshape((NY_acq, AdcNumSamples)))
    dir_signal = np.array(dir_signal)
    
    print(f"Calibration signal shape: {calib_signal.shape}")
    print(f"EPI signal shape: {epi_signal.shape}")
    print(f"Directional signal shapes: {[e.shape for e in dir_signal]}")



# %% ==============================================================================
# Ghost correction using calibration lines
# =================================================================================
calib_np = calib_signal.numpy() if hasattr(calib_signal, "numpy") else np.asarray(calib_signal)
for i in range(N_directions):
    dir_signal[i] = epi_ghost_correction(calib_np, dir_signal[i])



# %% ==============================================================================
#   FFT reconstruction
# =================================================================================
for i in range(N_directions):
    mag, image = fft_reconstruct_image(dir_signal[i], use_gpu=use_GPU)
    plt.figure()
    plt.imshow(np.abs(image), cmap='gray')
    plt.title(f"Reconstructed image for direction {i+1}")
    plt.axis('off')


# %% =============================================================================
# Non-uniform FFT reconstruction (NUFFT) - for the ramp sampling pattern in the EPI readout
# =================================================================================\
k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

# k_traj_adc shape: (3, n_adc_samples) → rows are kx, ky, kz
kx = k_traj_adc[0]  # shape: (n_total_adc_samples,)
ky = k_traj_adc[1]

# Normalize to [-0.5, 0.5]: Cartesian Nyquist NX/(2*fov) → ±0.5
# PyPulseq returns k in cycles/m; ramp samples may slightly exceed ±0.5 (expected)
kx_norm = kx * fov / NX
ky_norm = ky * fov / NY
print(f"[k-traj] kx range: [{kx_norm.min():.4f}, {kx_norm.max():.4f}] "
      f"(expected Nyquist ≈ ±0.5, ramp overshoot OK)")

traj = np.stack([kx_norm, ky_norm], axis=-1)  # shape: (n_samples, 2)

# %%
calibration_trajectory = traj[:samples_per_cal,]
direction_trajectory = traj[samples_per_cal:,]
direction_trajectories = []
for i in range(N_directions):
    start_idx = i * samples_per_dir
    end_idx = (i + 1) * samples_per_dir
    direction_trajectories.append(direction_trajectory[start_idx:end_idx,])
direction_trajectories = np.array(direction_trajectories)
print(f"Calibration trajectory shape: {calibration_trajectory.shape}")
print(f"Directional trajectory shapes: {[t.shape for t in direction_trajectories]}")

# calculate_kspace() integrates gradients cumulatively across the whole sequence.
# Spoiler gradients between TRs cause each direction_trajectories[i] to start at a
# different k-space offset. All EPI readouts share the same gradient waveform, so
# subtract the per-TR DC offset to align all trajectories to a common origin.
traj_ref = direction_trajectories[0]
aligned_trajectories = np.array([
    t - t[0] + traj_ref[0] for t in direction_trajectories
])

from mrinufft import get_operator
img_size = (NY, NX)  # your Cartesian reconstruction grid size

# Build one NUFFT operator shared across all directions (same EPI k-space trajectory)
nufft_op = get_operator(
    backend_name="cufinufft" if False else "finufft",
    samples=aligned_trajectories[0],
    shape=img_size,
    n_coils=1,
    density=True,
)

for i in range(N_directions):
    sig = dir_signal[i]
    sig = torch.from_numpy(sig).to(torch.complex64)  # Convert to PyTorch tensor
    print(f"Signal shape for direction {i+1}: {sig.shape}, dtype: {sig.dtype}")
    img_complex = nufft_op.adj_op(sig.flatten())  # shape: (1, Ny, Nx) if n_coils=1
    img_complex = img_complex.squeeze()    # (Ny, Nx)
    print(f"Reconstructed image shape for direction {i+1}: {img_complex.shape}, dtype: {img_complex.dtype}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    plt.suptitle(f"B {BVAL} s/mm², direction {directions[i]}")

    # NUFFT reconstruction
    ax[0].imshow(torch.abs(img_complex).cpu().rot90(-1).numpy(), cmap="gray")
    ax[0].set_title("NUFFT recon |img|")
    ax[0].axis("off")

    # Phantom (PD map of selected slice)
    phantom_img = T2.squeeze().cpu().rot90(1).numpy()
    ax[1].imshow(phantom_img, cmap="gray")
    ax[1].set_title("Phantom (T2)")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()

# %%
