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
from utils_simulation import *
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
slice_thickness = res * 1e-3

# Fixed TE for ADC: choose a single TE long enough to accommodate the largest b-value.
# All b-values must use the same TE so that T2 weighting is identical across the
# series and only the diffusion-weighting differs (Stejskal-Tanner assumption).
TE = 100  # [ms] - must be feasible for the highest b-value below

# Vary b-values instead of TEs. b=0 is required as the reference (S0).
# A typical clinical DWI protocol uses b=0 and b=1000; for ADC fitting any
# 2+ b-values work, but spreading them improves the fit conditioning.
b_values = np.arange(100, 2101, 250)  # [s/mm^2]

TR = 5000
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

# Containers indexed by b-value (outermost loop replaces the TEs loop)
nu_kspaces = []
trajectories = []
reconstructed_images = []

for b_value in b_values:
    # =================================================================================
    #   Generate sequence
    # =================================================================================
    name = f"DiffSE-b{int(b_value)}"
    seq = EPIDiffusionSEPulseqSeq(
        name=name,
        resolution=res,
        Nx=Nx,
        Ny=Ny,
        fov=fov,
        slice_thickness=slice_thickness,
        TE=TE,                # fixed across the series
        TR=TR,
        b_value=b_value,      # the swept variable
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
    print(f"Calibration signal shape: {calib_signal.shape}")
    print(f"EPI signal shape: {epi_signal.shape}")
    print(f"Directional signal shapes: {[e.shape for e in dir_signal]}")
    nu_kspaces.append(dir_signal)

    # ==============================================================================
    #   Calculate k-space trajectory
    # =================================================================================
    k_traj_adc, k_traj, _, _, t_adc = seq.seq.calculate_kspace()
    kx = k_traj_adc[0]
    ky = k_traj_adc[1]
    # Normalize to [-0.5, 0.5]: Cartesian Nyquist NX/(2*fov) → ±0.5
    kx_norm = kx * fov / Nx
    ky_norm = ky * fov / Ny
    print(
        f"[k-traj] kx range: [{kx_norm.min():.4f}, {kx_norm.max():.4f}] "
        f"(expected Nyquist ≈ ±0.5, ramp overshoot OK)"
    )
    traj = np.stack([kx_norm, ky_norm], axis=-1)
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
    img_size = (Ny, Nx)
    recon = []
    for i in range(len(seq.b_directions)):
        nufft_op = get_operator(
            backend_name="cufinufft" if False else "finufft",
            samples=direction_trajectories[i],
            shape=img_size,
            n_coils=1,
            density=True,
        )
        sig = dir_signal[i]
        sig = torch.from_numpy(sig).to(torch.complex64)
        print(f"Signal shape for direction {i+1}: {sig.shape}, dtype: {sig.dtype}")
        img_complex = nufft_op.adj_op(sig.flatten())
        img_complex = img_complex.squeeze()
        print(
            f"Reconstructed image shape for direction {i+1}: "
            f"{img_complex.shape}, dtype: {img_complex.dtype}"
        )
        recon.append(img_complex.cpu().numpy())
    reconstructed_images.append(recon)

print("Sequence generation complete.")

# %% ==============================================================================
#   Visualize reconstructed images per direction across b-values
# =================================================================================
for dir_idx, dir in enumerate(seq.b_directions):
    cols = min(5, len(b_values))
    rows = int(np.ceil(len(b_values) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axs = np.atleast_1d(axs).flatten()
    fig.suptitle(f"Reconstructed images (dir {seq.b_directions[dir_idx]})")
    for i, recon in enumerate(reconstructed_images):
        img = np.rot90(recon[dir_idx], -1)
        axs[i].imshow(np.abs(img), cmap="gray")
        axs[i].set_title(f"b={b_values[i]} s/mm²")
        axs[i].set_axis_off()
    for j in range(len(b_values), len(axs)):
        axs[j].set_axis_off()
    plt.tight_layout()
    plt.show()

reconstructed_images = np.array(reconstructed_images)  # shape: (n_b, n_directions, Ny, Nx)
print(f"Reconstructed images array shape: {reconstructed_images.shape}")

# Magnitude images for ADC fitting (n_b, n_directions, Ny, Nx)
reconstructed_magnitude_images = np.abs(reconstructed_images)

# Trace-DWI: geometric mean across directions per b-value gives a rotation-invariant
# signal that fits the isotropic Stejskal-Tanner model: S(b) = S0 * exp(-b * ADC).
# The geometric mean is the standard choice (see DTI literature) because for a
# diagonal-dominant tensor with b-values applied along orthogonal axes,
# (S_x * S_y * S_z)^(1/3) ∝ exp(-b * (Dxx+Dyy+Dzz)/3) = exp(-b * MD).
# Using arithmetic mean would bias the fit toward the larger eigenvalue.
eps = 1e-12
log_dir = np.log(reconstructed_magnitude_images + eps)
trace_dwi = np.exp(np.mean(log_dir, axis=1))  # shape: (n_b, Ny, Nx)
print(f"Trace DWI array shape: {trace_dwi.shape}")

# %% ==============================================================================
#   ADC fitting (per-pixel NLLS, Stejskal-Tanner mono-exponential)
# =================================================================================
from scipy.optimize import curve_fit


def stejskal_tanner(b, s0, adc):
    """S(b) = S0 * exp(-b * ADC)

    b   : b-value [s/mm^2]
    s0  : signal at b=0 (proton-density / T2 weighted intensity at fixed TE)
    adc : apparent diffusion coefficient [mm^2/s]
    """
    return s0 * np.exp(-b * adc)


def create_adc_map_scipy(data, b_list):
    """
    data   : (n_b, ny, nx) magnitude trace-DWI stack
    b_list : array of b-values [s/mm^2], same order as the first axis of `data`

    Returns
    -------
    adc_map : (ny, nx) ADC in mm^2/s
    s0_map  : (ny, nx) fitted S0
    """
    n_b, ny, nx = data.shape
    adc_map = np.zeros((ny, nx))
    s0_map = np.zeros((ny, nx))

    # Background mask: use the b=0 image (lowest b, brightest) as the SNR proxy.
    # This is more meaningful for diffusion than using "first echo" because
    # signal drop with b is the quantity we are trying to fit.
    if 0 in list(b_list):
        b0_idx = list(b_list).index(0)
    else:
        b0_idx = int(np.argmin(b_list))
    threshold = np.max(data[b0_idx]) * 0.1

    b_arr = np.asarray(b_list, dtype=float)

    for y in range(ny):
        for x in range(nx):
            pixel_series = data[:, y, x]

            # Reject background / low-SNR voxels using the b=0 intensity.
            if pixel_series[b0_idx] > threshold:
                try:
                    # p0: initial guess for [S0, ADC].
                    #   S0_guess: the b=0 intensity is the most direct estimate.
                    #   ADC_guess: 1e-3 mm^2/s ≈ typical brain parenchyma value.
                    # bounds: keep both non-negative and clip ADC to a
                    #   physiologically plausible upper limit (free water at 37 °C
                    #   is ~3e-3 mm^2/s; CSF ~3.0e-3, so 5e-3 is a safe ceiling).
                    popt, _ = curve_fit(
                        stejskal_tanner,
                        b_arr,
                        pixel_series,
                        p0=[pixel_series[b0_idx], 1e-3],
                        bounds=(0, [np.inf, 5e-3]),
                    )
                    s0_map[y, x], adc_map[y, x] = popt
                except Exception:
                    # Non-convergent fit - leave this voxel as 0.
                    continue

    return adc_map, s0_map


adc_result, s0_result = create_adc_map_scipy(trace_dwi, b_values)

plt.figure()
# Display in 1e-3 mm^2/s units (the standard clinical convention) and clip to
# the physiological window so the colorbar is informative.
plt.imshow(np.rot90(adc_result, -1) * 1e3, cmap="viridis", vmin=0, vmax=3.5)
plt.colorbar(label="ADC (x10⁻³ mm²/s)")
plt.title("NLLS Quantitative ADC Map")
plt.show()

# %% ==============================================================================
#   ADC fitting helper - same shape as the second T2 cell, with QC notes
# =================================================================================
def stejskal_tanner(b, s0, adc):
    return s0 * np.exp(-b * adc)


def create_adc_map_scipy(data, b_list):
    n_b, ny, nx = data.shape
    adc_map = np.zeros((ny, nx))
    s0_map = np.zeros((ny, nx))

    # Background mask: skip pixels whose b=0 signal is below this fraction of
    # the b=0 maximum. NB: 0.5 is aggressive - lower it for noisier data.
    if 0 in list(b_list):
        b0_idx = list(b_list).index(0)
    else:
        b0_idx = int(np.argmin(b_list))
    threshold = np.max(data[b0_idx]) * 0.5

    b_arr = np.asarray(b_list, dtype=float)

    # Per-pixel fit. The outer loops are over image rows/columns;
    # the inner curve_fit fits one diffusion decay curve at a time.
    for y in range(ny):
        for x in range(nx):
            pixel_series = data[:, y, x]

            # Reject background / low-SNR voxels using the b=0 intensity as a
            # proxy for tissue signal.
            if pixel_series[b0_idx] > threshold:
                try:
                    # p0: initial guess for [S0, ADC]. Using the b=0 intensity
                    #   as S0_guess is robust because S(b=0) is, by definition,
                    #   the closest direct estimate of S0 we have.
                    # bounds: keep both parameters non-negative and clip ADC
                    #   to a physiologically plausible range (≤ 5e-3 mm^2/s).
                    popt, _ = curve_fit(
                        stejskal_tanner,
                        b_arr,
                        pixel_series,
                        p0=[pixel_series[b0_idx], 1e-3],
                        bounds=(0, [np.inf, 5e-3]),
                    )
                    s0_map[y, x], adc_map[y, x] = popt
                except Exception:
                    # Non-convergent fit - leave this voxel as 0.
                    # Consider logging or counting these for QC.
                    continue

    return adc_map, s0_map


adc_result, s0_result = create_adc_map_scipy(trace_dwi, b_values)
plt.figure()
plt.imshow(np.rot90(adc_result, -1) * 1e3, cmap="viridis", vmin=0, vmax=3.5)
plt.colorbar(label="ADC (x10⁻³ mm²/s)")
plt.title("Quantitative ADC Map")
plt.show()

# %% ==============================================================================
#   Library-style call: NLLS vs log-linear, paralleling the T2 example.
#
#   `create_adc_map` is the diffusion analogue of `create_t2_map` and is
#   expected to live in a `diffusion_utils` (or similar) module. The two
#   methods are:
#       - 'nlls'      : non-linear least squares on S(b) = S0 * exp(-b*ADC).
#                       Bias-resistant when noise is roughly Gaussian.
#       - 'loglinear' : linear regression on ln(S) vs b. Fast and closed-form,
#                       but the log transform makes the fit heteroscedastic
#                       (high-b, low-SNR points dominate the residual), so
#                       it usually overestimates ADC at low SNR.
# =================================================================================
from utils_diffusion import create_adc_map  # noqa: E402

ims = []
for method in ["nlls", "loglinear"]:
    adc_result = create_adc_map(trace_dwi, b_values, method=method)
    ims.append(adc_result[0])

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
titles = ["NLLS Fit", "Log-Linear Fit"]
for i, ax in enumerate(axs):
    im = ax.imshow(np.rot90(ims[i], -1) * 1e3, cmap="viridis", vmin=0, vmax=3.5)
    ax.set_title(titles[i])
    fig.colorbar(im, ax=ax, label="ADC (x10⁻³ mm²/s)")
plt.tight_layout()
plt.show()

# %%