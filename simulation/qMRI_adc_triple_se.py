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
B_DIRS = 6
BLIP_DOWN = False
PHANTOM_IDX = 0

# =================================================================================
#   Paths
# =================================================================================
SEQUENCES_DIR_PATH = rf".\simulated\seq"
VOLUMES_DIR_PATH = rf".\simulated\brainmaps"
PHANTOMS_DIR_PATH = rf"C:\Users\User\OneDrive\PhD\Sumbission\ESMRMB26\Pulseq-DiffusionMESE\brainweb_phantoms"
ECHO_IMAGES_DIR_PATH = r"C:\Users\User\OneDrive\PhD\Sumbission\ESMRMB26\Pulseq-DiffusionMESE\simulation\simulated\diff_img"

# %% ==============================================================================
#   Simulation parameters
# =================================================================================
fov = 224e-3
res = 2.33333333
slice_thickness = res * 1e-3

# Fixed TE1 for the triple SE: long enough to accommodate the largest b-value gradient
# lobes. TE2 and TE3 are auto-computed by EPIDiffusionTripleSEPulseqSeq.
# All b-values use the same TE1 so that T2 weighting of echo 1 is identical across
# the b-value series (Stejskal-Tanner assumption), matching qMRI_adc.py.
TE1 = 100  # [ms]

# Vary b-values instead of TEs. Matches qMRI_adc.py exactly.
b_values = np.arange(0, 2001, 500, dtype=int)  # [s/mm^2]  — matches qMRI_adc.py

TR = 5000
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
#   Pre-loop containers
# ================================================================================
# all_echo_images will become (n_b, n_dirs, 3, Ny, Nx) after the loop
all_echo_images = []
te1_ms = te2_ms = te3_ms = None  # read once from the first iteration

# %% ================================================================================
#   Main loop: sweep b-values, acquire triple SE per b-value
# ================================================================================
for b_value in b_values:
    print(f"Simulating sequence | b={b_value} s/mm² | TE={te1_ms} ms")
    # =================================================================================
    #   Generate sequence
    # =================================================================================
    name = f"DiffTripleSE-b{int(b_value)}"
    seq = EPIDiffusionTripleSEPulseqSeq(
        name=name,
        resolution=res,
        Nx=Nx,
        Ny=Ny,
        fov=fov,
        slice_thickness=slice_thickness,
        TE=TE1,
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
        uniform_spoiler_directions=False,
        uniform_spoiler_areas=False,
        phase_cycling=True,
        partial_fourier_factor=1,
        logger=logger,
    )
    seq.write()

    # Echo times are fixed for all b-values (TE1 is constant); read once.
    if te1_ms is None:
        te1_ms = seq.TE * 1e3
        te2_ms = seq.TE2 * 1e3
        te3_ms = seq.TE3 * 1e3
        assert (
            te1_ms < te2_ms < te3_ms
        ), f"Echo times not monotonic: {te1_ms:.1f} / {te2_ms:.1f} / {te3_ms:.1f} ms"
        print(
            f"Echo times: TE1={te1_ms:.1f} ms, TE2={te2_ms:.1f} ms, TE3={te3_ms:.1f} ms"
        )

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
    # ==============================================================================
    # Signal layout per b-value:
    #   [calibration: 3×Nx]
    #   | [dir0: EPI1 | EPI2 | EPI3]
    #   | [dir1: EPI1 | EPI2 | EPI3]
    #   | ...
    #   | [dir92: EPI1 | EPI2 | EPI3]
    # EPI1 uses partial Fourier; EPI2 and EPI3 always use pff=1.0.
    samples_per_cal = int(3 * seq.adc.num_samples)
    samples_epi1 = int(seq.Ny * seq.partial_fourier_factor) * seq.adc.num_samples
    samples_epi2 = seq.Ny * seq.adc.num_samples
    samples_epi3 = seq.Ny * seq.adc.num_samples
    samples_per_dir = samples_epi1 + samples_epi2 + samples_epi3
    n_dirs = len(seq.b_directions)

    assert signal.shape[0] == samples_per_cal + n_dirs * samples_per_dir, (
        f"Signal length mismatch: expected {samples_per_cal + n_dirs * samples_per_dir}, "
        f"got {signal.shape[0]}"
    )

    epi_signal = signal[samples_per_cal:].squeeze()

    echo1_kspaces = []
    echo2_kspaces = []
    echo3_kspaces = []
    for d in range(n_dirs):
        base = d * samples_per_dir
        echo1_kspaces.append(
            epi_signal[base : base + samples_epi1].reshape(
                int(seq.Ny * seq.partial_fourier_factor), seq.adc.num_samples
            )
        )
        echo2_kspaces.append(
            epi_signal[
                base + samples_epi1 : base + samples_epi1 + samples_epi2
            ].reshape(seq.Ny, seq.adc.num_samples)
        )
        echo3_kspaces.append(
            epi_signal[
                base + samples_epi1 + samples_epi2 : base + samples_per_dir
            ].reshape(seq.Ny, seq.adc.num_samples)
        )

    print(
        f"[b={b_value}] Echo k-spaces: "
        f"EPI1 {echo1_kspaces[0].shape}, EPI2 {echo2_kspaces[0].shape}, EPI3 {echo3_kspaces[0].shape}"
    )

    # ==============================================================================
    #   Calculate k-space trajectory
    # ==============================================================================
    # The EPI readout waveform is identical across all diffusion directions — diffusion
    # gradients precede the readout and modify spin phase only, not the k-space path.
    # Extract the three per-echo trajectories from direction-0's block and reuse them
    # for every direction, avoiding 93 redundant calculations.
    k_traj_adc, k_traj, _, _, t_adc = seq.seq.calculate_kspace()
    kx_norm = k_traj_adc[0] * fov / Nx
    ky_norm = k_traj_adc[1] * fov / Ny
    # print(
    #     f"  [k-traj] kx range: [{kx_norm.min():.4f}, {kx_norm.max():.4f}] "
    #     f"(expected Nyquist ≈ ±0.5, ramp overshoot OK)"
    # )
    traj = np.stack([kx_norm, ky_norm], axis=-1)

    d0 = samples_per_cal
    traj_epi1 = traj[d0 : d0 + samples_epi1]
    traj_epi2 = traj[d0 + samples_epi1 : d0 + samples_epi1 + samples_epi2]
    traj_epi3 = traj[d0 + samples_epi1 + samples_epi2 : d0 + samples_per_dir]

    # =================================================================================
    #   NUFFT reconstruction — build one operator per echo, reuse for all directions
    # =================================================================================
    img_size = (Ny, Nx)
    nufft_op_e1 = get_operator(
        backend_name="finufft",
        samples=traj_epi1,
        shape=img_size,
        n_coils=1,
        density=True,
    )
    nufft_op_e2 = get_operator(
        backend_name="finufft",
        samples=traj_epi2,
        shape=img_size,
        n_coils=1,
        density=True,
    )
    nufft_op_e3 = get_operator(
        backend_name="finufft",
        samples=traj_epi3,
        shape=img_size,
        n_coils=1,
        density=True,
    )

    dir_echo_images = []
    for d in range(n_dirs):
        imgs = []
        for ksp, op in [
            (echo1_kspaces[d], nufft_op_e1),
            (echo2_kspaces[d], nufft_op_e2),
            (echo3_kspaces[d], nufft_op_e3),
        ]:
            sig_t = torch.from_numpy(np.array(ksp)).to(torch.complex64)
            img_complex = op.adj_op(sig_t.flatten()).squeeze()
            imgs.append(img_complex.cpu().numpy())
        dir_echo_images.append(np.stack(imgs, axis=0))  # (3, Ny, Nx)

    all_echo_images.append(np.stack(dir_echo_images, axis=0))  # (n_dirs, 3, Ny, Nx)
    # print(f"  Reconstructed b={b_value}: {all_echo_images[-1].shape}")

    affine = np.array([[res, 0, 0, 0], [0, res, 0, 0], [0, 0, res, 0], [0, 0, 0, 1]])
    for d in range(n_dirs):
        for echo_idx, te_ms_echo in enumerate([te1_ms, te2_ms, te3_ms]):
            mag_img = np.abs(dir_echo_images[d][echo_idx])  # (Ny, Nx)
            echo_name = f"DiffTripleSE-b{int(b_value)}-dir{d}-TE{int(te_ms_echo)}-{'blipdown' if BLIP_DOWN else 'blipup'}"
            nii_path = os.path.join(ECHO_IMAGES_DIR_PATH, f"{echo_name}.nii.gz")
            nib.save(
                nib.Nifti1Image(
                    np.asarray(mag_img[:, :, np.newaxis], dtype=np.float32),
                    affine=affine,
                ),
                nii_path,
            )


print("Simulation loop complete.")

# %% ==============================================================================
#   Post-loop: assemble arrays
# =================================================================================
all_echo_images = np.array(all_echo_images)  # (n_b, n_dirs, 3, Ny, Nx)
print(f"all_echo_images shape: {all_echo_images.shape}")
assert all_echo_images.shape == (len(b_values), n_dirs, 3, Ny, Nx)

mag_images = np.abs(all_echo_images)  # (n_b, n_dirs, 3, Ny, Nx)
mag_echo1 = mag_images[:, :, 0, :, :]  # (n_b, n_dirs, Ny, Nx)
mag_echo2 = mag_images[:, :, 1, :, :]
mag_echo3 = mag_images[:, :, 2, :, :]
mag_images_combined = mag_images.mean(axis=2)  # (n_b, n_dirs, Ny, Nx)

# %% ==============================================================================
#   Visualize reconstructed images — rows: echoes, columns: subset of b-values
# =================================================================================
SHOW_DIR = 0
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


trace_dwi = compute_trace_dwi(mag_echo1)  # (n_b, Ny, Nx)
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
#   ADC vs reference comparison (all echoes combined)
# =================================================================================
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Triple SE ADC (all echoes combined) vs Reference")
ims_data = [
    (np.rot90(adc_nlls, -1) * 1e3, "NLLS (all echoes)", "ADC (x10⁻³ mm²/s)", "viridis"),
    (
        np.rot90(adc_ll, -1) * 1e3,
        "Log-Linear (all echoes)",
        "ADC (x10⁻³ mm²/s)",
        "viridis",
    ),
    (np.rot90(D, 1), "Reference D map", "D (x10⁻³ mm²/s)", "viridis"),
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
)
axs[1].set_title("Fractional Anisotropy (FA)")
axs[1].set_axis_off()
fig.colorbar(im, ax=axs[1], label="FA")
fig.suptitle("DTI maps from Triple SE (all echoes combined)")
plt.tight_layout()
plt.show()

# %% ==============================================================================
#   Save outputs
# =================================================================================
try:
    np.save(
        f"simulated/vol/ADC_MESE_{'blipdown' if BLIP_DOWN else 'blipup'}.npy", adc_nlls[0]
    )
    np.save(f"{VOLUMES_DIR_PATH}/ADC_triple_se.npy", adc_nlls)
    print(f"Saved all maps to {VOLUMES_DIR_PATH}")
except Exception as e:
    print(f"Could not save volumes: {e}")

# %%
