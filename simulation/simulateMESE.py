"""
Bloch-equation simulation of the diffusion triple spin-echo (MESE) EPI sequence.

Signal structure per TR (after the navigator calibration block)::

    [echo1: NY*PFF lines] → [echo2: NY lines] → [echo3: NY lines]   *  N_directions

The script:
    1. Loads a BrainWeb voxel-grid phantom (optionally inserts a synthetic tumour).
    2. Imports the Pulseq ``.seq`` file and extracts sequence definitions (TE, TR,
       b-value, diffusion directions, partial-Fourier factor, ADC samples).
    3. Runs a MRzeroCore Bloch-equation graph simulation (GPU or CPU).
    4. Splits the flat signal vector into calibration + per-direction * per-echo chunks.
    5. Reconstructs each echo with FFT (zero-filled partial Fourier) and NUFFT
       (density-compensated, ramp-sampling-aware via cufinufft).

Author      : Aron Gimesi <aron.gimesi@tecnico.ulisboa.pt>
Affiliation : Instituto Superior Técnico | MSCA-DN IQ-BRAIN
Date        : 2026
Context     : ESMRMB 2026 - Pulseq DiffusionMESE showcase

Funding acknowledgement (mandatory):
    IQ-BRAIN is funded by the European Union (MSCA Doctoral Network,
    December 2024–November 2028, Grant Agreement No. 101169519).
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
from mrinufft import get_operator

from utils_simulation import *

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
N_ECHOES = 3  # triple spin-echo

use_GPU = torch.cuda.is_available()

# %% ==============================================================================
#   Phantom parameters
# =================================================================================
PHANTOM_IDX = 0
NZ = 10
SLICE_IDX = 4

add_tumor = True
tumor_size = (10, 10, 10)

# %% ==============================================================================
#   Load phantom
# =================================================================================
phantoms = [f for f in os.listdir(PHANTOMS_DIR_PATH) if f.endswith(".npz")]
print("Available phantoms:", phantoms)
phantom_path = os.path.join(PHANTOMS_DIR_PATH, phantoms[PHANTOM_IDX])

if os.path.isfile(phantom_path) and phantom_path.endswith(".npz"):
    phantom = mr0.VoxelGridPhantom.load(phantom_path)
    print(f"Loaded phantom {os.path.split(phantom_path)[-1]}. Shape: {phantom.D.shape}")
    phantom = phantom.interpolate(NX, NY, NZ)
    print(f"Resized phantom. Shape: {phantom.D.shape}")
    phantom.voxel_size = torch.Tensor([0.00233, 0.00233, 0.00233])
    vox = phantom.voxel_size

    if add_tumor:
        phantom = add_tumor_to_phantom(
            phantom,
            tumor_size=tumor_size,
            tumor_location="bl",
            adc_tumor_core=1.5,
            adc_tumor_border=2.5,
        )
    phantom = phantom.slices([SLICE_IDX])
    print(f"Selected slice. Shape: {phantom.D.shape}")
else:
    raise FileNotFoundError(f"Error: Invalid phantom file at {phantom_path}")

PD = phantom.PD
T1 = phantom.T1
T2 = phantom.T2
D = phantom.D
B0 = phantom.B0
B1 = phantom.B1

phantom.plot()
phantom_data = phantom.build()

# %% ==============================================================================
#   Load sequence  (DiffMESE only)
# =================================================================================
SEQUENCE_IDX = 0


sequences = [
    f for f in os.listdir(SEQUENCES_DIR_PATH) if f.endswith(".seq") and "v14" in f
]
sequences = [
    f for f in sequences if "DiffSE" not in f
]  # keep MESE, drop single-echo SE

print("Available sequences:", sequences)
sequence_path = os.path.join(SEQUENCES_DIR_PATH, sequences[SEQUENCE_IDX])

seq0 = mr0.Sequence.import_file(sequence_path)
seq = Sequence()

# IMPORTANT!
# The read-in Pulseq sequence needs to be in v15 in order to use the new calculate_kspace() API
seq.read(sequence_path.replace("v14", "v15"))
print(f"Loaded sequence {os.path.split(sequence_path)[-1]}")

TR = seq.definitions.get("TR")
TEs = seq.definitions.get("TE")  # list [TE1, TE2, TE3] in seconds
AdcNumSamples = int(seq.definitions.get("AdcNumSamples"))
N_navigator_lines = int(seq.definitions.get("NNavigatorLines"))
NX = int(seq.definitions.get("Nx"))
NY = int(seq.definitions.get("Ny"))
PFF = float(seq.definitions.get("PartialFourierFactor"))  # echo 1 only
BVAL = seq.definitions.get("bValue")
N_directions_raw = seq.definitions.get("DiffusionDirections")
fov = float(seq.definitions.get("FOV")[0])  # in-plane FOV in metres

# Echo 1 uses partial Fourier; echoes 2 and 3 acquire the full k-space.
NY_acq = [int(NY * PFF), NY, NY]
samples_per_echo = [n * AdcNumSamples for n in NY_acq]
samples_per_dir = sum(samples_per_echo)
samples_per_cal = N_navigator_lines * AdcNumSamples

if isinstance(N_directions_raw, str):
    groups = re.findall(r"\[[^\[\]]*\]", N_directions_raw)
    directions = [list(ast.literal_eval(g)) for g in groups] if groups else []
    N_directions = len(directions)
elif isinstance(N_directions_raw, (list, tuple, np.ndarray)):
    directions = [list(x) for x in N_directions_raw]
    N_directions = len(directions)
else:
    directions = []
    N_directions = 0

print(
    f"N_directions={N_directions}, NY_acq per echo={NY_acq}, "
    f"samples_per_echo={samples_per_echo}, samples_per_dir={samples_per_dir}"
)

# %% ==============================================================================
#   Simulate sequence
# =================================================================================
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

print(f"Signal shape: {signal.shape}, dtype: {signal.dtype}")

TR_idx = 1
seq.plot(plot_now=False, time_range=(TR * TR_idx, TR * TR_idx + 0.4))
mr0.util.insert_signal_plot(seq=seq, signal=signal.numpy())
plt.show()

# %% ==============================================================================
#   Split signal: navigator + per-direction × per-echo
#
#   Layout:  [calib: N_nav × AdcNumSamples]
#            then per direction: [echo1: NY_acq[0] × AdcNumSamples]
#                                [echo2: NY_acq[1] × AdcNumSamples]
#                                [echo3: NY_acq[2] × AdcNumSamples]
# =================================================================================
signal_np = signal.squeeze().numpy()

calib_signal = signal_np[:samples_per_cal].reshape((N_navigator_lines, AdcNumSamples))
epi_signal = signal_np[samples_per_cal:]

# dir_echo_signal[d][e] → np.ndarray (NY_acq[e], AdcNumSamples)
dir_echo_signal = []
for d in range(N_directions):
    dir_start = d * samples_per_dir
    echo_signals = []
    offset = 0
    for e in range(N_ECHOES):
        chunk = epi_signal[
            dir_start + offset : dir_start + offset + samples_per_echo[e]
        ]
        echo_signals.append(chunk.reshape((NY_acq[e], AdcNumSamples)))
        offset += samples_per_echo[e]
    dir_echo_signal.append(echo_signals)

print(f"Calibration signal shape: {calib_signal.shape}")
print(
    f"Per-direction, per-echo signal shapes: "
    f"{[[e.shape for e in d] for d in dir_echo_signal]}"
)
# dir_echo_signal = np.stack(dir_echo_signal)  # shape (N_directions, N_ECHOES, NY_acq[e], AdcNumSamples)

# %% ==============================================================================
#   Ghost correction using calibration lines
#   The navigator is a single-echo 3-line readout at TE1. The same linear-phase
#   correction is applied to all three echoes: the timing-related ghost phase is
#   determined by the readout gradient waveform, which is identical for every echo.
# =================================================================================
# calib_np = (
#     calib_signal.numpy() if hasattr(calib_signal, "numpy") else np.asarray(calib_signal)
# )
# gc_dir_signal = np.zeros_like(dir_echo_signal)
# for i in range(N_directions):
#     gc_dir_signal[i] = epi_ghost_correction(calib_np, dir_echo_signal[i])


# %% ==============================================================================
#   FFT reconstruction
#   Echo 1 is zero-filled to (NY, NX): acquired lines at [0:NY_acq[0]], zeros after.
# =================================================================================
for d in range(N_directions):
    fig, axes = plt.subplots(1, N_ECHOES + 1, figsize=(16, 4))
    plt.suptitle(f"FFT recon — b={BVAL} s/mm², direction {directions[d]}")

    for e in range(N_ECHOES):
        kspace = dir_echo_signal[d][e]
        if NY_acq[e] < NY:
            # Zero-fill missing ky lines: acquired data covers from most-negative ky
            # upward; pad the high-ky end with zeros.
            kspace_full = np.zeros((NY, AdcNumSamples), dtype=kspace.dtype)
            kspace_full[: NY_acq[e]] = kspace
            kspace = kspace_full
        _, image = fft_reconstruct_image(kspace, use_gpu=use_GPU)
        axes[e].imshow(np.abs(image), cmap="gray")
        axes[e].set_title(f"Echo {e + 1}  TE={TEs[e] * 1e3:.0f} ms")
        axes[e].axis("off")

    axes[N_ECHOES].imshow(D.squeeze().cpu().rot90(1).numpy(), cmap="gray")
    axes[N_ECHOES].set_title("Phantom (T2)")
    axes[N_ECHOES].axis("off")
    plt.tight_layout()
    plt.show()

# %% ==============================================================================
#   K-space trajectory extraction and normalisation
#   calculate_kspace() integrates gradients cumulatively, so we align all
#   per-direction trajectories to a common origin (see simulateSE.py for rationale).
# =================================================================================
k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

kx_norm = k_traj_adc[0] * fov / NX
ky_norm = k_traj_adc[1] * fov / NY
print(
    f"[k-traj] kx range: [{kx_norm.min():.4f}, {kx_norm.max():.4f}] "
    f"(expected Nyquist ≈ ±0.5, ramp overshoot OK)"
)

traj = np.stack([kx_norm, ky_norm], axis=-1)

# %%  Split trajectory with the same structure as the signal
calibration_trajectory = traj[:samples_per_cal]
rest_traj = traj[samples_per_cal:]

# dir_echo_traj[d][e] → np.ndarray (samples_per_echo[e], 2)
dir_echo_traj = []
for d in range(N_directions):
    dir_start = d * samples_per_dir
    echo_trajs = []
    offset = 0
    for e in range(N_ECHOES):
        chunk = rest_traj[dir_start + offset : dir_start + offset + samples_per_echo[e]]
        echo_trajs.append(chunk)
        offset += samples_per_echo[e]
    dir_echo_traj.append(echo_trajs)

# DC offset alignment: anchor every direction's echo-e trajectory to direction 0.
# All EPI readouts share the same gradient waveform; only the cumulative starting
# offset differs between TRs due to unbalanced spoiler gradients.
traj_anchors = [dir_echo_traj[0][e][0] for e in range(N_ECHOES)]

aligned_echo_traj = [
    [t - t[0] + traj_anchors[e] for e, t in enumerate(dir_echo_traj[d])]
    for d in range(N_directions)
]

print(f"Calibration trajectory shape: {calibration_trajectory.shape}")
print(
    f"Per-direction, per-echo trajectory shapes: "
    f"{[[t.shape for t in d] for d in aligned_echo_traj]}"
)

# %% ==============================================================================
#   Build NUFFT operators  (one per echo, shared across directions)
#   Echoes 2 and 3 use the same full-kspace waveform but get separate operator
#   objects for clarity.
# =================================================================================
img_size = (NY, NX)

nufft_ops = [
    get_operator(
        backend_name="cufinufft",
        samples=aligned_echo_traj[0][e],  # direction 0 is the reference
        shape=img_size,
        n_coils=1,
        density=True,
    )
    for e in range(N_ECHOES)
]

# %% ==============================================================================
#   NUFFT reconstruction per direction × per echo
# =================================================================================
for d in range(N_directions):
    fig, axes = plt.subplots(1, N_ECHOES + 1, figsize=(16, 4))
    plt.suptitle(f"NUFFT recon — b={BVAL} s/mm², direction {directions[d]}")

    for e in range(N_ECHOES):
        sig = torch.from_numpy(dir_echo_signal[d][e]).to(torch.complex64)
        img_complex = nufft_ops[e].adj_op(sig.flatten()).squeeze()

        axes[e].imshow(torch.abs(img_complex).cpu().rot90(-1).numpy(), cmap="gray")
        axes[e].set_title(f"Echo {e + 1}  TE={TEs[e] * 1e3:.0f} ms")
        axes[e].axis("off")

    axes[N_ECHOES].imshow(D.squeeze().cpu().rot90(1).numpy(), cmap="gray")
    axes[N_ECHOES].set_title("Phantom (D)")
    axes[N_ECHOES].axis("off")
    plt.tight_layout()
    plt.show()

# %%
visualize_kspace_trajectory(seq)
# %%
