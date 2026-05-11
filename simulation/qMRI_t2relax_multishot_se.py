# IQ-BRAIN is funded by the European Union (MSCA Doctoral Network,
# December 2024–November 2028, Grant Agreement No. 101169519).

# %%
import os
import sys
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

seq_path = r"C:\Users\User\OneDrive\PhD\Sumbission\ESMRMB26\Pulseq-DiffusionMESE\pulseq_diffusion_mese"
if seq_path not in sys.path:
    sys.path.append(seq_path)

# %% ================================================================================
#   Imports
# ================================================================================
import MRzeroCore as mr0
import torch
import matplotlib.pyplot as plt

from SEMultishotPulseqSeq import SEMultishotPulseqSeq
from utils import SystemLimitType, fft_reconstruct_image
from utils_simulation import add_tumor_to_phantom, pad_to_cube

np.int = int
np.float = float
np.complex = complex

use_GPU = torch.cuda.is_available()

# ================================================================================
#   Paths
# ================================================================================
SEQUENCES_DIR_PATH = r".\simulated\seq"
VOLUMES_DIR_PATH = r".\simulated\vol"
PHANTOMS_DIR_PATH = r".\phantoms\brainweb"

# %% ==============================================================================
#   Simulation parameters
# ==============================================================================
fov = 224e-3
res = 2.33333333
slice_thickness = res * 1e-3
TEs = np.arange(80, 200, 10)   # ms — 12 TE values
TR = 5000
ETL = 1                       # echo train length; scan time = (Ny/ETL) × TR per TE
Nx = Ny = int(fov / slice_thickness)

N_shots = Ny // ETL
print(
    f"Matrix: {Ny}×{Nx}  ETL={ETL}  N_shots per TE={N_shots}  "
    f"Total TRs for all TEs: {len(TEs) * N_shots}  "
    f"(vs {len(TEs) * Ny} for ETL=1 — {ETL}× faster)"
)

# %% ==============================================================================
#   Load phantom
# ==============================================================================
PHANTOM_IDX = 4
NZ = 10
SLICE_IDX = 4

add_tumor = True
tumor_size = (10, 10, 10)

phantoms = [f for f in os.listdir(PHANTOMS_DIR_PATH) if f.endswith(".npz")]
print("Available phantoms:", phantoms)
phantom_path = os.path.join(PHANTOMS_DIR_PATH, phantoms[PHANTOM_IDX])

if os.path.isfile(phantom_path) and phantom_path.endswith(".npz"):
    phantom = mr0.VoxelGridPhantom.load(phantom_path)
    print(f"Loaded phantom {os.path.split(phantom_path)[-1]}. Shape: {phantom.D.shape}")

    if add_tumor:
        phantom = add_tumor_to_phantom(
            phantom,
            tumor_size=tumor_size,
            tumor_location="br",
            adc_tumor_core=1.5,
            adc_tumor_border=2.5,
        )
    phantom = phantom.interpolate(Nx, Ny, NZ)
    print(f"Resized phantom. Shape: {phantom.D.shape}")

    phantom = phantom.slices([SLICE_IDX])
    print(f"Selected slice. Shape: {phantom.D.shape}")
else:
    raise FileNotFoundError(f"Error: Invalid phantom file at {phantom_path}")

phantom_data = phantom.build()

# %% ================================================================================
#   Main simulation loop
# ================================================================================
# For each TE: build the multishot SE sequence, simulate all N_shots TRs, reshape
# the Cartesian k-space, and reconstruct with a plain inverse FFT.
# No NUFFT is needed because the readout is a conventional Cartesian gradient echo.
# ================================================================================
reconstructed_images = []   # (n_TEs,) list of (Ny, Nx) magnitude images

for te in TEs:
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
        system_type=SystemLimitType.EXTRASAFE,
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
for ax in axs[len(TEs):]:
    ax.set_visible(False)
plt.tight_layout()
plt.show()

# %% ==============================================================================
#   T2 fitting
# ==============================================================================
from utils_relaxometry import create_t2_map

images_stack = np.array(reconstructed_images)   # (n_TEs, Ny, Nx)
print(f"Images stack: {images_stack.shape}  TEs: {TEs}")

ims = []
for method in ["nlls", "loglinear"]:
    t2_result = create_t2_map(images_stack, TEs, method=method)
    ims.append(t2_result[0])

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
titles = ["NLLS Fit", "Log-Linear Fit"]
for i, ax in enumerate(axs):
    im = ax.imshow(np.rot90(ims[i], -1), cmap="viridis")
    ax.set_title(titles[i])
    fig.colorbar(im, ax=ax, label="T2 (ms)")
plt.suptitle(
    f"Multishot SE T2 maps (ETL={ETL}, {len(TEs)} TEs, {len(TEs)*N_shots} TRs)"
)
plt.tight_layout()
plt.show()

# %% ==============================================================================
#   Reference phantom comparison
# ==============================================================================
ref = mr0.VoxelGridPhantom.load(rf"{PHANTOMS_DIR_PATH}\{phantoms[PHANTOM_IDX]}")
max_dim = max(ref.D.shape)
ref.D = pad_to_cube(ref.D, max_dim)
ref.T2 = pad_to_cube(ref.T2, max_dim)
ref.T2dash = pad_to_cube(ref.T2dash, max_dim)
ref.T1 = pad_to_cube(ref.T1, max_dim)
ref.PD = pad_to_cube(ref.PD, max_dim)
ref.B0 = pad_to_cube(ref.B0, max_dim)
ref.B1 = pad_to_cube(ref.B1, max_dim)
ref = ref.interpolate(Nx, Ny, NZ).slices([SLICE_IDX])
ref.plot()

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
ims2 = [*ims, ref.T2]
for i, ax in enumerate(axs):
    if i < 2:
        im_data = np.rot90(ims2[i], 0) / 1000
        im_data = np.fliplr(im_data)  # flip horizontally to match ref orientation
    else:
        im_data = np.rot90(ims2[i], 1)
    ims2[i] = im_data  # save for later
    im = ax.imshow(im_data, cmap="viridis",)
    ax.set_title(titles[i])
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, label="T2 (s)")
plt.tight_layout()
plt.show()

# %%
try:
    np.save(rf"{VOLUMES_DIR_PATH}\T2_multishot_se.npy", ims2[0])
    np.save(rf"{VOLUMES_DIR_PATH}\T2_ref.npy", np.rot90(ims2[2], 2))
    print("Saved T2 maps.")
except Exception as e:
    print(f"Could not save: {e}")
# %%
