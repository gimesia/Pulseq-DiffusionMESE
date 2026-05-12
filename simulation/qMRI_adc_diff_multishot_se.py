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

from DiffusionSEMultishotPulseqSeq import DiffusionSEMultishotPulseqSeq
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

# Fixed TE across all b-values so T2 weighting is constant (Stejskal-Tanner).
TE = 100            # [ms]
TR = 5000           # [ms]
ETL = 1             # echo train length (1 = conventional SE per shot)

b_values = np.arange(100, 1501, 250)  # [s/mm²]
b_directions_count = 3               # electrostatic scheme — enables DTI
small_delta = 0.018                   # [s]
big_DELTA = 0.03                      # [s]

Nx = Ny = int(fov / slice_thickness)
N_shots = Ny // ETL

print(
    f"Matrix: {Ny}*{Nx}  ETL={ETL}  N_shots={N_shots}  Total TRs: {len(b_values) * N_shots}  "
    f"b-values: {b_values}  directions: {b_directions_count}"
)

# %% ==============================================================================
#   Load phantom
# ==============================================================================
PHANTOM_IDX = 4
NZ = 10
SLICE_IDX = 4

add_tumor = True
tumor_size = (20, 20, 20)

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
    phantom.plot()
    phantom = phantom.interpolate(Nx, Ny, NZ)
    print(f"Resized phantom. Shape: {phantom.D.shape}")
    phantom = phantom.slices([SLICE_IDX])
    print(f"Selected slice. Shape: {phantom.D.shape}")
else:
    raise FileNotFoundError(f"Error: Invalid phantom file at {phantom_path}")

phantom_data = phantom.build()

1# %% ================================================================================
#   Main simulation loop: sweep b-values
# ================================================================================
# Signal layout per b-value:
#   n_dirs × N_shots × ETL × Nx = n_dirs × Ny × Nx samples
# Per-direction k-space: signal[d*Ny*Nx:(d+1)*Ny*Nx].reshape(Ny, Nx)
# ================================================================================
all_images = []     # will become (n_b, n_dirs, Ny, Nx) after the loop
n_dirs = None       # read from first sequence

for b_value in b_values:
    # ============================================================================
    #   Build sequence
    # ============================================================================
    name = f"DiffSEMultishot-b{int(b_value)}"
    seq = DiffusionSEMultishotPulseqSeq(
        name=name,
        fov=fov,
        Nx=Nx,
        Ny=Ny,
        slice_thickness=slice_thickness,
        TR=TR,
        TE=TE,
        ETL=ETL,
        b_value=b_value,
        b_directions=b_directions_count,
        small_delta=small_delta,
        big_DELTA=big_DELTA,
        save_dir=SEQUENCES_DIR_PATH,
        v141_compat=True,
        system_type=SystemLimitType.SAFE,
    )
    seq.build_seq()
    seq.write()

    if n_dirs is None:
        n_dirs = len(seq.b_directions)
        print(f"n_dirs={n_dirs}, TE={seq.TE*1e3:.1f} ms")

    seq_filename = seq.get_save_filename()
    print(f"[b={b_value}]  file: {seq_filename}")

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

    assert signal.shape[0] == n_dirs * Ny * Nx, (
        f"Signal length mismatch: expected {n_dirs * Ny * Nx}, got {signal.shape[0]}"
    )

    # ============================================================================
    #   Reconstruct per direction (Cartesian iFFT — no NUFFT needed)
    # ============================================================================
    dir_images = []
    for d in range(n_dirs):
        kspace = signal[d * Ny * Nx : (d + 1) * Ny * Nx].numpy().reshape(Ny, Nx)
        img_mag, _ = fft_reconstruct_image(kspace, use_gpu=use_GPU)
        dir_images.append(img_mag.squeeze())
    all_images.append(np.stack(dir_images, axis=0))   # (n_dirs, Ny, Nx)
    print(f"  Reconstructed b={b_value}: {all_images[-1].shape}")

print("Simulation loop complete.")

# %% ==============================================================================
#   Assemble output arrays
# ==============================================================================
all_images = np.array(all_images)   # (n_b, n_dirs, Ny, Nx)
mag_images = np.abs(all_images)
print(f"mag_images shape: {mag_images.shape}")

# %% ==============================================================================
#   Visualize — show a subset of b-values for direction 0
# ==============================================================================
SHOW_DIR = 0
b_subset = np.linspace(0, len(b_values) - 1, min(6, len(b_values)), dtype=int)

fig, axs = plt.subplots(1, len(b_subset), figsize=(3 * len(b_subset), 3))
fig.suptitle(f"Diffusion SE Multishot DWI (direction {SHOW_DIR}, TE={TE} ms)")
for col, b_idx in enumerate(b_subset):
    axs[col].imshow(mag_images[b_idx, SHOW_DIR], cmap="gray")
    axs[col].set_title(f"b={b_values[b_idx]:.0f}")
    axs[col].set_axis_off()
plt.tight_layout()
plt.show()

# %% ==============================================================================
#   Trace DWI — geometric mean across directions
# ==============================================================================
eps = 1e-12


def compute_trace_dwi(mag):
    """Geometric mean across diffusion directions. mag: (n_b, n_dirs, Ny, Nx)."""
    return np.exp(np.mean(np.log(mag + eps), axis=1))


trace_dwi = compute_trace_dwi(mag_images)   # (n_b, Ny, Nx)
print(f"Trace DWI shape: {trace_dwi.shape}")

# %% ==============================================================================
#   ADC maps
# ==============================================================================
from utils_diffusion import create_adc_map  # noqa: E402

adc_nlls, _ = create_adc_map(trace_dwi, b_values, method="nlls")
adc_ll, _ = create_adc_map(trace_dwi, b_values, method="loglinear")

mask = adc_nlls > 0
print(
    f"ADC NLLS:       range [{adc_nlls.min()*1e3:.3f}, {adc_nlls.max()*1e3:.3f}] "
    f"x10⁻³ mm²/s, median (brain) = {np.median(adc_nlls[mask])*1e3:.3f}"
)
print(
    f"ADC log-linear: range [{adc_ll.min()*1e3:.3f}, {adc_ll.max()*1e3:.3f}] "
    f"x10⁻³ mm²/s, median (brain) = {np.median(adc_ll[mask])*1e3:.3f}"
)

# %% ==============================================================================
#   ADC comparison plot (NLLS vs log-linear)
# ==============================================================================
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
titles = ["NLLS Fit", "Log-Linear Fit"]
for i, adc in enumerate([adc_nlls, adc_ll]):
    im = axs[i].imshow(adc * 1e3, cmap="viridis")
    axs[i].set_title(titles[i])
    axs[i].set_axis_off()
    fig.colorbar(im, ax=axs[i], label="ADC (x10⁻³ mm²/s)")
plt.suptitle(
    f"Diffusion SE Multishot ADC (TE={TE} ms, ETL={ETL}, "
    f"{b_directions_count} directions)"
)
plt.tight_layout()
plt.show()

# %% ==============================================================================
#   DTI maps — FA and MD
# ==============================================================================
from utils_diffusion import create_dti_maps  # noqa: E402

fa_map, md_map, eigvals_map, dti_s0_map = create_dti_maps(
    mag_images,           # (n_b, n_dirs, Ny, Nx)
    b_values,
    seq.b_directions,     # (n_dirs, 3) unit vectors from the last loop iteration
)
print(
    f"FA range: [{fa_map.min():.3f}, {fa_map.max():.3f}]  "
    f"MD range: [{md_map.min()*1e3:.3f}, {md_map.max()*1e3:.3f}] x10⁻³ mm²/s"
)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
im = axs[0].imshow(md_map * 1e3, cmap="viridis")
axs[0].set_title("Mean Diffusivity (MD)")
axs[0].set_axis_off()
fig.colorbar(im, ax=axs[0], label="MD (x10⁻³ mm²/s)")

im = axs[1].imshow(fa_map, cmap="inferno")
axs[1].set_title("Fractional Anisotropy (FA)")
axs[1].set_axis_off()
fig.colorbar(im, ax=axs[1], label="FA")
plt.suptitle(f"Diffusion SE Multishot DTI (TE={TE} ms, ETL={ETL})")
plt.tight_layout()
plt.show()

# %% ==============================================================================
#   Reference phantom comparison (ADC)
# ==============================================================================
ref = mr0.VoxelGridPhantom.load(rf"{PHANTOMS_DIR_PATH}\{phantoms[PHANTOM_IDX]}")
max_dim = max(ref.D.shape)
if add_tumor:
    ref = add_tumor_to_phantom(
        ref,
        tumor_size=tumor_size,
        tumor_location="br",
        adc_tumor_core=1.5,
        adc_tumor_border=2.5,
    )
ref.D = pad_to_cube(ref.D, max_dim)
ref.T2 = pad_to_cube(ref.T2, max_dim)
ref.T2dash = pad_to_cube(ref.T2dash, max_dim)
ref.T1 = pad_to_cube(ref.T1, max_dim)
ref.PD = pad_to_cube(ref.PD, max_dim)
ref.B0 = pad_to_cube(ref.B0, max_dim)
ref.B1 = pad_to_cube(ref.B1, max_dim)



ref = ref.interpolate(Nx, Ny, NZ).slices([SLICE_IDX])
ref.plot()

fig, axs = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Diffusion SE Multishot ADC vs Reference")
ims_data = [
    (np.fliplr(adc_nlls) * 1e3, "NLLS ADC", "viridis"),
    (np.fliplr(adc_ll) * 1e3, "Log-Linear ADC", "viridis"),
    (np.rot90(ref.D, 1), "Reference D map", "viridis"),
]
for ax, (im_data, title, cmap) in zip(axs, ims_data):
    im = ax.imshow(im_data, cmap=cmap)
    ax.set_title(title)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, label="ADC (x10⁻³ mm²/s)")
plt.tight_layout()
plt.show()

# %%
try:
    np.save(rf"{VOLUMES_DIR_PATH}\ADC_multishot_se.npy", adc_nlls)
    # np.save(rf"{VOLUMES_DIR_PATH}\ADC_multishot_se_loglinear.npy", adc_ll)
    # np.save(rf"{VOLUMES_DIR_PATH}\FA_multishot_se.npy", fa_map)
    # np.save(rf"{VOLUMES_DIR_PATH}\MD_multishot_se.npy", md_map)
    # np.save(rf"{VOLUMES_DIR_PATH}\mag_images_multishot_se_adc.npy", mag_images)
    print("Saved ADC/DTI maps.")
except Exception as e:
    print(f"Could not save: {e}")
# %%
