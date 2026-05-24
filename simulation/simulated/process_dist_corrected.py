# %%===============================
# Imports
# =================================
import re

from utils_relaxometry import create_t2_map
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os


# =================================
# Helper Functions
# =================================
def load_nifti(file_path):
    """Load a NIfTI file and return the image data and affine."""
    nifti = nib.load(file_path)
    data = nifti.get_fdata()
    return nifti, data


def get_TEs(filepaths):
    """Extract echo times (TEs) from the filenames."""
    TEs = []
    for path in filepaths:
        # Assuming the TE is in the format 'TE{number}' in the filename
        basename = os.path.basename(path)
        te_str = basename.split("_")[-2]  # Adjust index based on filename structure
        te_value = float(te_str.replace("TE", ""))
        TEs.append(te_value)
    return np.array(sorted(TEs)) * 0.001


def te_value(f):
    m = re.search(r"TE(\d+)", f)
    return int(m.group(1)) if m else float("inf")


# =================================
# Discover Files and Prepare TE List
# =================================
file_dir = os.path.dirname(os.path.abspath(__file__))
t2_dir = rf"{file_dir}/t2_vol"
t2_blipup_paths = sorted(
    [i for i in os.listdir(rf"{t2_dir}") if i.endswith("blipup.nii.gz")], key=te_value
)
t2_blipdown_paths = sorted(
    [i for i in os.listdir(rf"{t2_dir}") if i.endswith("blipdown.nii.gz")], key=te_value
)
TEs = get_TEs(t2_blipup_paths)
print(f"Echo times (TEs) extracted: {TEs}")
print(
    f"Found {len(t2_blipup_paths)} blipup and {len(t2_blipdown_paths)} blipdown files."
)


# %%===============================
# Load NIfTI Volumes (Blipup / Blipdown)
# =================================
niftis_bu = []
for pth in t2_blipup_paths:
    print(f"Loading {pth}...", end="\r", flush=True)
    nifti, data = load_nifti(os.path.join(t2_dir, pth))
    niftis_bu.append((data))
niftis_bu = np.stack(niftis_bu, axis=-1)
print(f"All blipup files loaded. Shape: {niftis_bu.shape}")
niftis_bd = []
for pth in t2_blipdown_paths:
    print(f"Loading {pth}...", end="\r", flush=True)
    nifti, data = load_nifti(os.path.join(t2_dir, pth))
    niftis_bd.append((data))
niftis_bd = np.stack(niftis_bd, axis=-1)
print(f"All blipdown files loaded. Shape: {niftis_bd.shape}")

# =================================
# Transpose/Resample Data
# =================================
niftis_bu_resampled = np.transpose(niftis_bu, (3, 2, 0, 1))
niftis_bd_resampled = np.transpose(niftis_bd, (3, 2, 0, 1))
print(f"Resampled blipup shape: {niftis_bu_resampled.shape}")
print(f"Resampled blipdown shape: {niftis_bd_resampled.shape}")

# %%===============================
# Process T2 Maps
# =================================
affine = load_nifti(os.path.join(t2_dir, t2_blipdown_paths[0]))[0].affine

t2_blipup_map = []
s0_blipup_map = []
for i in range(niftis_bu_resampled.shape[1]):
    print(
        f"Processing blipup slice {i+1}/{niftis_bu_resampled.shape[1]}...",
        end="\r",
        flush=True,
    )
    t2_map, s0_map = create_t2_map(niftis_bu_resampled[:, i, :, :], TEs)
    t2_blipup_map.append(t2_map)
    s0_blipup_map.append(s0_map)
t2_blipup_map = np.stack(t2_blipup_map, axis=0)
s0_blipup_map = np.stack(s0_blipup_map, axis=0)
print(f"Blipup T2 map shape: {t2_blipup_map.shape}")

t2_blipup_map = np.transpose(t2_blipup_map, (2, 1, 0))

bu_out = nib.Nifti1Image(np.asarray(t2_blipup_map), affine)
nib.save(bu_out, os.path.join(".", "t2_blipup_map.nii.gz"))

# ================================
t2_blipdown_map = []
s0_blipdown_map = []
for i in range(niftis_bd_resampled.shape[1]):
    print(
        f"Processing blipdown slice {i+1}/{niftis_bd_resampled.shape[1]}...",
        end="\r",
        flush=True,
    )
    t2_map, s0_map = create_t2_map(
        niftis_bd_resampled[:, i, :, :], TEs, t2_bounds=(0.0, 2.2)
    )
    t2_blipdown_map.append(t2_map)
    s0_blipdown_map.append(s0_map)
t2_blipdown_map = np.stack(t2_blipdown_map, axis=0)
s0_blipdown_map = np.stack(s0_blipdown_map, axis=0)
print(f"Blipdown T2 map shape: {t2_blipdown_map.shape}")

t2_blipdown_map = np.transpose(t2_blipdown_map, (2, 1, 0))
bd_out = nib.Nifti1Image(np.asarray(t2_blipdown_map), affine)
nib.save(bd_out, os.path.join(".", "t2_blipdown_map.nii.gz"))


# # ===== Alternative: Multiprocessing Implementation =====
# import multiprocessing
# from functools import partial
# from tqdm.auto import tqdm  # auto picks notebook bar if ipywidgets is present, else falls back gracefully

# def process_slice(args, TEs):
#     i, slice_data = args
#     return create_t2_map(slice_data, TEs)

# def run_processing(data, TEs):
#     num_processes = min(multiprocessing.cpu_count(), 6)
#     slices = [(i, data[:, i, :, :]) for i in range(data.shape[1])]
#     process_func = partial(process_slice, TEs=TEs)

#     with multiprocessing.Pool(processes=num_processes) as pool:
#         results = list(tqdm(
#             pool.imap(process_func, slices),
#             total=len(slices),
#             desc="Processing slices"
#         ))

#     t2_maps = [res[0] for res in results]
#     s0_maps = [res[1] for res in results]
#     return np.stack(t2_maps, axis=0), np.stack(s0_maps, axis=0)

# if __name__ == '__main__':
#     print("Processing blipup slices...")
#     t2_blipup_map, s0_blipup_map = run_processing(niftis_bu_resampled, TEs)
#     print(f"Blipup T2 map shape: {t2_blipup_map.shape}")

#     print("Processing blipdown slices...")
#     t2_blipdown_map, s0_blipdown_map = run_processing(niftis_bd_resampled, TEs)
#     print(f"Blipdown T2 map shape: {t2_blipdown_map.shape}")

# %%===============================
# Save Output
# =================================
t2_blipup_map_rot = np.rot90(t2_blipup_map, k=0, axes=(0, 1))
t2_blipdown_map_rot = np.rot90(t2_blipdown_map, k=0, axes=(0, 1))

bd_out = nib.Nifti1Image(t2_blipdown_map_rot, affine)
bu_out = nib.Nifti1Image(t2_blipup_map_rot, affine)
nib.save(bu_out, os.path.join(".", "t2_blipup_map.nii.gz"))
nib.save(bd_out, os.path.join(".", "t2_blipdown_map.nii.gz"))

# %%
