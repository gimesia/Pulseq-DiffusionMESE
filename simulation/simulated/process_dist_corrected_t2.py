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

fieldmap_dir = rf"{file_dir}/topup_results (FIELDMAPS)"
t2_dir = rf"{file_dir}/t2_vol"
t2_corrected_dir = rf"{fieldmap_dir}/t2_volumes_corrected_same"


t2_blipdown_paths = sorted(
    [i for i in os.listdir(rf"{t2_dir}") if i.endswith("blipdown.nii.gz")], key=te_value
)
t2_blipdown_corrected_paths = sorted(
    [i for i in os.listdir(rf"{t2_corrected_dir}") if i.endswith("blipdown_corrected.nii.gz")], key=te_value
)

t2_blipup_paths = sorted(
    [i for i in os.listdir(rf"{t2_dir}") if i.endswith("blipup.nii.gz")], key=te_value
)
t2_blipup_corrected_paths = sorted(
    [i for i in os.listdir(rf"{t2_corrected_dir}") if i.endswith("blipup_corrected.nii.gz")], key=te_value
)

TEs = get_TEs(t2_blipup_paths)
print(f"Echo times (TEs) extracted: {TEs}")
print(
    f"Found {len(t2_blipup_paths)} blipup and {len(t2_blipdown_paths)} blipdown files."
)
print(f"Found {len(t2_blipup_corrected_paths)} corrected blipup and {len(t2_blipdown_corrected_paths)} corrected blipdown files.")


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
niftis_bd_corrected = []
for pth in t2_blipdown_corrected_paths:
    print(f"Loading {pth}...", end="\r", flush=True)
    nifti, data = load_nifti(os.path.join(t2_corrected_dir, pth))
    niftis_bd_corrected.append((data))
niftis_bd_corrected = np.stack(niftis_bd_corrected, axis=-1)
print(f"All corrected blipdown files loaded. Shape: {niftis_bd_corrected.shape}")
niftis_bu_corrected = []
for pth in t2_blipup_corrected_paths:
    print(f"Loading {pth}...", end="\r", flush=True)
    nifti, data = load_nifti(os.path.join(t2_corrected_dir, pth))
    niftis_bu_corrected.append((data))
niftis_bu_corrected = np.stack(niftis_bu_corrected, axis=-1)
print(f"All corrected blipup files loaded. Shape: {niftis_bu_corrected.shape}")

    


# %%===============================
# Transpose/Resample Data
# =================================
niftis_bu_resampled = np.transpose(niftis_bu, (3, 2, 0, 1))
niftis_bd_resampled = np.transpose(niftis_bd, (3, 2, 0, 1))
niftis_bd_corrected_resampled = np.transpose(niftis_bd_corrected, (3, 2, 0, 1))
niftis_bu_corrected_resampled = np.transpose(niftis_bu_corrected, (3, 2, 0, 1))
print(f"Resampled blipup shape: {niftis_bu_resampled.shape}")
print(f"Resampled blipdown shape: {niftis_bd_resampled.shape}")
print(f"Resampled corrected blipdown shape: {niftis_bd_corrected_resampled.shape}")
print(f"Resampled corrected blipup shape: {niftis_bu_corrected_resampled.shape}")

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

t2_blipup_map_transposed = np.transpose(t2_blipup_map, (2, 1, 0))

bu_out = nib.Nifti1Image(np.asarray(t2_blipup_map_transposed), affine)
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

t2_blipdown_map_transposed = np.transpose(t2_blipdown_map, (2, 1, 0))
bd_out = nib.Nifti1Image(np.asarray(t2_blipdown_map_transposed), affine)
nib.save(bd_out, os.path.join(".", "t2_blipdown_map.nii.gz"))

# %%==============================
t2_blipdown_map_corrected = []
s0_blipdown_map_corrected = []
for i in range(niftis_bd_corrected_resampled.shape[1]):
    print(
        f"Processing corrected blipdown slice {i+1}/{niftis_bd_corrected_resampled.shape[1]}...",
        end="\r",
        flush=True,
    )
    t2_map, s0_map = create_t2_map(
        niftis_bd_corrected_resampled[:, i, :, :], TEs, t2_bounds=(0.0, 2.2)
    )
    t2_blipdown_map_corrected.append(t2_map)
    s0_blipdown_map_corrected.append(s0_map)
t2_blipdown_map_corrected = np.stack(t2_blipdown_map_corrected, axis=0)
s0_blipdown_map_corrected = np.stack(s0_blipdown_map_corrected, axis=0)
print(f"Corrected blipdown T2 map shape: {t2_blipdown_map_corrected.shape}")

t2_blipdown_map_corrected_transposed = np.transpose(t2_blipdown_map_corrected, (2, 1, 0))
bd_out_corrected = nib.Nifti1Image(np.asarray(t2_blipdown_map_corrected_transposed), affine)
nib.save(bd_out_corrected, os.path.join(".", "t2_blipdown_map_corrected.nii.gz"))

# =================================
t2_blipup_map_corrected = []
s0_blipup_map_corrected = []
for i in range(niftis_bu_corrected_resampled.shape[1]):
    print(
        f"Processing corrected blipup slice {i+1}/{niftis_bu_corrected_resampled.shape[1]}...",
        end="\r",
        flush=True,
    )
    t2_map, s0_map = create_t2_map(
        niftis_bu_corrected_resampled[:, i, :, :], TEs, t2_bounds=(0.0, 2.2)
    )
    t2_blipup_map_corrected.append(t2_map)
    s0_blipup_map_corrected.append(s0_map)
t2_blipup_map_corrected = np.stack(t2_blipup_map_corrected, axis=0)
s0_blipup_map_corrected = np.stack(s0_blipup_map_corrected, axis=0)
print(f"Corrected blipup T2 map shape: {t2_blipup_map_corrected.shape}")

t2_blipup_map_corrected_transposed = np.transpose(t2_blipup_map_corrected, (2, 1, 0))
bu_out_corrected = nib.Nifti1Image(np.asarray(t2_blipup_map_corrected_transposed), affine)
nib.save(bu_out_corrected, os.path.join(".", "t2_blipup_map_corrected.nii.gz"))

# %%===============================
# Save Output
# =================================
t2_blipup_map_rot = np.flip(np.rot90(t2_blipup_map_transposed, k=1, axes=(0, 1)), axis=(0))
t2_blipdown_map_rot = np.flip(np.rot90(t2_blipdown_map_transposed, k=1, axes=(0, 1)), axis=(0))
t2_blipup_map_corrected_rot = np.flip(np.rot90(t2_blipup_map_corrected_transposed, k=1, axes=(0, 1)), axis=(0))
t2_blipdown_map_corrected_rot = np.flip(np.rot90(t2_blipdown_map_corrected_transposed, k=1, axes=(0, 1)), axis=(0))

bd_out = nib.Nifti1Image(t2_blipdown_map_rot, affine)
bu_out = nib.Nifti1Image(t2_blipup_map_rot, affine)
bd_out_corrected = nib.Nifti1Image(t2_blipdown_map_corrected_rot, affine)
bu_out_corrected = nib.Nifti1Image(t2_blipup_map_corrected_rot, affine)
nib.save(bu_out, os.path.join(".", "t2_blipup_map.nii.gz"))
nib.save(bd_out, os.path.join(".", "t2_blipdown_map.nii.gz"))
nib.save(bu_out_corrected, os.path.join(".", "t2_blipup_map_corrected.nii.gz"))
nib.save(bd_out_corrected, os.path.join(".", "t2_blipdown_map_corrected.nii.gz"))


