# %%===============================
# Imports
# =================================
import re

from utils_diffusion import create_adc_map
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


def get_Bvals(filepaths):
    """Extract b-values from the filenames."""
    Bs = []
    for path in filepaths:
        # b-value is in the format 'b{number}' in the filename
        basename = os.path.basename(path)
        m = re.search(r"b(\d+)", basename)
        Bs.append(float(m.group(1)))
    return np.unique(np.array(sorted(Bs)))


def b_value(f):
    m = re.search(r"b(\d+)", f)
    return int(m.group(1)) if m else float("inf")


# =================================
# Discover Files and Prepare B List
# =================================
# Directions dir0 / dir1 / dir2 == Z / Y / X (orthogonal).
TE = "TE100"          # echo to process
ADC_MAX = 3.6         # physiological ceiling
B_MAX = np.inf        # upper b-value cutoff; np.inf -> use all b-values

file_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(file_dir, "volumes_corrected")
fieldmap_dir = rf"{file_dir}/topup_results (FIELDMAPS)"
adc_dir = rf"{file_dir}/diff_vol"
adc_corrected_dir = rf"{fieldmap_dir}/diff_volumes_corrected_same"


# --- blipup (raw): one sorted list per direction ---
diff_blipup_dir0_paths = sorted(
    [i for i in os.listdir(adc_dir) if i.endswith("blipup.nii.gz") and "_dir0_" in i and TE in i],
    key=b_value,
)
diff_blipup_dir1_paths = sorted(
    [i for i in os.listdir(adc_dir) if i.endswith("blipup.nii.gz") and "_dir1_" in i and TE in i],
    key=b_value,
)
diff_blipup_dir2_paths = sorted(
    [i for i in os.listdir(adc_dir) if i.endswith("blipup.nii.gz") and "_dir2_" in i and TE in i],
    key=b_value,
)

# --- blipdown (raw) ---
diff_blipdown_dir0_paths = sorted(
    [i for i in os.listdir(adc_dir) if i.endswith("blipdown.nii.gz") and "_dir0_" in i and TE in i],
    key=b_value,
)
diff_blipdown_dir1_paths = sorted(
    [i for i in os.listdir(adc_dir) if i.endswith("blipdown.nii.gz") and "_dir1_" in i and TE in i],
    key=b_value,
)
diff_blipdown_dir2_paths = sorted(
    [i for i in os.listdir(adc_dir) if i.endswith("blipdown.nii.gz") and "_dir2_" in i and TE in i],
    key=b_value,
)

# --- blipup (corrected) ---
diff_blipup_corrected_dir0_paths = sorted(
    [i for i in os.listdir(adc_corrected_dir) if i.endswith("blipup_corrected.nii.gz") and "_dir0_" in i and TE in i],
    key=b_value,
)
diff_blipup_corrected_dir1_paths = sorted(
    [i for i in os.listdir(adc_corrected_dir) if i.endswith("blipup_corrected.nii.gz") and "_dir1_" in i and TE in i],
    key=b_value,
)
diff_blipup_corrected_dir2_paths = sorted(
    [i for i in os.listdir(adc_corrected_dir) if i.endswith("blipup_corrected.nii.gz") and "_dir2_" in i and TE in i],
    key=b_value,
)

# --- blipdown (corrected) ---
diff_blipdown_corrected_dir0_paths = sorted(
    [i for i in os.listdir(adc_corrected_dir) if i.endswith("blipdown_corrected.nii.gz") and "_dir0_" in i and TE in i],
    key=b_value,
)
diff_blipdown_corrected_dir1_paths = sorted(
    [i for i in os.listdir(adc_corrected_dir) if i.endswith("blipdown_corrected.nii.gz") and "_dir1_" in i and TE in i],
    key=b_value,
)
diff_blipdown_corrected_dir2_paths = sorted(
    [i for i in os.listdir(adc_corrected_dir) if i.endswith("blipdown_corrected.nii.gz") and "_dir2_" in i and TE in i],
    key=b_value,
)

Bs = get_Bvals(diff_blipup_dir0_paths)
print(f"B-values extracted: {Bs}")

# Subject prefix: everything before "_b{n}_" in the first input filename
_subj_match = re.match(r"^(.*?)_b\d+_", os.path.basename(diff_blipup_dir0_paths[0]))
subject = _subj_match.group(1) if _subj_match else "subject"
print(f"Subject: {subject}")

print(f"Found {len(diff_blipup_dir0_paths)} blipup dir0 files.")
print(f"Found {len(diff_blipdown_dir0_paths)} blipdown dir0 files.")
print(f"Found {len(diff_blipup_corrected_dir0_paths)} corrected blipup dir0 files.")
print(f"Found {len(diff_blipdown_corrected_dir0_paths)} corrected blipdown dir0 files.")

# b_max mask (applied to every loaded stack the same way)
b_keep = Bs <= B_MAX
Bs_used = Bs[b_keep]
print(f"Using b-values (after B_MAX={B_MAX}): {Bs_used}")


# %%===============================
# Load NIfTI Volumes - blipup (raw), one stack per direction
# =================================
niftis_bu_dir0 = []
for pth in diff_blipup_dir0_paths:
    print(f"Loading {pth}...", end="\r", flush=True)
    nifti, data = load_nifti(os.path.join(adc_dir, pth))
    niftis_bu_dir0.append((data))
niftis_bu_dir0 = np.stack(niftis_bu_dir0, axis=-1)

niftis_bu_dir1 = []
for pth in diff_blipup_dir1_paths:
    print(f"Loading {pth}...", end="\r", flush=True)
    nifti, data = load_nifti(os.path.join(adc_dir, pth))
    niftis_bu_dir1.append((data))
niftis_bu_dir1 = np.stack(niftis_bu_dir1, axis=-1)

niftis_bu_dir2 = []
for pth in diff_blipup_dir2_paths:
    print(f"Loading {pth}...", end="\r", flush=True)
    nifti, data = load_nifti(os.path.join(adc_dir, pth))
    niftis_bu_dir2.append((data))
niftis_bu_dir2 = np.stack(niftis_bu_dir2, axis=-1)
print(f"All blipup files loaded. Shapes: {niftis_bu_dir0.shape}, {niftis_bu_dir1.shape}, {niftis_bu_dir2.shape}")

# blipdown (raw)
niftis_bd_dir0 = []
for pth in diff_blipdown_dir0_paths:
    print(f"Loading {pth}...", end="\r", flush=True)
    nifti, data = load_nifti(os.path.join(adc_dir, pth))
    niftis_bd_dir0.append((data))
niftis_bd_dir0 = np.stack(niftis_bd_dir0, axis=-1)

niftis_bd_dir1 = []
for pth in diff_blipdown_dir1_paths:
    print(f"Loading {pth}...", end="\r", flush=True)
    nifti, data = load_nifti(os.path.join(adc_dir, pth))
    niftis_bd_dir1.append((data))
niftis_bd_dir1 = np.stack(niftis_bd_dir1, axis=-1)

niftis_bd_dir2 = []
for pth in diff_blipdown_dir2_paths:
    print(f"Loading {pth}...", end="\r", flush=True)
    nifti, data = load_nifti(os.path.join(adc_dir, pth))
    niftis_bd_dir2.append((data))
niftis_bd_dir2 = np.stack(niftis_bd_dir2, axis=-1)
print(f"All blipdown files loaded. Shapes: {niftis_bd_dir0.shape}, {niftis_bd_dir1.shape}, {niftis_bd_dir2.shape}")

# blipup (corrected)
niftis_bu_corrected_dir0 = []
for pth in diff_blipup_corrected_dir0_paths:
    print(f"Loading {pth}...", end="\r", flush=True)
    nifti, data = load_nifti(os.path.join(adc_corrected_dir, pth))
    niftis_bu_corrected_dir0.append((data))
niftis_bu_corrected_dir0 = np.stack(niftis_bu_corrected_dir0, axis=-1)

niftis_bu_corrected_dir1 = []
for pth in diff_blipup_corrected_dir1_paths:
    print(f"Loading {pth}...", end="\r", flush=True)
    nifti, data = load_nifti(os.path.join(adc_corrected_dir, pth))
    niftis_bu_corrected_dir1.append((data))
niftis_bu_corrected_dir1 = np.stack(niftis_bu_corrected_dir1, axis=-1)

niftis_bu_corrected_dir2 = []
for pth in diff_blipup_corrected_dir2_paths:
    print(f"Loading {pth}...", end="\r", flush=True)
    nifti, data = load_nifti(os.path.join(adc_corrected_dir, pth))
    niftis_bu_corrected_dir2.append((data))
niftis_bu_corrected_dir2 = np.stack(niftis_bu_corrected_dir2, axis=-1)
print(f"All corrected blipup files loaded. Shapes: {niftis_bu_corrected_dir0.shape}, {niftis_bu_corrected_dir1.shape}, {niftis_bu_corrected_dir2.shape}")

# blipdown (corrected)
niftis_bd_corrected_dir0 = []
for pth in diff_blipdown_corrected_dir0_paths:
    print(f"Loading {pth}...", end="\r", flush=True)
    nifti, data = load_nifti(os.path.join(adc_corrected_dir, pth))
    niftis_bd_corrected_dir0.append((data))
niftis_bd_corrected_dir0 = np.stack(niftis_bd_corrected_dir0, axis=-1)

niftis_bd_corrected_dir1 = []
for pth in diff_blipdown_corrected_dir1_paths:
    print(f"Loading {pth}...", end="\r", flush=True)
    nifti, data = load_nifti(os.path.join(adc_corrected_dir, pth))
    niftis_bd_corrected_dir1.append((data))
niftis_bd_corrected_dir1 = np.stack(niftis_bd_corrected_dir1, axis=-1)

niftis_bd_corrected_dir2 = []
for pth in diff_blipdown_corrected_dir2_paths:
    print(f"Loading {pth}...", end="\r", flush=True)
    nifti, data = load_nifti(os.path.join(adc_corrected_dir, pth))
    niftis_bd_corrected_dir2.append((data))
niftis_bd_corrected_dir2 = np.stack(niftis_bd_corrected_dir2, axis=-1)
print(f"All corrected blipdown files loaded. Shapes: {niftis_bd_corrected_dir0.shape}, {niftis_bd_corrected_dir1.shape}, {niftis_bd_corrected_dir2.shape}")


# %%===============================
# Transpose/Resample Data  -> (n_b, slice, y, x)
# =================================
niftis_bu_dir0_resampled = np.transpose(niftis_bu_dir0, (3, 2, 0, 1))
niftis_bu_dir1_resampled = np.transpose(niftis_bu_dir1, (3, 2, 0, 1))
niftis_bu_dir2_resampled = np.transpose(niftis_bu_dir2, (3, 2, 0, 1))

niftis_bd_dir0_resampled = np.transpose(niftis_bd_dir0, (3, 2, 0, 1))
niftis_bd_dir1_resampled = np.transpose(niftis_bd_dir1, (3, 2, 0, 1))
niftis_bd_dir2_resampled = np.transpose(niftis_bd_dir2, (3, 2, 0, 1))

niftis_bu_corrected_dir0_resampled = np.transpose(niftis_bu_corrected_dir0, (3, 2, 0, 1))
niftis_bu_corrected_dir1_resampled = np.transpose(niftis_bu_corrected_dir1, (3, 2, 0, 1))
niftis_bu_corrected_dir2_resampled = np.transpose(niftis_bu_corrected_dir2, (3, 2, 0, 1))

niftis_bd_corrected_dir0_resampled = np.transpose(niftis_bd_corrected_dir0, (3, 2, 0, 1))
niftis_bd_corrected_dir1_resampled = np.transpose(niftis_bd_corrected_dir1, (3, 2, 0, 1))
niftis_bd_corrected_dir2_resampled = np.transpose(niftis_bd_corrected_dir2, (3, 2, 0, 1))

# apply b_max along the b-axis (axis 0) for every stack
niftis_bu_dir0_resampled = niftis_bu_dir0_resampled[b_keep]
niftis_bu_dir1_resampled = niftis_bu_dir1_resampled[b_keep]
niftis_bu_dir2_resampled = niftis_bu_dir2_resampled[b_keep]
niftis_bd_dir0_resampled = niftis_bd_dir0_resampled[b_keep]
niftis_bd_dir1_resampled = niftis_bd_dir1_resampled[b_keep]
niftis_bd_dir2_resampled = niftis_bd_dir2_resampled[b_keep]
niftis_bu_corrected_dir0_resampled = niftis_bu_corrected_dir0_resampled[b_keep]
niftis_bu_corrected_dir1_resampled = niftis_bu_corrected_dir1_resampled[b_keep]
niftis_bu_corrected_dir2_resampled = niftis_bu_corrected_dir2_resampled[b_keep]
niftis_bd_corrected_dir0_resampled = niftis_bd_corrected_dir0_resampled[b_keep]
niftis_bd_corrected_dir1_resampled = niftis_bd_corrected_dir1_resampled[b_keep]
niftis_bd_corrected_dir2_resampled = niftis_bd_corrected_dir2_resampled[b_keep]

# geometric-mean trace per pipeline: (S_dir0 * S_dir1 * S_dir2) ** (1/3)
eps = 1e-12
niftis_bu_trace_resampled = (
    np.maximum(niftis_bu_dir0_resampled, eps)
    * np.maximum(niftis_bu_dir1_resampled, eps)
    * np.maximum(niftis_bu_dir2_resampled, eps)
) ** (1.0 / 3.0)
niftis_bd_trace_resampled = (
    np.maximum(niftis_bd_dir0_resampled, eps)
    * np.maximum(niftis_bd_dir1_resampled, eps)
    * np.maximum(niftis_bd_dir2_resampled, eps)
) ** (1.0 / 3.0)
niftis_bu_corrected_trace_resampled = (
    np.maximum(niftis_bu_corrected_dir0_resampled, eps)
    * np.maximum(niftis_bu_corrected_dir1_resampled, eps)
    * np.maximum(niftis_bu_corrected_dir2_resampled, eps)
) ** (1.0 / 3.0)
niftis_bd_corrected_trace_resampled = (
    np.maximum(niftis_bd_corrected_dir0_resampled, eps)
    * np.maximum(niftis_bd_corrected_dir1_resampled, eps)
    * np.maximum(niftis_bd_corrected_dir2_resampled, eps)
) ** (1.0 / 3.0)

print(f"Resampled blipup trace shape: {niftis_bu_trace_resampled.shape}")
print(f"Resampled blipdown trace shape: {niftis_bd_trace_resampled.shape}")
print(f"Resampled corrected blipup trace shape: {niftis_bu_corrected_trace_resampled.shape}")
print(f"Resampled corrected blipdown trace shape: {niftis_bd_corrected_trace_resampled.shape}")


# %%===============================
# Process ADC Maps - blipup (raw): 3 directions + trace
# =================================
affine = load_nifti(os.path.join(adc_dir, diff_blipup_dir0_paths[0]))[0].affine

# loop over the 3 directions + the trace; key -> resampled stack
bu_inputs = {
    "dir0": niftis_bu_dir0_resampled,
    "dir1": niftis_bu_dir1_resampled,
    "dir2": niftis_bu_dir2_resampled,
    "trace": niftis_bu_trace_resampled,
}
for key, stack in bu_inputs.items():
    adc_blipup_map = []
    for i in range(stack.shape[1]):
        print(f"Processing blipup {key} slice {i+1}/{stack.shape[1]}...", end="\r", flush=True)
        adc_map, s0_map = create_adc_map(stack[:, i, :, :], Bs_used, adc_max=ADC_MAX)
        adc_blipup_map.append(adc_map)
    adc_blipup_map = np.stack(adc_blipup_map, axis=0)
    print(f"Blipup {key} ADC map shape: {adc_blipup_map.shape}")

    adc_blipup_map_transposed = np.transpose(adc_blipup_map, (2, 1, 0))
    adc_blipup_map_rot = np.flip(np.rot90(adc_blipup_map_transposed, k=1, axes=(0, 1)), axis=(0))
    nib.save(nib.Nifti1Image(adc_blipup_map_rot, affine), os.path.join(save_dir, f"{subject}_adc_blipup_{key}.nii.gz"))


# %%===============================
# Process ADC Maps - blipdown (raw): 3 directions + trace
# =================================
bd_inputs = {
    "dir0": niftis_bd_dir0_resampled,
    "dir1": niftis_bd_dir1_resampled,
    "dir2": niftis_bd_dir2_resampled,
    "trace": niftis_bd_trace_resampled,
}
for key, stack in bd_inputs.items():
    adc_blipdown_map = []
    for i in range(stack.shape[1]):
        print(f"Processing blipdown {key} slice {i+1}/{stack.shape[1]}...", end="\r", flush=True)
        adc_map, s0_map = create_adc_map(stack[:, i, :, :], Bs_used, adc_max=ADC_MAX)
        adc_blipdown_map.append(adc_map)
    adc_blipdown_map = np.stack(adc_blipdown_map, axis=0)
    print(f"Blipdown {key} ADC map shape: {adc_blipdown_map.shape}")

    adc_blipdown_map_transposed = np.transpose(adc_blipdown_map, (2, 1, 0))
    adc_blipdown_map_rot = np.flip(np.rot90(adc_blipdown_map_transposed, k=1, axes=(0, 1)), axis=(0))
    nib.save(nib.Nifti1Image(adc_blipdown_map_rot, affine), os.path.join(save_dir, f"{subject}_adc_blipdown_{key}.nii.gz"))


# %%===============================
# Process ADC Maps - blipup (corrected): 3 directions + trace
# =================================
bu_corrected_inputs = {
    "dir0": niftis_bu_corrected_dir0_resampled,
    "dir1": niftis_bu_corrected_dir1_resampled,
    "dir2": niftis_bu_corrected_dir2_resampled,
    "trace": niftis_bu_corrected_trace_resampled,
}
for key, stack in bu_corrected_inputs.items():
    adc_blipup_map_corrected = []
    for i in range(stack.shape[1]):
        print(f"Processing corrected blipup {key} slice {i+1}/{stack.shape[1]}...", end="\r", flush=True)
        adc_map, s0_map = create_adc_map(stack[:, i, :, :], Bs_used, adc_max=ADC_MAX)
        adc_blipup_map_corrected.append(adc_map)
    adc_blipup_map_corrected = np.stack(adc_blipup_map_corrected, axis=0)
    print(f"Corrected blipup {key} ADC map shape: {adc_blipup_map_corrected.shape}")

    adc_blipup_map_corrected_transposed = np.transpose(adc_blipup_map_corrected, (2, 1, 0))
    adc_blipup_map_corrected_rot = np.flip(np.rot90(adc_blipup_map_corrected_transposed, k=1, axes=(0, 1)), axis=(0))
    nib.save(nib.Nifti1Image(adc_blipup_map_corrected_rot, affine), os.path.join(save_dir, f"{subject}_adc_blipup_corrected_{key}.nii.gz"))


# %%===============================
# Process ADC Maps - blipdown (corrected): 3 directions + trace
# =================================
bd_corrected_inputs = {
    "dir0": niftis_bd_corrected_dir0_resampled,
    "dir1": niftis_bd_corrected_dir1_resampled,
    "dir2": niftis_bd_corrected_dir2_resampled,
    "trace": niftis_bd_corrected_trace_resampled,
}
for key, stack in bd_corrected_inputs.items():
    adc_blipdown_map_corrected = []
    for i in range(stack.shape[1]):
        print(f"Processing corrected blipdown {key} slice {i+1}/{stack.shape[1]}...", end="\r", flush=True)
        adc_map, s0_map = create_adc_map(stack[:, i, :, :], Bs_used, adc_max=ADC_MAX)
        adc_blipdown_map_corrected.append(adc_map)
    adc_blipdown_map_corrected = np.stack(adc_blipdown_map_corrected, axis=0)
    print(f"Corrected blipdown {key} ADC map shape: {adc_blipdown_map_corrected.shape}")

    adc_blipdown_map_corrected_transposed = np.transpose(adc_blipdown_map_corrected, (2, 1, 0))
    adc_blipdown_map_corrected_rot = np.flip(np.rot90(adc_blipdown_map_corrected_transposed, k=1, axes=(0, 1)), axis=(0))
    nib.save(nib.Nifti1Image(adc_blipdown_map_corrected_rot, affine), os.path.join(save_dir, f"{subject}_adc_blipdown_corrected_{key}.nii.gz"))


# %%===============================
# 
# =================================
