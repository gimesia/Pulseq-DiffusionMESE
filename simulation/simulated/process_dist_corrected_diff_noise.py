# IQ-BRAIN is funded by the European Union (MSCA Doctoral Network,
# December 2024–November 2028, Grant Agreement No. 101169519).
#
# Noise-injected counterpart of ``process_dist_corrected_diff.py``.
#
# Same I/O contract — load per-direction blipup / blipdown raw and corrected
# stacks at TE=100, fit ADC with ``create_adc_map`` per slice on each
# direction plus the geometric-mean trace, save the resulting maps — but
# injects image-domain Rician noise just before the fit.
#
# sigma is calibrated *once per stack variant* (blipup-raw, blipdown-raw,
# blipup-corrected, blipdown-corrected) from the b=0 image of the highest-SNR
# echo (TE1 = 100 ms, the only TE loaded) averaged across diffusion
# directions, restricted to white matter:
#
#     sigma_var = mean(S_b0_TE1_dirAvg[WM_mask]) / SNR_TARGET
#
# The same scalar sigma is then reused for every (b, direction) volume in
# that variant. The trace is recomputed from the noisy per-direction stacks
# (geometric mean across directions), exactly as in the noise-free script,
# so the fitter sees a self-consistent noisy dataset.
#
# Outputs:
#   - {subject}_adc_{blip}_{dir|trace}_noise.nii.gz
#   - {subject}_adc_{blip}_corrected_{dir|trace}_noise.nii.gz
#   - {subject}_adcw_b0_TE1_{blip}_dir0_noise_injected.nii.gz (example)
#
# Author      : Aron Gimesi <aron.gimesi@tecnico.ulisboa.pt>
# Affiliation : Instituto Superior Técnico | MSCA-DN IQ-BRAIN
# Date        : 2026
# Context     : ESMRMB 2026 — Pulseq DiffusionMESE showcase

# %%===============================
# Imports
# =================================
import os
import re
import sys

import nibabel as nib
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR  = os.path.dirname(_THIS_DIR)
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from utils_diffusion import create_adc_map
from utils_sim_lib import register_to_reference_3d
from utils_noise import (
    add_rician_noise,
    compute_sigma,
    load_tissue_masks,
)


# =================================================================================
# Config
# =================================================================================
SNR_TARGET     = 20.0     # sigma_var = WM(b=0, TE1, dir-avg) mean / SNR_TARGET
SEED           = 0        # base seed; each variant gets a derived sub-seed
REGISTER_MASKS = False    # if True, ANTs-Rigid-register tissue masks to each
                          # variant's b=0 / TE1 / dir-averaged volume before
                          # sigma calibration. Default off; raw blipup/blipdown
                          # are aligned with the masks by construction. Enable
                          # when the corrected (topup-warped) variants show
                          # residual misalignment.


# =================================
# Helper Functions
# =================================
def load_nifti(file_path):
    """Load a NIfTI file and return the image data and affine."""
    nifti = nib.load(file_path)
    data = nifti.get_fdata()
    return nifti, data


def get_Bvals(filepaths):
    Bs = []
    for path in filepaths:
        basename = os.path.basename(path)
        m = re.search(r"b(\d+)", basename)
        Bs.append(float(m.group(1)))
    return np.unique(np.array(sorted(Bs)))


def b_value(f):
    m = re.search(r"b(\d+)", f)
    return int(m.group(1)) if m else float("inf")


def _b0_index(b_values: np.ndarray) -> int:
    """Index of b=0 in a sorted ``b_values`` array; falls back to argmin."""
    bs = b_values.tolist()
    if 0 in bs or 0.0 in bs:
        return bs.index(0) if 0 in bs else bs.index(0.0)
    return int(np.argmin(b_values))


def _b0_reference_yxz(dir_stacks: tuple, b_values: np.ndarray) -> np.ndarray:
    """Direction-averaged b=0 magnitude volume in (y, x, slice) form."""
    b0 = _b0_index(b_values)
    s = np.mean(np.stack([d[b0] for d in dir_stacks], axis=0), axis=0)  # (slice, y, x)
    return np.transpose(s, (1, 2, 0)).astype(np.float32)                # (y, x, slice)


def _register_masks_to_b0(
    masks_yxz: dict,
    dir_stacks: tuple,
    b_values: np.ndarray,
    label: str,
) -> dict:
    """Rigid-register each tissue mask to the variant's b=0 / TE1 reference."""
    ref_yxz = _b0_reference_yxz(dir_stacks, b_values)
    out = {}
    for tissue, mask in masks_yxz.items():
        moving = mask.astype(np.float32)
        warped = register_to_reference_3d(moving, ref_yxz, type_of_transform="Rigid")
        warped = np.nan_to_num(warped, nan=0.0, posinf=0.0, neginf=0.0)
        out[tissue] = warped >= 0.5
        moved = int(np.abs(out[tissue].astype(int) - mask.astype(int)).sum())
        print(f"[ADC-noise] {label}: registered {tissue} mask, "
              f"voxel-flip vs. unregistered = {moved}")
    return out


def _calibrate_sigma_diff(
    dir_stacks: tuple,           # (dir0, dir1, dir2) each (n_b, slice, y, x)
    b_values: np.ndarray,
    wm_mask_yxz: np.ndarray,
    snr_target: float,
    label: str,
) -> float:
    """One scalar sigma from the dir-averaged b=0 / TE1 WM mean."""
    b0 = _b0_index(b_values)
    # Geometric mean would zero out direction-wise noise; for the SNR proxy
    # the arithmetic mean over the 3 directions is the right thing — it
    # reflects the per-volume noise floor without exponentiating it.
    s_ref_volume = np.mean(np.stack([d[b0] for d in dir_stacks], axis=0), axis=0)
    # transpose mask to (slice, y, x) to match stack axes
    wm_mask = np.transpose(wm_mask_yxz, (2, 0, 1))
    if s_ref_volume.shape != wm_mask.shape:
        raise ValueError(
            f"{label}: b=0 volume {s_ref_volume.shape} vs WM mask {wm_mask.shape}"
        )
    active = s_ref_volume.sum(axis=(1, 2)) > 0
    if not active.any():
        raise ValueError(f"{label}: b=0 volume has no non-empty slices")
    wm_active = wm_mask & active[:, None, None]
    print(f"[ADC-noise] {label}: b=0 active slices = "
          f"{int(active.sum())}/{s_ref_volume.shape[0]}")
    return compute_sigma(s_ref_volume, wm_active, snr_target)


def _apply_noise(stack: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return add_rician_noise(stack, sigma, rng)


def _trace_from_dirs(dir0: np.ndarray, dir1: np.ndarray, dir2: np.ndarray) -> np.ndarray:
    """Eps-floored geometric mean across 3 directions (matches noise-free script)."""
    eps = 1e-12
    return (np.maximum(dir0, eps) * np.maximum(dir1, eps) * np.maximum(dir2, eps)) ** (1.0 / 3.0)


def _fit_adc_volume(stack: np.ndarray, Bs_used: np.ndarray, adc_max: float, label: str):
    """Per-slice ADC fit, mirroring the original dist-corrected loop."""
    out = []
    for i in range(stack.shape[1]):
        print(f"Processing {label} slice {i+1}/{stack.shape[1]}...", end="\r", flush=True)
        adc_map, _ = create_adc_map(stack[:, i, :, :], Bs_used, adc_max=adc_max)
        out.append(adc_map)
    print()
    return np.stack(out, axis=0)


# =================================
# Discover Files and Prepare B List
# =================================
# Directions dir0 / dir1 / dir2 == Z / Y / X (orthogonal).
TE = "TE100"          # echo to process — same as the noise-free reference script
ADC_MAX = 3.6         # physiological ceiling (units consistent with adc_max in utils_diffusion)
B_MAX = np.inf        # upper b-value cutoff; np.inf -> use all b-values

file_dir = _THIS_DIR

fieldmap_dir = rf"{file_dir}/topup_results (FIELDMAPS)"
adc_dir = rf"{file_dir}/diff_vol"
adc_corrected_dir = rf"{fieldmap_dir}/diff_volumes_corrected_same"


# --- blipup (raw) ---
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

_subj_match = re.match(r"^(.*?)_b\d+_", os.path.basename(diff_blipup_dir0_paths[0]))
subject = _subj_match.group(1) if _subj_match else "subject"
print(f"Subject: {subject}")

print(f"Found {len(diff_blipup_dir0_paths)} blipup dir0 files.")
print(f"Found {len(diff_blipdown_dir0_paths)} blipdown dir0 files.")
print(f"Found {len(diff_blipup_corrected_dir0_paths)} corrected blipup dir0 files.")
print(f"Found {len(diff_blipdown_corrected_dir0_paths)} corrected blipdown dir0 files.")

b_keep = Bs <= B_MAX
Bs_used = Bs[b_keep]
print(f"Using b-values (after B_MAX={B_MAX}): {Bs_used}")

# Tissue masks (need WM for the calibration anchor).
masks_dir = os.path.join(file_dir, "masks")
phantom_name = subject if subject.startswith("brainweb-") else f"brainweb-{subject}"
tissue_masks = load_tissue_masks(masks_dir, phantom_name, ("wm", "gm", "csf"))
print(f"Tissue masks loaded   shape={tissue_masks['wm'].shape}   "
      f"|WM|={int(tissue_masks['wm'].sum())}")


# %%===============================
# Load NIfTI Volumes — blipup (raw), one stack per direction
# =================================
niftis_bu_dir0 = []
for pth in diff_blipup_dir0_paths:
    print(f"Loading {pth}...", end="\r", flush=True)
    nifti, data = load_nifti(os.path.join(adc_dir, pth))
    niftis_bu_dir0.append(data)
niftis_bu_dir0 = np.stack(niftis_bu_dir0, axis=-1)

niftis_bu_dir1 = []
for pth in diff_blipup_dir1_paths:
    print(f"Loading {pth}...", end="\r", flush=True)
    nifti, data = load_nifti(os.path.join(adc_dir, pth))
    niftis_bu_dir1.append(data)
niftis_bu_dir1 = np.stack(niftis_bu_dir1, axis=-1)

niftis_bu_dir2 = []
for pth in diff_blipup_dir2_paths:
    print(f"Loading {pth}...", end="\r", flush=True)
    nifti, data = load_nifti(os.path.join(adc_dir, pth))
    niftis_bu_dir2.append(data)
niftis_bu_dir2 = np.stack(niftis_bu_dir2, axis=-1)
print(f"All blipup files loaded. Shapes: {niftis_bu_dir0.shape}, {niftis_bu_dir1.shape}, {niftis_bu_dir2.shape}")

# blipdown (raw)
niftis_bd_dir0 = []
for pth in diff_blipdown_dir0_paths:
    print(f"Loading {pth}...", end="\r", flush=True)
    nifti, data = load_nifti(os.path.join(adc_dir, pth))
    niftis_bd_dir0.append(data)
niftis_bd_dir0 = np.stack(niftis_bd_dir0, axis=-1)

niftis_bd_dir1 = []
for pth in diff_blipdown_dir1_paths:
    print(f"Loading {pth}...", end="\r", flush=True)
    nifti, data = load_nifti(os.path.join(adc_dir, pth))
    niftis_bd_dir1.append(data)
niftis_bd_dir1 = np.stack(niftis_bd_dir1, axis=-1)

niftis_bd_dir2 = []
for pth in diff_blipdown_dir2_paths:
    print(f"Loading {pth}...", end="\r", flush=True)
    nifti, data = load_nifti(os.path.join(adc_dir, pth))
    niftis_bd_dir2.append(data)
niftis_bd_dir2 = np.stack(niftis_bd_dir2, axis=-1)
print(f"All blipdown files loaded. Shapes: {niftis_bd_dir0.shape}, {niftis_bd_dir1.shape}, {niftis_bd_dir2.shape}")

# blipup (corrected)
niftis_bu_corrected_dir0 = []
for pth in diff_blipup_corrected_dir0_paths:
    print(f"Loading {pth}...", end="\r", flush=True)
    nifti, data = load_nifti(os.path.join(adc_corrected_dir, pth))
    niftis_bu_corrected_dir0.append(data)
niftis_bu_corrected_dir0 = np.stack(niftis_bu_corrected_dir0, axis=-1)

niftis_bu_corrected_dir1 = []
for pth in diff_blipup_corrected_dir1_paths:
    print(f"Loading {pth}...", end="\r", flush=True)
    nifti, data = load_nifti(os.path.join(adc_corrected_dir, pth))
    niftis_bu_corrected_dir1.append(data)
niftis_bu_corrected_dir1 = np.stack(niftis_bu_corrected_dir1, axis=-1)

niftis_bu_corrected_dir2 = []
for pth in diff_blipup_corrected_dir2_paths:
    print(f"Loading {pth}...", end="\r", flush=True)
    nifti, data = load_nifti(os.path.join(adc_corrected_dir, pth))
    niftis_bu_corrected_dir2.append(data)
niftis_bu_corrected_dir2 = np.stack(niftis_bu_corrected_dir2, axis=-1)
print(f"All corrected blipup files loaded. Shapes: {niftis_bu_corrected_dir0.shape}, {niftis_bu_corrected_dir1.shape}, {niftis_bu_corrected_dir2.shape}")

# blipdown (corrected)
niftis_bd_corrected_dir0 = []
for pth in diff_blipdown_corrected_dir0_paths:
    print(f"Loading {pth}...", end="\r", flush=True)
    nifti, data = load_nifti(os.path.join(adc_corrected_dir, pth))
    niftis_bd_corrected_dir0.append(data)
niftis_bd_corrected_dir0 = np.stack(niftis_bd_corrected_dir0, axis=-1)

niftis_bd_corrected_dir1 = []
for pth in diff_blipdown_corrected_dir1_paths:
    print(f"Loading {pth}...", end="\r", flush=True)
    nifti, data = load_nifti(os.path.join(adc_corrected_dir, pth))
    niftis_bd_corrected_dir1.append(data)
niftis_bd_corrected_dir1 = np.stack(niftis_bd_corrected_dir1, axis=-1)

niftis_bd_corrected_dir2 = []
for pth in diff_blipdown_corrected_dir2_paths:
    print(f"Loading {pth}...", end="\r", flush=True)
    nifti, data = load_nifti(os.path.join(adc_corrected_dir, pth))
    niftis_bd_corrected_dir2.append(data)
niftis_bd_corrected_dir2 = np.stack(niftis_bd_corrected_dir2, axis=-1)
print(f"All corrected blipdown files loaded. Shapes: {niftis_bd_corrected_dir0.shape}, {niftis_bd_corrected_dir1.shape}, {niftis_bd_corrected_dir2.shape}")


# %%===============================
# Transpose to (n_b, slice, y, x) and apply b_max cutoff
# =================================
def _resample(arr):
    return np.transpose(arr, (3, 2, 0, 1))[b_keep]

niftis_bu_dir0_resampled = _resample(niftis_bu_dir0)
niftis_bu_dir1_resampled = _resample(niftis_bu_dir1)
niftis_bu_dir2_resampled = _resample(niftis_bu_dir2)
niftis_bd_dir0_resampled = _resample(niftis_bd_dir0)
niftis_bd_dir1_resampled = _resample(niftis_bd_dir1)
niftis_bd_dir2_resampled = _resample(niftis_bd_dir2)
niftis_bu_corrected_dir0_resampled = _resample(niftis_bu_corrected_dir0)
niftis_bu_corrected_dir1_resampled = _resample(niftis_bu_corrected_dir1)
niftis_bu_corrected_dir2_resampled = _resample(niftis_bu_corrected_dir2)
niftis_bd_corrected_dir0_resampled = _resample(niftis_bd_corrected_dir0)
niftis_bd_corrected_dir1_resampled = _resample(niftis_bd_corrected_dir1)
niftis_bd_corrected_dir2_resampled = _resample(niftis_bd_corrected_dir2)


# %%===============================
# Optional: ANTs-Rigid-register tissue masks to each variant's b=0 reference
# =================================
bu_dirs       = (niftis_bu_dir0_resampled,           niftis_bu_dir1_resampled,           niftis_bu_dir2_resampled)
bd_dirs       = (niftis_bd_dir0_resampled,           niftis_bd_dir1_resampled,           niftis_bd_dir2_resampled)
bu_corr_dirs  = (niftis_bu_corrected_dir0_resampled, niftis_bu_corrected_dir1_resampled, niftis_bu_corrected_dir2_resampled)
bd_corr_dirs  = (niftis_bd_corrected_dir0_resampled, niftis_bd_corrected_dir1_resampled, niftis_bd_corrected_dir2_resampled)

if REGISTER_MASKS:
    print()
    print("--- Registering tissue masks to b=0 / TE1 (ANTs Rigid, per-slice) ---")
    masks_bu           = _register_masks_to_b0(tissue_masks, bu_dirs,      Bs_used, "blipup (raw)")
    masks_bd           = _register_masks_to_b0(tissue_masks, bd_dirs,      Bs_used, "blipdown (raw)")
    masks_bu_corrected = _register_masks_to_b0(tissue_masks, bu_corr_dirs, Bs_used, "blipup (corrected)")
    masks_bd_corrected = _register_masks_to_b0(tissue_masks, bd_corr_dirs, Bs_used, "blipdown (corrected)")
else:
    masks_bu = masks_bd = masks_bu_corrected = masks_bd_corrected = tissue_masks


# %%===============================
# Calibrate sigma per variant from the b=0, TE1=100, dir-averaged WM mean
# =================================
print()
print(f"--- Calibrating sigma at SNR_TARGET={SNR_TARGET} from b=0 / TE={TE} WM mean ---")
sigma_bu           = _calibrate_sigma_diff(bu_dirs,      Bs_used, masks_bu["wm"],           SNR_TARGET, "blipup (raw)")
sigma_bd           = _calibrate_sigma_diff(bd_dirs,      Bs_used, masks_bd["wm"],           SNR_TARGET, "blipdown (raw)")
sigma_bu_corrected = _calibrate_sigma_diff(bu_corr_dirs, Bs_used, masks_bu_corrected["wm"], SNR_TARGET, "blipup (corrected)")
sigma_bd_corrected = _calibrate_sigma_diff(bd_corr_dirs, Bs_used, masks_bd_corrected["wm"], SNR_TARGET, "blipdown (corrected)")


# %%===============================
# Inject Rician noise — one sigma per variant, reused across all dirs and b-values
# =================================
print()
print("--- Injecting Rician noise into every (b, dir) volume ---")
def _noisify(stacks, sigma, seed_offset):
    """Apply Rician noise to a 3-tuple of (dir0, dir1, dir2) stacks.

    Different sub-seeds per direction so the noise field is independent
    across directions but the variance level (sigma) is the same.
    """
    rng_seed = SEED + seed_offset
    return tuple(_apply_noise(s, sigma, rng_seed + 100 * d) for d, s in enumerate(stacks))

(niftis_bu_dir0_noisy, niftis_bu_dir1_noisy, niftis_bu_dir2_noisy) = _noisify(
    (niftis_bu_dir0_resampled, niftis_bu_dir1_resampled, niftis_bu_dir2_resampled),
    sigma_bu, 1,
)
(niftis_bd_dir0_noisy, niftis_bd_dir1_noisy, niftis_bd_dir2_noisy) = _noisify(
    (niftis_bd_dir0_resampled, niftis_bd_dir1_resampled, niftis_bd_dir2_resampled),
    sigma_bd, 2,
)
(niftis_bu_corr_dir0_noisy, niftis_bu_corr_dir1_noisy, niftis_bu_corr_dir2_noisy) = _noisify(
    (niftis_bu_corrected_dir0_resampled, niftis_bu_corrected_dir1_resampled, niftis_bu_corrected_dir2_resampled),
    sigma_bu_corrected, 3,
)
(niftis_bd_corr_dir0_noisy, niftis_bd_corr_dir1_noisy, niftis_bd_corr_dir2_noisy) = _noisify(
    (niftis_bd_corrected_dir0_resampled, niftis_bd_corrected_dir1_resampled, niftis_bd_corrected_dir2_resampled),
    sigma_bd_corrected, 4,
)

# Recompute trace from the noisy per-direction stacks (eps-floored geomean).
niftis_bu_trace_noisy           = _trace_from_dirs(niftis_bu_dir0_noisy,      niftis_bu_dir1_noisy,      niftis_bu_dir2_noisy)
niftis_bd_trace_noisy           = _trace_from_dirs(niftis_bd_dir0_noisy,      niftis_bd_dir1_noisy,      niftis_bd_dir2_noisy)
niftis_bu_corrected_trace_noisy = _trace_from_dirs(niftis_bu_corr_dir0_noisy, niftis_bu_corr_dir1_noisy, niftis_bu_corr_dir2_noisy)
niftis_bd_corrected_trace_noisy = _trace_from_dirs(niftis_bd_corr_dir0_noisy, niftis_bd_corr_dir1_noisy, niftis_bd_corr_dir2_noisy)

# Save one example noise-injected (b=0, dir0) volume per variant for inspection.
# Output lives in the same directory the source was loaded from, named
# ``<source_basename>_noise_injected.nii.gz``, and inherits the source's affine
# so it overlays perfectly on the original in any viewer.
def _save_b0_example(
    stack_n_b_slice_y_x: np.ndarray,
    source_dir: str,
    source_basename: str,
) -> None:
    b0 = _b0_index(Bs_used)
    img = stack_n_b_slice_y_x[b0]                  # (slice, y, x)
    img_yxz = np.transpose(img, (1, 2, 0))         # (y, x, slice)
    src_nifti, _ = load_nifti(os.path.join(source_dir, source_basename))
    out_name = source_basename.replace(".nii.gz", "_noise_injected.nii.gz")
    out_path = os.path.join(source_dir, out_name)
    nib.save(nib.Nifti1Image(img_yxz.astype(np.float32), src_nifti.affine), out_path)
    print(f"  saved {out_path}")

# Sort by b-value, then take the [0] entry so we save the actual b=0/dir0/TE1
# file's basename even if discovery order isn't strictly sorted.
def _b0_filename(paths: list) -> str:
    return sorted(paths, key=b_value)[0]

_save_b0_example(niftis_bu_dir0_noisy,      adc_dir,           _b0_filename(diff_blipup_dir0_paths))
_save_b0_example(niftis_bd_dir0_noisy,      adc_dir,           _b0_filename(diff_blipdown_dir0_paths))
_save_b0_example(niftis_bu_corr_dir0_noisy, adc_corrected_dir, _b0_filename(diff_blipup_corrected_dir0_paths))
_save_b0_example(niftis_bd_corr_dir0_noisy, adc_corrected_dir, _b0_filename(diff_blipdown_corrected_dir0_paths))


# %%===============================
# Process ADC Maps — fit each variant's 3 directions + trace, then save to ./volumes_noised/
# =================================
import json
from datetime import datetime

affine = load_nifti(os.path.join(adc_dir, diff_blipup_dir0_paths[0]))[0].affine

# All noised fit maps go to ./volumes_noised/. Filenames embed subject, variant,
# dir/trace, SNR target, seed, and registration state. JSON sidecar carries the
# remaining provenance (sigmas, source paths, b-values, ...).
out_dir = os.path.join(file_dir, "volumes_noised")
os.makedirs(out_dir, exist_ok=True)

# Trim the trailing modality token (-T2w / -ADCw) from the captured subject so
# the output filenames carry the bare subject id, e.g. ``brainweb-subj04``.
subject_id = re.sub(r"-(T2w|ADCw)$", "", subject)
stamp = f"SNR{int(SNR_TARGET)}_seed{SEED}_reg{'ON' if REGISTER_MASKS else 'OFF'}"

print()
print("--- Fitting ADC on noise-injected stacks ---")

variant_inputs = {
    "blipup": {
        "dir0":  niftis_bu_dir0_noisy,
        "dir1":  niftis_bu_dir1_noisy,
        "dir2":  niftis_bu_dir2_noisy,
        "trace": niftis_bu_trace_noisy,
    },
    "blipdown": {
        "dir0":  niftis_bd_dir0_noisy,
        "dir1":  niftis_bd_dir1_noisy,
        "dir2":  niftis_bd_dir2_noisy,
        "trace": niftis_bd_trace_noisy,
    },
    "blipup_corrected": {
        "dir0":  niftis_bu_corr_dir0_noisy,
        "dir1":  niftis_bu_corr_dir1_noisy,
        "dir2":  niftis_bu_corr_dir2_noisy,
        "trace": niftis_bu_corrected_trace_noisy,
    },
    "blipdown_corrected": {
        "dir0":  niftis_bd_corr_dir0_noisy,
        "dir1":  niftis_bd_corr_dir1_noisy,
        "dir2":  niftis_bd_corr_dir2_noisy,
        "trace": niftis_bd_corrected_trace_noisy,
    },
}
saved_outputs: list[str] = []
for variant_name, inputs in variant_inputs.items():
    for key, stack in inputs.items():
        adc_map_vol = _fit_adc_volume(stack, Bs_used, adc_max=ADC_MAX, label=f"{variant_name} {key}")
        transposed = np.transpose(adc_map_vol, (2, 1, 0))
        rotated = np.flip(np.rot90(transposed, k=1, axes=(0, 1)), axis=0)
        out_name = f"{subject_id}_ADC_{variant_name}_{key}_noise_{stamp}.nii.gz"
        out_path = os.path.join(out_dir, out_name)
        nib.save(nib.Nifti1Image(rotated, affine), out_path)
        saved_outputs.append(out_name)
        print(f"  saved {out_path}")

# JSON sidecar with run provenance.
metadata = {
    "timestamp_utc":     datetime.utcnow().isoformat(timespec="seconds") + "Z",
    "script":            os.path.basename(__file__),
    "subject_id":        subject_id,
    "subject_raw":       subject,
    "modality":          "ADC",
    "SNR_TARGET":        SNR_TARGET,
    "SEED":              SEED,
    "REGISTER_MASKS":    REGISTER_MASKS,
    "TE_echo_ms":        int(TE.replace("TE", "")),
    "b_values":          [int(b) for b in Bs_used],
    "B_MAX":             None if not np.isfinite(B_MAX) else float(B_MAX),
    "ADC_MAX":           float(ADC_MAX),
    "sources": {
        "raw_dir":              adc_dir,
        "corrected_dir":        adc_corrected_dir,
        "first_blipup_dir0":    diff_blipup_dir0_paths[0],
        "first_blipdown_dir0":  diff_blipdown_dir0_paths[0],
        "first_blipup_corr_dir0":   diff_blipup_corrected_dir0_paths[0],
        "first_blipdown_corr_dir0": diff_blipdown_corrected_dir0_paths[0],
    },
    "sigmas": {
        "blipup":             float(sigma_bu),
        "blipdown":           float(sigma_bd),
        "blipup_corrected":   float(sigma_bu_corrected),
        "blipdown_corrected": float(sigma_bd_corrected),
    },
    "outputs": saved_outputs,
}
metadata_path = os.path.join(out_dir, f"{subject_id}_ADC_noise_run_{stamp}.json")
with open(metadata_path, "w") as fh:
    json.dump(metadata, fh, indent=2)
print(f"  saved {metadata_path}")

print()
print(f"Noisy ADC maps written to {out_dir}/ with stamp '{stamp}'. "
      f"b=0/dir0 example weighted volumes (suffix '_noise_injected') live next to their sources.")
