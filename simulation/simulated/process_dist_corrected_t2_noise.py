# IQ-BRAIN is funded by the European Union (MSCA Doctoral Network,
# December 2024–November 2028, Grant Agreement No. 101169519).
#
# Noise-injected counterpart of ``process_dist_corrected_t2.py``.
#
# Same I/O contract — load blipup / blipdown / corrected variants of the
# T2-weighted EPI volumes, fit T2 with ``create_t2_map`` per slice, save the
# resulting maps — but injects image-domain Rician noise just before the fit.
#
# sigma is calibrated *once per stack variant* from the TE1 (shortest TE)
# volume's white-matter mean magnitude:
#
#     sigma = mean(S_TE1[WM_mask]) / SNR_TARGET
#
# The TE1 echo has the highest SNR of the T2 series, so calibrating from it
# matches the convention used by hardware QA noise studies. The same scalar
# sigma is then reused for every TE in that variant (echoes are not noisier
# than each other in the receiver chain — only their signal is lower).
#
# Outputs:
#   - {subject}_t2_{blip}_map_noise.nii.gz                (raw, noisy fit)
#   - {subject}_t2_{blip}_map_corrected_noise.nii.gz      (corrected, noisy fit)
#   - {subject}_t2w_TE1_{blip}_noise_injected.nii.gz      (example noise-injected TE1 volume)
#
# Author      : Aron Gimesi <aron.gimesi@tecnico.ulisboa.pt>
# Affiliation : Instituto Superior Técnico | MSCA-DN IQ-BRAIN
# Date        : 2026
# Context     : ESMRMB 2026 — Pulseq DiffusionMESE showcase

# %% ==============================================================================
#   Imports
# =================================================================================
import os
import re
import sys

import nibabel as nib
import numpy as np

# Sibling-package imports (utils_relaxometry lives one level up).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_DIR  = os.path.dirname(_THIS_DIR)
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from utils_relaxometry import create_t2_map
from utils_noise import (
    add_rician_noise,
    compute_sigma,
    load_tissue_masks,
    register_volume_to_reference,
)


# =================================================================================
# Config
# =================================================================================
SNR_TARGET     = 1.0     # SNR_TARGET ~> sigma = WM_mean / SNR_TARGET on TE1
SEED           = 420        # base seed; each stack variant gets a derived sub-seed
REGISTER_MASKS = True    # if True, ANTs-Rigid-register tissue masks to each
                          # variant's TE1 magnitude volume before sigma
                          # calibration and per-tissue stats. Default off; the
                          # raw blipup/blipdown volumes are aligned with the
                          # masks by construction. Enable when the corrected
                          # (topup-warped) variants show residual misalignment.
FOREGROUND_FRAC = 0.05    # foreground mask threshold as fraction of TE1 max.
                          # Voxels below this in the noise-free TE1 volume are
                          # treated as "outside the object" — noise is NOT
                          # added there, and the fitted T2 is forced to 0 in
                          # those voxels. Prevents the topup-warped background
                          # haze from being fit to T2 = t2_bounds[1] (the
                          # saturation hits hardest at SNR>=40). Set to 0 to
                          # disable foreground masking and add noise everywhere.


# =================================================================================
# Helper Functions
# =================================================================================
def load_nifti(file_path):
    """Load a NIfTI file and return the image data and affine."""
    nifti = nib.load(file_path)
    data = nifti.get_fdata()
    return nifti, data


def get_TEs(filepaths):
    """Extract echo times (TEs) from the filenames; return in seconds, sorted."""
    TEs = []
    for path in filepaths:
        basename = os.path.basename(path)
        te_str = basename.split("_")[-2]  # Adjust index based on filename structure
        te_value = float(te_str.replace("TE", ""))
        TEs.append(te_value)
    return np.array(sorted(TEs)) * 0.001


def te_value(f):
    m = re.search(r"TE(\d+)", f)
    return int(m.group(1)) if m else float("inf")


def _calibrate_sigma_from_te1(
    resampled_stack: np.ndarray,
    wm_mask_yxz: np.ndarray,
    snr_target: float,
    label: str,
) -> float:
    """Compute one scalar Rician sigma from the WM mean of the TE1 volume.

    ``resampled_stack`` has shape ``(n_te, slice, y, x)`` matching the original
    dist-corrected script's transpose convention. TE1 is index 0 (the list is
    sorted ascending). The mask is supplied in ``(y, x, slice)`` and is
    transposed to ``(slice, y, x)`` internally.
    """
    te1_volume = resampled_stack[0]  # (slice, y, x)
    wm_mask_slice_first = np.transpose(wm_mask_yxz, (2, 0, 1))  # (slice, y, x)
    if te1_volume.shape != wm_mask_slice_first.shape:
        raise ValueError(
            f"{label}: TE1 volume {te1_volume.shape} vs WM mask "
            f"{wm_mask_slice_first.shape}"
        )
    # Restrict the mean to slices where the acquisition actually produced
    # signal (the on-disk volumes may have all-zero slices from partial runs).
    active = (te1_volume.sum(axis=(1, 2)) > 0)
    if not active.any():
        raise ValueError(f"{label}: TE1 volume has no non-empty slices")
    wm_active = wm_mask_slice_first & active[:, None, None]
    print(f"[T2-noise] {label}: TE1 active slices = "
          f"{int(active.sum())}/{te1_volume.shape[0]}")
    return compute_sigma(te1_volume, wm_active, snr_target)


def _register_masks_to_te1(
    masks_yxz: dict,
    resampled_stack: np.ndarray,
    label: str,
) -> dict:
    """Volume-level (true 3-D) rigid registration of tissue masks to TE1.

    Both inputs are conceptually in ``(y, x, slice)`` form. ``resampled_stack``
    has shape ``(n_te, slice, y, x)`` — we transpose its TE1 slice to
    ``(y, x, slice)`` so the moving mask and the reference live on the same
    grid. A single 3-D rigid transform is estimated on the full volume (not
    per-slice) via :func:`utils_noise.register_volume_to_reference`, so any
    out-of-plane drift is corrected too.

    Resampled probability values are thresholded at 0.5 to recover a boolean
    mask. WM is the primary contrast anchor; GM and CSF are registered
    independently against the same TE1 reference for consistency.
    """
    te1_yxz = np.transpose(resampled_stack[0], (1, 2, 0)).astype(np.float32)
    out = {}
    for tissue, mask in masks_yxz.items():
        moving = mask.astype(np.float32)
        warped = register_volume_to_reference(moving, te1_yxz, type_of_transform="Rigid")
        out[tissue] = warped >= 0.5
        moved = int(np.abs(out[tissue].astype(int) - mask.astype(int)).sum())
        print(f"[T2-noise] {label}: registered {tissue} mask (3-D Rigid), "
              f"voxel-flip vs. unregistered = {moved}")
    return out


def _apply_rician_noise_stack(stack: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Add Rician noise to every (TE, slice, y, x) sample with a single sigma."""
    rng = np.random.default_rng(seed)
    return add_rician_noise(stack, sigma, rng)


def _foreground_mask_from_te1(
    resampled_stack: np.ndarray,
    frac: float,
    label: str,
) -> np.ndarray:
    """Boolean (slice, y, x) mask of voxels with noise-free TE1 above ``frac * max``.

    Defines the "object" region that genuinely carries signal. Used to clamp
    noise injection and to zero the fitted T2 map in the background. The mask
    is built from the *noise-free* TE1 volume so it is identical across
    Monte-Carlo realisations.
    """
    if frac <= 0.0:
        return np.ones(resampled_stack.shape[1:], dtype=bool)
    te1 = resampled_stack[0]
    thr = frac * float(te1.max())
    fg = te1 > thr
    print(f"[T2-noise] {label}: foreground mask "
          f"({int(fg.sum())}/{fg.size} voxels, threshold={thr:.3g})")
    return fg


def _fit_t2_volume(stack_resampled: np.ndarray, TEs: np.ndarray, t2_bounds, label: str):
    """Replicate the original dist-corrected per-slice fit loop."""
    t2_volume = []
    s0_volume = []
    for i in range(stack_resampled.shape[1]):
        print(
            f"Processing {label} slice {i+1}/{stack_resampled.shape[1]}...",
            end="\r", flush=True,
        )
        t2_map, s0_map = create_t2_map(stack_resampled[:, i, :, :], TEs, t2_bounds=t2_bounds)
        t2_volume.append(t2_map)
        s0_volume.append(s0_map)
    print()
    return np.stack(t2_volume, axis=0), np.stack(s0_volume, axis=0)


# =================================================================================
# Discover Files and Prepare TE List
# =================================================================================
file_dir = _THIS_DIR

fieldmap_dir = rf"{file_dir}/topup_results (FIELDMAPS)"
t2_dir = rf"{file_dir}/t2_vol"
t2_corrected_dir = rf"{fieldmap_dir}/t2_volumes_corrected_same"


# Filter out files with TE in [0, 5] ms — non-physical and only present as a
# pre-bug-fix artifact from sim_t2_triple's stem-rounding bug (every triple-SE
# file was saved as TE0_*_e{n} because the TE-in-seconds was rounded to int).
# Including them would make TE=0 the calibration anchor and corrupt the fit.
def _physical_te(f):
    return te_value(f) >= 10

t2_blipdown_paths = sorted(
    [i for i in os.listdir(rf"{t2_dir}") if i.endswith("blipdown.nii.gz") and _physical_te(i)], key=te_value
)
t2_blipdown_corrected_paths = sorted(
    [i for i in os.listdir(rf"{t2_corrected_dir}") if i.endswith("blipdown_corrected.nii.gz") and _physical_te(i)], key=te_value
)

t2_blipup_paths = sorted(
    [i for i in os.listdir(rf"{t2_dir}") if i.endswith("blipup.nii.gz") and _physical_te(i)], key=te_value
)
t2_blipup_corrected_paths = sorted(
    [i for i in os.listdir(rf"{t2_corrected_dir}") if i.endswith("blipup_corrected.nii.gz") and _physical_te(i)], key=te_value
)

TEs = get_TEs(t2_blipup_paths)
print(f"Echo times (TEs) extracted: {TEs}")
print(f"TE1 (calibration anchor) = {TEs[0]:.4f} s")

# Subject prefix: everything before "_TE{n}_" in the first input filename
_subj_match = re.match(r"^(.*?)_TE\d+_", os.path.basename(t2_blipup_paths[0]))
subject = _subj_match.group(1) if _subj_match else "subject"
print(f"Subject: {subject}")

print(
    f"Found {len(t2_blipup_paths)} blipup and {len(t2_blipdown_paths)} blipdown files."
)
print(f"Found {len(t2_blipup_corrected_paths)} corrected blipup and {len(t2_blipdown_corrected_paths)} corrected blipdown files.")

# Tissue masks (WM/GM/CSF). The on-disk masks are already aligned with the
# weighted volumes — no reorient needed; see utils_noise.load_tissue_masks doc.
masks_dir = os.path.join(file_dir, "masks")
phantom_name = "-".join(subject.split("-")[:2]) if subject.startswith("brainweb-") else f"brainweb-{subject}"
tissue_masks = load_tissue_masks(masks_dir, phantom_name, ("wm", "gm", "csf"))
print(f"Tissue masks loaded   shape={tissue_masks['wm'].shape}   "
      f"|WM|={int(tissue_masks['wm'].sum())}")


# %%===============================
# Load NIfTI Volumes (Blipup / Blipdown / corrected)
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
# Transpose to (n_te, slice, y, x)
# =================================
niftis_bu_resampled            = np.transpose(niftis_bu,            (3, 2, 0, 1))
niftis_bd_resampled            = np.transpose(niftis_bd,            (3, 2, 0, 1))
niftis_bd_corrected_resampled  = np.transpose(niftis_bd_corrected,  (3, 2, 0, 1))
niftis_bu_corrected_resampled  = np.transpose(niftis_bu_corrected,  (3, 2, 0, 1))
print(f"Resampled blipup shape: {niftis_bu_resampled.shape}")
print(f"Resampled blipdown shape: {niftis_bd_resampled.shape}")
print(f"Resampled corrected blipdown shape: {niftis_bd_corrected_resampled.shape}")
print(f"Resampled corrected blipup shape: {niftis_bu_corrected_resampled.shape}")


# %%===============================
# Optional: ANTs-Rigid-register tissue masks to each variant's TE1 reference
# =================================
if REGISTER_MASKS:
    print()
    print("--- Registering tissue masks to TE1 (ANTs Rigid, per-slice) ---")
    masks_bu           = _register_masks_to_te1(tissue_masks, niftis_bu_resampled,           "blipup (raw)")
    masks_bd           = _register_masks_to_te1(tissue_masks, niftis_bd_resampled,           "blipdown (raw)")
    masks_bu_corrected = _register_masks_to_te1(tissue_masks, niftis_bu_corrected_resampled, "blipup (corrected)")
    masks_bd_corrected = _register_masks_to_te1(tissue_masks, niftis_bd_corrected_resampled, "blipdown (corrected)")
else:
    masks_bu = masks_bd = masks_bu_corrected = masks_bd_corrected = tissue_masks


# %%===============================
# Calibrate sigma from TE1 (WM mean) for each stack variant
# =================================
print()
print(f"--- Calibrating sigma at SNR_TARGET={SNR_TARGET} from TE1={TEs[0]*1e3:.0f}ms WM mean ---")
sigma_bu           = _calibrate_sigma_from_te1(niftis_bu_resampled,           masks_bu["wm"],           SNR_TARGET, "blipup (raw)")
sigma_bd           = _calibrate_sigma_from_te1(niftis_bd_resampled,           masks_bd["wm"],           SNR_TARGET, "blipdown (raw)")
sigma_bu_corrected = _calibrate_sigma_from_te1(niftis_bu_corrected_resampled, masks_bu_corrected["wm"], SNR_TARGET, "blipup (corrected)")
sigma_bd_corrected = _calibrate_sigma_from_te1(niftis_bd_corrected_resampled, masks_bd_corrected["wm"], SNR_TARGET, "blipdown (corrected)")


# %%===============================
# Foreground masks (per variant, from the noise-free TE1 volume)
# =================================
print()
print(f"--- Building per-variant foreground masks at FOREGROUND_FRAC={FOREGROUND_FRAC} ---")
fg_bu           = _foreground_mask_from_te1(niftis_bu_resampled,           FOREGROUND_FRAC, "blipup (raw)")
fg_bd           = _foreground_mask_from_te1(niftis_bd_resampled,           FOREGROUND_FRAC, "blipdown (raw)")
fg_bu_corrected = _foreground_mask_from_te1(niftis_bu_corrected_resampled, FOREGROUND_FRAC, "blipup (corrected)")
fg_bd_corrected = _foreground_mask_from_te1(niftis_bd_corrected_resampled, FOREGROUND_FRAC, "blipdown (corrected)")


# %%===============================
# Inject Rician noise (single realization per variant, reproducible via SEED)
# Noise is added everywhere physically, then the foreground mask zeros voxels
# outside the object so the fitter sees a clean zero background (otherwise
# the topup-warped background haze + noise floor get fit to t2_bounds[1]).
# =================================
print()
print("--- Injecting Rician noise into every TE volume of each variant ---")
niftis_bu_noisy           = _apply_rician_noise_stack(niftis_bu_resampled,           sigma_bu,           SEED + 1) * fg_bu[None, ...]
niftis_bd_noisy           = _apply_rician_noise_stack(niftis_bd_resampled,           sigma_bd,           SEED + 2) * fg_bd[None, ...]
niftis_bu_corrected_noisy = _apply_rician_noise_stack(niftis_bu_corrected_resampled, sigma_bu_corrected, SEED + 3) * fg_bu_corrected[None, ...]
niftis_bd_corrected_noisy = _apply_rician_noise_stack(niftis_bd_corrected_resampled, sigma_bd_corrected, SEED + 4) * fg_bd_corrected[None, ...]

# Save one example noise-injected TE1 volume per variant for visual inspection.
# Output lives in the same directory the source was loaded from, named
# ``<source_basename>_noise_injected.nii.gz``, and inherits the source's affine
# so it overlays perfectly on the original in any viewer.
def _save_te1_example(
    stack_sliceyx: np.ndarray,
    source_dir: str,
    source_basename: str,
) -> None:
    te1 = stack_sliceyx[0]                          # (slice, y, x)
    te1_yxz = np.transpose(te1, (1, 2, 0))          # (y, x, slice)
    src_nifti, _ = load_nifti(os.path.join(source_dir, source_basename))
    out_name = source_basename.replace(".nii.gz", "_noise_injected.nii.gz")
    out_path = os.path.join(source_dir, out_name)
    nib.save(nib.Nifti1Image(te1_yxz.astype(np.float32), src_nifti.affine), out_path)
    print(f"  saved {out_path}")

_save_te1_example(niftis_bu_noisy,           t2_dir,           t2_blipup_paths[0])
_save_te1_example(niftis_bd_noisy,           t2_dir,           t2_blipdown_paths[0])
_save_te1_example(niftis_bu_corrected_noisy, t2_corrected_dir, t2_blipup_corrected_paths[0])
_save_te1_example(niftis_bd_corrected_noisy, t2_corrected_dir, t2_blipdown_corrected_paths[0])


# %%===============================
# Process T2 Maps (noisy)
# =================================
affine = load_nifti(os.path.join(t2_dir, t2_blipdown_paths[0]))[0].affine

print()
print("--- Fitting T2 on noise-injected stacks ---")
t2_blipup_map_noise, _ = _fit_t2_volume(
    niftis_bu_noisy, TEs, t2_bounds=(0.0, 3.0), label="blipup (noisy)"
)
t2_blipup_map_noise *= fg_bu                       # zero outside the foreground
print(f"Blipup T2 map shape: {t2_blipup_map_noise.shape}")
t2_blipup_map_transposed = np.transpose(t2_blipup_map_noise, (2, 1, 0))

t2_blipdown_map_noise, _ = _fit_t2_volume(
    niftis_bd_noisy, TEs, t2_bounds=(0.0, 2.2), label="blipdown (noisy)"
)
t2_blipdown_map_noise *= fg_bd
print(f"Blipdown T2 map shape: {t2_blipdown_map_noise.shape}")
t2_blipdown_map_transposed = np.transpose(t2_blipdown_map_noise, (2, 1, 0))

t2_blipdown_map_corrected_noise, _ = _fit_t2_volume(
    niftis_bd_corrected_noisy, TEs, t2_bounds=(0.0, 2.2), label="blipdown_corrected (noisy)"
)
t2_blipdown_map_corrected_noise *= fg_bd_corrected
print(f"Corrected blipdown T2 map shape: {t2_blipdown_map_corrected_noise.shape}")
t2_blipdown_map_corrected_transposed = np.transpose(t2_blipdown_map_corrected_noise, (2, 1, 0))

t2_blipup_map_corrected_noise, _ = _fit_t2_volume(
    niftis_bu_corrected_noisy, TEs, t2_bounds=(0.0, 2.2), label="blipup_corrected (noisy)"
)
t2_blipup_map_corrected_noise *= fg_bu_corrected
print(f"Corrected blipup T2 map shape: {t2_blipup_map_corrected_noise.shape}")
t2_blipup_map_corrected_transposed = np.transpose(t2_blipup_map_corrected_noise, (2, 1, 0))


# %%===============================
# Save Output (rotated to match the conventional in-plane orientation)
# =================================
t2_blipup_map_rot              = np.flip(np.rot90(t2_blipup_map_transposed,             k=1, axes=(0, 1)), axis=(0))
t2_blipdown_map_rot            = np.flip(np.rot90(t2_blipdown_map_transposed,           k=1, axes=(0, 1)), axis=(0))
t2_blipup_map_corrected_rot    = np.flip(np.rot90(t2_blipup_map_corrected_transposed,   k=1, axes=(0, 1)), axis=(0))
t2_blipdown_map_corrected_rot  = np.flip(np.rot90(t2_blipdown_map_corrected_transposed, k=1, axes=(0, 1)), axis=(0))

# All noised fit maps go to ./volumes_noised/. Filenames embed subject, modality,
# variant, SNR target, seed, and registration state for unambiguous provenance.
# The accompanying JSON sidecar logs everything else (sigmas, paths, TE list, ...).
import json
from datetime import datetime

out_dir = os.path.join(file_dir, "volumes_noised")
os.makedirs(out_dir, exist_ok=True)

# Trim the trailing modality token (-T2w / -ADCw) from the captured subject so
# the output filenames carry the bare subject id, e.g. ``brainweb-subj04``.
subject_id = re.sub(r"-(T2w|ADCw)$", "", subject)
stamp = f"SNR{int(SNR_TARGET)}_seed{SEED}_reg{'ON' if REGISTER_MASKS else 'OFF'}"

_map_outputs = {
    "blipup":             (t2_blipup_map_rot,             sigma_bu,           (0.0, 3.0)),
    "blipdown":           (t2_blipdown_map_rot,           sigma_bd,           (0.0, 2.2)),
    "blipup_corrected":   (t2_blipup_map_corrected_rot,   sigma_bu_corrected, (0.0, 2.2)),
    "blipdown_corrected": (t2_blipdown_map_corrected_rot, sigma_bd_corrected, (0.0, 2.2)),
}
for variant, (map_rot, sigma_v, bounds) in _map_outputs.items():
    out_path = os.path.join(out_dir, f"{subject_id}_T2_{variant}_noise_{stamp}.nii.gz")
    nib.save(nib.Nifti1Image(map_rot, affine), out_path)
    print(f"  saved {out_path}")

# JSON sidecar with the full run provenance — sigmas, TE list, source paths,
# active-slice ranges, registration state, RNG seed. Open in any editor.
metadata = {
    "timestamp_utc":     datetime.utcnow().isoformat(timespec="seconds") + "Z",
    "script":            os.path.basename(__file__),
    "subject_id":        subject_id,
    "subject_raw":       subject,
    "modality":          "T2",
    "SNR_TARGET":        SNR_TARGET,
    "SEED":              SEED,
    "REGISTER_MASKS":    REGISTER_MASKS,
    "FOREGROUND_FRAC":   FOREGROUND_FRAC,
    "TEs_seconds":       TEs.tolist(),
    "TE1_anchor_s":      float(TEs[0]),
    "n_TEs":             int(TEs.size),
    "sources": {
        "raw_dir":              t2_dir,
        "corrected_dir":        t2_corrected_dir,
        "first_blipup":         t2_blipup_paths[0],
        "first_blipdown":       t2_blipdown_paths[0],
        "first_blipup_corr":    t2_blipup_corrected_paths[0],
        "first_blipdown_corr":  t2_blipdown_corrected_paths[0],
    },
    "sigmas": {
        "blipup":             float(sigma_bu),
        "blipdown":           float(sigma_bd),
        "blipup_corrected":   float(sigma_bu_corrected),
        "blipdown_corrected": float(sigma_bd_corrected),
    },
    "t2_bounds_seconds": {
        "blipup":             [0.0, 3.0],
        "blipdown":           [0.0, 3],
        "blipup_corrected":   [0.0, 3],
        "blipdown_corrected": [0.0, 3],
    },
    "outputs": [
        f"{subject_id}_T2_{variant}_noise_{stamp}.nii.gz"
        for variant in _map_outputs.keys()
    ],
}
metadata_path = os.path.join(out_dir, f"{subject_id}_T2_noise_run_{stamp}.json")
with open(metadata_path, "w") as fh:
    json.dump(metadata, fh, indent=2)
print(f"  saved {metadata_path}")

print()
print(f"Noisy T2 maps written to {out_dir}/ with stamp '{stamp}'. "
      f"TE1 example weighted volumes (suffix '_noise_injected') live next to their sources.")
