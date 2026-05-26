# IQ-BRAIN is funded by the European Union (MSCA Doctoral Network,
# December 2024–November 2028, Grant Agreement No. 101169519).
# %% ==============================================================================
#   Imports & data loading
# =================================================================================
import glob
import os
import re

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

filepath = os.path.dirname(os.path.abspath(__file__))
corrected_dir = os.path.join(filepath, "volumes_corrected")
noised_dir = os.path.join(filepath, "volumes_noised")

SUBJECT_ID = 0
subjects = np.unique(
    [f.split("-")[1] for f in os.listdir("volumes/") if f.startswith("brainweb-subj")]
)
subject = subjects[SUBJECT_ID]
print(f"Selected subject: {subject}")

REGISTER = True
DISPLAY_AXIS = 2
SLICE_IDX = None


def load_nii(path: str) -> np.ndarray:
    return nib.load(path).get_fdata().astype(np.float32)


def get_slice(arr: np.ndarray, axis: int = DISPLAY_AXIS, idx: int | None = SLICE_IDX) -> np.ndarray:
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr
    if idx is None:
        idx = arr.shape[axis] // 2
    return np.take(arr, idx, axis=axis)


def mae_tissue(im1, im2, mask) -> float:
    """MAE over 3D volume restricted to tissue mask (values in ×10⁻³ mm²/s)."""
    im1 = np.squeeze(im1)
    im2 = np.squeeze(im2)
    mask = np.squeeze(mask)
    valid = (mask > 0) & (im1 > 0) & (im2 > 0)
    if valid.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(im1[valid] - im2[valid])))


# Discover ADC noise stamp from most recent run in volumes_noised/
_adc_noise_matches = sorted(
    glob.glob(os.path.join(noised_dir, f"brainweb-{subject}_ADC_blipup_trace_noise_*.nii.gz"))
)
if not _adc_noise_matches:
    raise FileNotFoundError(
        f"No noisy ADC maps found in {noised_dir}. "
        "Run process_dist_corrected_diff_noise.py first."
    )
_stamp_m = re.search(r"_noise_(SNR\d+_seed\d+_reg\w+)\.nii\.gz$", _adc_noise_matches[0])
adc_stamp = _stamp_m.group(1) if _stamp_m else "SNR10_seed420_regON"
print(f"ADC noise stamp: {adc_stamp}")

# Noiseless volumes — already in ×10⁻³ mm²/s
adcRef = load_nii(f"volumes/brainweb-{subject}-D_ref_volume.nii.gz")
adcMESE_blipup = load_nii(
    f"volumes/brainweb-{subject}-ADC_NLLS_adc_triple_blipup_volume.nii.gz"
)
adcMESE_blipdown = load_nii(
    f"volumes/brainweb-{subject}-ADC_NLLS_adc_triple_blipdown_volume.nii.gz"
)
# Corrected noiseless — raw file is in mm²/s, ×1000 to match reference units
adcMESE_blipup_corrected = (
    load_nii(
        os.path.join(corrected_dir, f"brainweb-{subject}-ADCw_adc_blipup_corrected_trace.nii.gz")
    )
    * 1000
)
adcMESE_blipdown_corrected = (
    load_nii(
        os.path.join(corrected_dir, f"brainweb-{subject}-ADCw_adc_blipdown_corrected_trace.nii.gz")
    )
    * 1000
)

# Noisy volumes — raw file is in mm²/s, ×1e3 to match reference units
adcMESE_blipup_noise = (
    load_nii(
        os.path.join(noised_dir, f"brainweb-{subject}_ADC_blipup_trace_noise_{adc_stamp}.nii.gz")
    )
    * 1e3
)
adcMESE_blipdown_noise = (
    load_nii(
        os.path.join(noised_dir, f"brainweb-{subject}_ADC_blipdown_trace_noise_{adc_stamp}.nii.gz")
    )
    * 1e3
)
adcMESE_blipup_corrected_noise = (
    load_nii(
        os.path.join(
            noised_dir,
            f"brainweb-{subject}_ADC_blipup_corrected_trace_noise_{adc_stamp}.nii.gz",
        )
    )
    * 1e3
)
adcMESE_blipdown_corrected_noise = (
    load_nii(
        os.path.join(
            noised_dir,
            f"brainweb-{subject}_ADC_blipdown_corrected_trace_noise_{adc_stamp}.nii.gz",
        )
    )
    * 1e3
)

# %% ==============================================================================
#   Registration — all images to Reference (3D rigid, ANTs)
# =================================================================================
adcRef_reg = np.squeeze(adcRef)

if REGISTER:
    import ants

    _ref_ants = ants.image_read(f"volumes/brainweb-{subject}-D_ref_volume.nii.gz")

    def _reg(path: str, scale: float = 1.0) -> np.ndarray:
        return (
            ants.registration(
                _ref_ants,
                ants.image_read(path),
                type_of_transform="Rigid",
            )["warpedmovout"].numpy()
            * scale
        )

    adcMESE_blipup_reg = _reg(
        f"volumes/brainweb-{subject}-ADC_NLLS_adc_triple_blipup_volume.nii.gz"
    )
    adcMESE_blipdown_reg = _reg(
        f"volumes/brainweb-{subject}-ADC_NLLS_adc_triple_blipdown_volume.nii.gz"
    )
    adcMESE_blipup_corrected_reg = _reg(
        os.path.join(corrected_dir, f"brainweb-{subject}-ADCw_adc_blipup_corrected_trace.nii.gz"),
        scale=1000,
    )
    adcMESE_blipdown_corrected_reg = _reg(
        os.path.join(
            corrected_dir, f"brainweb-{subject}-ADCw_adc_blipdown_corrected_trace.nii.gz"
        ),
        scale=1000,
    )
    adcMESE_blipup_noise_reg = _reg(
        os.path.join(noised_dir, f"brainweb-{subject}_ADC_blipup_trace_noise_{adc_stamp}.nii.gz"),
        scale=1e3,
    )
    adcMESE_blipdown_noise_reg = _reg(
        os.path.join(
            noised_dir, f"brainweb-{subject}_ADC_blipdown_trace_noise_{adc_stamp}.nii.gz"
        ),
        scale=1e3,
    )
    adcMESE_blipup_corrected_noise_reg = _reg(
        os.path.join(
            noised_dir,
            f"brainweb-{subject}_ADC_blipup_corrected_trace_noise_{adc_stamp}.nii.gz",
        ),
        scale=1e3,
    )
    adcMESE_blipdown_corrected_noise_reg = _reg(
        os.path.join(
            noised_dir,
            f"brainweb-{subject}_ADC_blipdown_corrected_trace_noise_{adc_stamp}.nii.gz",
        ),
        scale=1e3,
    )
    print("Registration complete.")
else:
    adcMESE_blipup_reg = np.squeeze(adcMESE_blipup)
    adcMESE_blipdown_reg = np.squeeze(adcMESE_blipdown)
    adcMESE_blipup_corrected_reg = np.squeeze(adcMESE_blipup_corrected)
    adcMESE_blipdown_corrected_reg = np.squeeze(adcMESE_blipdown_corrected)
    adcMESE_blipup_noise_reg = np.squeeze(adcMESE_blipup_noise)
    adcMESE_blipdown_noise_reg = np.squeeze(adcMESE_blipdown_noise)
    adcMESE_blipup_corrected_noise_reg = np.squeeze(adcMESE_blipup_corrected_noise)
    adcMESE_blipdown_corrected_noise_reg = np.squeeze(adcMESE_blipdown_corrected_noise)
    print("Registration skipped — using raw images.")

# %% ==============================================================================
#   Tissue masks
# =================================================================================
tissue_mask_wm = load_nii(f"masks/brainweb-{subject}-mask_wm_volume.nii.gz")
tissue_mask_gm = load_nii(f"masks/brainweb-{subject}-mask_gm_volume.nii.gz")
tissue_mask_csf = load_nii(f"masks/brainweb-{subject}-mask_csf_volume.nii.gz")

tissue_masks_arr = np.stack(
    [tissue_mask_wm, tissue_mask_gm, tissue_mask_csf], axis=0
).astype(np.float32)
tissue_names = ["WM", "GM", "CSF"]
n_tissues = len(tissue_names)

# %% ==============================================================================
#   Per-tissue MAE vs Reference — 8 individual conditions (3D registered)
# =================================================================================
# Ordered as: blip-down, blip-up, blip-down noise, blip-up noise,
#             blip-down corrected, blip-up corrected, blip-down corrected noise, blip-up corrected noise
conditions = {
    "MSE blip-down": adcMESE_blipdown_reg,
    "MSE blip-up": adcMESE_blipup_reg,
    "MSE blip-down\n(noise SNR)": adcMESE_blipdown_noise_reg,
    "MSE blip-up\n(noise SNR)": adcMESE_blipup_noise_reg,
    "MSE blip-down\ncorrected": adcMESE_blipdown_corrected_reg,
    "MSE blip-up\ncorrected": adcMESE_blipup_corrected_reg,
    "MSE blip-down\ncorrected (noise)": adcMESE_blipdown_corrected_noise_reg,
    "MSE blip-up\ncorrected (noise)": adcMESE_blipup_corrected_noise_reg,
}

mae_per_condition = {
    name: [mae_tissue(img, adcRef_reg, tissue_masks_arr[t]) for t in range(n_tissues)]
    for name, img in conditions.items()
}

# Print table
col_w = 12
cond_names_flat = [n.replace("\n", " ") for n in conditions]
print(f"\nPer-tissue ADC MAE vs Reference (×10⁻³ mm²/s) — 3D registered")
print(f"{'Condition':<45}" + "".join(f"{t:>{col_w}}" for t in tissue_names))
print("-" * (45 + col_w * n_tissues))
for name, vals in zip(cond_names_flat, mae_per_condition.values()):
    print(f"{name:<45}" + "".join(f"{v:>{col_w}.4f}" for v in vals))

x = np.arange(n_tissues)
n_cond = len(conditions)
width = 0.8 / n_cond
offsets = np.arange(n_cond) * width - (n_cond - 1) * width / 2
colors = plt.get_cmap("tab10")(np.linspace(0, 0.9, n_cond))

fig, ax = plt.subplots(figsize=(14, 6))
for idx, (name, vals) in enumerate(mae_per_condition.items()):
    bars = ax.bar(x + offsets[idx], vals, width, label=name.replace("\n", " "), color=colors[idx])
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=90,
            )
ax.set_xticks(x)
ax.set_xticklabels(tissue_names, fontsize=12)
ax.set_ylabel("MAE (×10⁻³ mm²/s)")
ax.set_title("ADC MAE per tissue vs Reference — all 8 conditions (3D registered)")
ax.legend(fontsize=7, ncol=2, loc="upper right")
plt.tight_layout()
plt.savefig("adc_per_tissue_mae_all_conditions.png", dpi=300, bbox_inches="tight")
plt.show()

# %% ==============================================================================
#   Averaged blip-up / blip-down MAE per tissue — 4 grouped methods
# =================================================================================
avg_methods = {
    "MSE EPI": (adcMESE_blipup_reg, adcMESE_blipdown_reg),
    "MSE EPI (noise SNR)": (adcMESE_blipup_noise_reg, adcMESE_blipdown_noise_reg),
    "MSE EPI corrected": (adcMESE_blipup_corrected_reg, adcMESE_blipdown_corrected_reg),
    "MSE EPI corrected\n(noise SNR)": (
        adcMESE_blipup_corrected_noise_reg,
        adcMESE_blipdown_corrected_noise_reg,
    ),
}

mae_avg = {}
for name, (img_up, img_down) in avg_methods.items():
    vals = []
    for t in range(n_tissues):
        mask = tissue_masks_arr[t]
        m_up = mae_tissue(img_up, adcRef_reg, mask)
        m_down = mae_tissue(img_down, adcRef_reg, mask)
        vals.append((m_up + m_down) / 2)
    mae_avg[name] = vals

print(f"\nPer-tissue ADC MAE vs Reference (×10⁻³ mm²/s) — blip-up/blip-down averaged")
print(f"{'Method':<40}" + "".join(f"{t:>{col_w}}" for t in tissue_names))
print("-" * (40 + col_w * n_tissues))
for name, vals in mae_avg.items():
    print(f"{name.replace(chr(10), ' '):<40}" + "".join(f"{v:>{col_w}.4f}" for v in vals))

n_avg = len(mae_avg)
width_a = 0.6 / n_avg
offsets_a = np.arange(n_avg) * width_a - (n_avg - 1) * width_a / 2
colors_a = plt.get_cmap("plasma")(np.linspace(0.1, 0.9, n_avg))

fig, ax_a = plt.subplots(figsize=(9, 5))
for idx, (name, vals) in enumerate(mae_avg.items()):
    bars = ax_a.bar(
        x + offsets_a[idx], vals, width_a, label=name, color=colors_a[idx]
    )
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax_a.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
ax_a.set_xticks(x)
ax_a.set_xticklabels(tissue_names, fontsize=12)
ax_a.set_ylabel("MAE (×10⁻³ mm²/s)")
ax_a.set_title(
    "ADC MAE per tissue vs Reference\n(blip-up / blip-down averaged, 3D registered)"
)
ax_a.legend(fontsize=8)
plt.tight_layout()
plt.savefig("adc_per_tissue_mae_averaged_blipupdown.png", dpi=300, bbox_inches="tight")
plt.show()

# %% ==============================================================================
#   Mean ADC per tissue — Reference + all 8 conditions (3D registered)
# =================================================================================
def mean_tissue(img, mask) -> float:
    """Mean value over 3D volume restricted to tissue mask, non-zero voxels only."""
    img = np.squeeze(img)
    mask = np.squeeze(mask)
    valid = (mask > 0) & (img > 0)
    if valid.sum() == 0:
        return float("nan")
    return float(np.mean(img[valid]))


all_acqs = {
    "Reference": (adcRef_reg, None),
    "MSE EPI": (adcMESE_blipup_reg, adcMESE_blipdown_reg),
    "MSE EPI\n(noise SNR)": (adcMESE_blipup_noise_reg, adcMESE_blipdown_noise_reg),
    "MSE EPI\ncorrected": (adcMESE_blipup_corrected_reg, adcMESE_blipdown_corrected_reg),
    "MSE EPI\ncorrected (noise SNR)": (
        adcMESE_blipup_corrected_noise_reg,
        adcMESE_blipdown_corrected_noise_reg,
    ),
}

mean_per_acq = {}
for name, (img_a, img_b) in all_acqs.items():
    vals = []
    for t in range(n_tissues):
        mask = tissue_masks_arr[t]
        m_a = mean_tissue(img_a, mask)
        if img_b is not None:
            m_b = mean_tissue(img_b, mask)
            vals.append((m_a + m_b) / 2)
        else:
            vals.append(m_a)
    mean_per_acq[name] = vals

# Print table
print(f"\nMean ADC per tissue (×10⁻³ mm²/s) — 3D registered")
print(f"{'Approach':<45}" + "".join(f"{t:>{col_w}}" for t in tissue_names))
print("-" * (45 + col_w * n_tissues))
for name, vals in mean_per_acq.items():
    print(f"{name.replace(chr(10), ' '):<45}" + "".join(f"{v:>{col_w}.4f}" for v in vals))

n_acqs = len(mean_per_acq)
width_m = 0.6 / n_acqs
offsets_m = np.arange(n_acqs) * width_m - (n_acqs - 1) * width_m / 2

# Reference gets a distinct colour (dark grey); rest use plasma
ref_color = np.array([[0.15, 0.15, 0.15, 1.0]])
method_colors = plt.get_cmap("plasma")(np.linspace(0.1, 0.9, n_acqs - 1))
colors_m = np.vstack([ref_color, method_colors])

fig, ax_m = plt.subplots(figsize=(10, 5))
for idx, (name, vals) in enumerate(mean_per_acq.items()):
    bars = ax_m.bar(
        x + offsets_m[idx], vals, width_m,
        label=name.replace("\n", " "),
        color=colors_m[idx],
    )
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax_m.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

ax_m.set_xticks(x)
ax_m.set_xticklabels(tissue_names, fontsize=12)
ax_m.set_ylabel("Mean ADC (×10⁻³ mm²/s)")
ax_m.set_title(
    "Mean ADC per tissue — Reference vs all conditions\n(blip-up / blip-down averaged, 3D registered)"
)
ax_m.legend(fontsize=8, ncol=2, loc="upper right")
plt.tight_layout()
plt.savefig("adc_per_tissue_mean_all_conditions.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
