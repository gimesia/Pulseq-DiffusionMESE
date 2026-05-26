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
noised_dir = os.path.join(filepath, "volumes_noised")

SUBJECT_ID = 0
subjects = np.unique(
    [f.split("-")[1] for f in os.listdir("volumes/") if f.startswith("brainweb-subj")]
)
subject = subjects[SUBJECT_ID]
print(f"Selected subject: {subject}")

REGISTER = True
DISPLAY_AXIS = 2  # axis along which to slice for 2D display (0=x, 1=y, 2=z)
SLICE_IDX = None  # None → middle slice


def load_nii(path: str) -> np.ndarray:
    return nib.load(path).get_fdata().astype(np.float32)


def get_slice(
    arr: np.ndarray, axis: int = DISPLAY_AXIS, idx: int | None = SLICE_IDX
) -> np.ndarray:
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr
    if idx is None:
        idx = arr.shape[axis] // 2
    return np.take(arr, idx, axis=axis)


def mae(im1, im2, non_zero=True) -> float:
    im1 = np.squeeze(im1)
    im2 = np.squeeze(im2)
    if non_zero:
        non_zero_mask = (im1 > 0) & (im2 > 0)
        return np.mean(np.abs(im1[non_zero_mask] - im2[non_zero_mask]))
    else:
        return np.mean(np.abs(im1 - im2))


# Discover the noise stamp from the most recent run in volumes_noised/
_adc_noise_matches = sorted(
    glob.glob(os.path.join(noised_dir, f"brainweb-{subject}_ADC_blipup_trace_noise_*.nii.gz"))
)
if not _adc_noise_matches:
    raise FileNotFoundError(
        f"No noisy ADC maps found in {noised_dir}. "
        "Run process_dist_corrected_diff_noise.py first."
    )
_adc_stamp_m = re.search(
    r"_noise_(SNR\d+_seed\d+_reg\w+)\.nii\.gz$", _adc_noise_matches[-1]
)
adc_stamp = _adc_stamp_m.group(1) if _adc_stamp_m else "SNR40_seed420_regON"
print(f"ADC noise stamp: {adc_stamp}")

# Load noise-injected ADC trace maps from volumes_noised/.
# These are the geometric-mean trace ADC maps fitted by
# process_dist_corrected_diff_noise.py — all maps are in ×10⁻³ mm²/s
# (same units as volumes/), so no ×1000 scaling is needed.
adcMESE_blipup = load_nii(
    os.path.join(noised_dir, f"brainweb-{subject}_ADC_blipup_trace_noise_{adc_stamp}.nii.gz")
)* 1e3
adcMESE_blipdown = load_nii(
    os.path.join(noised_dir, f"brainweb-{subject}_ADC_blipdown_trace_noise_{adc_stamp}.nii.gz")
)* 1e3
adcMESE_blipup_corrected = load_nii(
    os.path.join(noised_dir, f"brainweb-{subject}_ADC_blipup_corrected_trace_noise_{adc_stamp}.nii.gz")
)* 1e3
adcMESE_blipdown_corrected = load_nii(
    os.path.join(noised_dir, f"brainweb-{subject}_ADC_blipdown_corrected_trace_noise_{adc_stamp}.nii.gz")
)* 1e3

# Reference and multishot are noise-free ground truth loaded from volumes/./1000
adcRef = load_nii(f"volumes/brainweb-{subject}-D_ref_volume.nii.gz") 
adcmultishot_se = load_nii(f"volumes/brainweb-{subject}-ADC_NLLS_adc_multishot_volume.nii.gz")


ims = [
    adcMESE_blipup,
    adcMESE_blipdown,
    adcmultishot_se,
    adcRef,
    adcMESE_blipup_corrected,
    adcMESE_blipdown_corrected,
]
titles = [
    "ADC MESE Blip-Up (noise)",
    "ADC MESE Blip-Down (noise)",
    "ADC Multishot SE",
    "ADC Reference",
    "ADC MESE Blip-Up Corrected (noise)",
    "ADC MESE Blip-Down Corrected (noise)",
]

n_images = len(ims)
n_cols = 2
n_rows = (n_images + n_cols - 1) // n_cols
fig, ax = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
ax = ax.flatten()
for i, (im, title) in enumerate(zip(ims, titles)):
    ax[i].imshow(get_slice(im), cmap="gray")
    ax[i].set_title(title)
    ax[i].axis("off")

for i in range(n_images, len(ax)):
    ax[i].axis("off")

plt.tight_layout()


# %% ==============================================================================
#   Registration — all ADC images to Reference
# =================================================================================
adcRef_reg = np.squeeze(adcRef)

if REGISTER:
    import ants

    _ref_ants = ants.image_read(f"volumes/brainweb-{subject}-D_ref_volume.nii.gz")

    def _reg(path: str) -> np.ndarray:
        return ants.registration(
            _ref_ants,
            ants.image_read(path),
            type_of_transform="Rigid",
        )["warpedmovout"].numpy()

    adcMESE_blipup_reg = _reg(
        os.path.join(noised_dir, f"brainweb-{subject}_ADC_blipup_trace_noise_{adc_stamp}.nii.gz")
    ) * 1e3
    adcMESE_blipdown_reg = _reg(
        os.path.join(noised_dir, f"brainweb-{subject}_ADC_blipdown_trace_noise_{adc_stamp}.nii.gz")
    ) * 1e3
    adcMESE_blipup_corrected_reg = _reg(
        os.path.join(noised_dir, f"brainweb-{subject}_ADC_blipup_corrected_trace_noise_{adc_stamp}.nii.gz")
    ) * 1e3
    adcMESE_blipdown_corrected_reg = _reg(
        os.path.join(noised_dir, f"brainweb-{subject}_ADC_blipdown_corrected_trace_noise_{adc_stamp}.nii.gz")
    ) * 1e3
    adcmultishot_se_reg = _reg(
        f"volumes/brainweb-{subject}-ADC_NLLS_adc_multishot_volume.nii.gz"
    )
    print("Registration complete.")
else:
    adcMESE_blipup_reg = np.squeeze(adcMESE_blipup)
    adcMESE_blipdown_reg = np.squeeze(adcMESE_blipdown)
    adcMESE_blipup_corrected_reg = np.squeeze(adcMESE_blipup_corrected)
    adcMESE_blipdown_corrected_reg = np.squeeze(adcMESE_blipdown_corrected)
    adcmultishot_se_reg = np.squeeze(adcmultishot_se)
    print("Registration skipped — using raw images.")

# %% ==============================================================================
#   Pairwise comparison: MESE blipup vs blipdown, corrected, and multishot vs ref
# =================================================================================
pairs = [
    (
        adcMESE_blipup_reg,
        adcMESE_blipdown_reg,
        "ADC MESE Blip-Up (registered)",
        "ADC MESE Blip-Down (registered)",
        "Difference Blip-Up vs Blip-Down",
    ),
    (
        adcMESE_blipup_corrected_reg,
        adcMESE_blipdown_corrected_reg,
        "ADC MESE Blip-Up Corrected (registered)",
        "ADC MESE Blip-Down Corrected (registered)",
        "Difference Blip-Up vs Blip-Down (Corrected)",
    ),
    (
        adcMESE_blipup_reg ,
        adcMESE_blipup_corrected_reg ,
        "ADC MESE Blip-Up (registered)",
        "ADC MESE Blip-Up Corrected (registered)",
        "Difference Blip-Up raw vs corrected",
    ),
    (
        adcmultishot_se_reg,
        adcRef_reg,
        "ADC Multishot SE (registered)",
        "ADC Reference",
        "Difference Multishot SE vs Ref",
    ),
]

fig, ax = plt.subplots(len(pairs), 3, figsize=(15, 5 * len(pairs)))
for i, (img_a, img_b, title_a, title_b, title_diff) in enumerate(pairs):
    sl_a = get_slice(img_a)
    sl_b = get_slice(img_b)
    diff = np.abs(sl_a - sl_b)
    mae_value = mae(img_a, img_b, non_zero=True)
    ax[i, 0].imshow(sl_a, cmap="gray")
    ax[i, 0].set_title(title_a)
    ax[i, 1].imshow(sl_b, cmap="gray")
    ax[i, 1].set_title(title_b)
    ax[i, 2].imshow(diff, cmap="plasma")
    ax[i, 2].set_title(f"{title_diff}\nMAE={mae_value:.4f} ×10⁻³ mm²/s")
    for j in range(3):
        ax[i, j].axis("off")
        fig.colorbar(ax[i, j].images[0], ax=ax[i, j], fraction=0.046, pad=0.04)
plt.tight_layout()

# %% ==============================================================================
#   All-pairs MAE matrix
# =================================================================================
refs = [
    adcRef_reg,
    adcmultishot_se_reg,
    adcMESE_blipup_reg,
    adcMESE_blipdown_reg,
    adcMESE_blipup_corrected_reg,
    adcMESE_blipdown_corrected_reg,
]
ref_names = [
    "Reference",
    "Multishot SE",
    "MSE EPI blip-up (noise)",
    "MSE EPI blip-down (noise)",
    "MSE EPI blip-up corrected (noise)",
    "MSE EPI blip-down corrected (noise)",
]
size = len(refs)

fig, ax = plt.subplots(
    size + 1, size + 1, figsize=(30 * (size + 1) / 6, 30 * (size + 1) / 6)
)
ax[0, 0].axis("off")
for i in range(len(refs)):
    ax[0, i + 1].imshow(get_slice(refs[i]), cmap="gray")
    ax[0, i + 1].axis("off")
    ax[0, i + 1].set_title(ref_names[i])
    ax[i + 1, 0].imshow(get_slice(refs[i]), cmap="gray")
    ax[i + 1, 0].axis("off")
    ax[i + 1, 0].set_title(ref_names[i])

for i in range(len(refs)):
    ref = refs[i]
    ref_name = ref_names[i]
    for j in range(len(refs)):
        print(f"i,j shape: {np.squeeze(ref).shape}, {np.squeeze(refs[j]).shape}")
        ax[i + 1, j + 1].axis("off")
        if i == j:
            continue
        target = refs[j]
        target_name = ref_names[j]
        im = np.abs(get_slice(ref) - get_slice(target))
        mae_value = mae(ref, target, non_zero=True)
        ax[i + 1, j + 1].imshow(im, cmap="plasma")
        ax[i + 1, j + 1].set_title(
            f"{ref_name} vs {target_name}\nMAE={mae_value:.4f} ×10⁻³ mm²/s"
        )
plt.show()

# %% ==============================================================================
#   ADC MAE matrix - Reference row only
# =================================================================================
refs = [
    adcRef_reg,
    adcmultishot_se_reg,
    adcMESE_blipup_reg  ,
    adcMESE_blipdown_reg  ,
    adcMESE_blipup_corrected_reg,
    adcMESE_blipdown_corrected_reg  ,
]
ref_names = [
    "Reference",
    "Multishot SE",
    "MSE EPI blip-up (noise)",
    "MSE EPI blip-down (noise)",
    "MSE EPI blip-up corrected (noise)",
    "MSE EPI blip-down corrected (noise)",
]
size = len(refs)

naming = lambda name: "ADC " + name + " [10⁻³ mm²/s]"

fig, ax = plt.subplots(2, size, figsize=(30 * size / 6, 30 * 2 / 6))

ax[0, 0].axis("off")
ax[0, 0].text(
    0.5,
    0.5,
    "ADC",
    fontsize=36,
    fontweight="bold",
    ha="center",
    va="center",
    transform=ax[0, 0].transAxes,
)
for i in range(size - 1):
    ax[0, i + 1].imshow(get_slice(refs[i + 1]), cmap="gray")
    ax[0, i + 1].axis("off")
    ax[0, i + 1].set_title(naming(ref_names[i + 1]))

for j in range(size):
    ax[1, j].axis("off")

ax[1, 0].imshow(get_slice(refs[0]), cmap="gray")
ax[1, 0].set_title(naming(ref_names[0]))

ref = refs[0]
ref_name = ref_names[0]
for j in range(1, size):
    target = refs[j]
    target_name = ref_names[j]
    im = np.abs(get_slice(ref) - get_slice(target))
    mae_value = mae(ref, target, non_zero=True)
    ax[1, j].imshow(im, cmap="plasma")
    ax[1, j].set_title(f"{ref_name} vs {target_name}\nMAE={mae_value:.4f} ×10⁻³ mm²/s")
fig.savefig("adc_noise_comparison_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# %% ==============================================================================
#   Per-tissue MAE (WM / GM / CSF)
# =================================================================================
tissue_mask_wm = load_nii(f"masks/brainweb-{subject}-mask_wm_volume.nii.gz")
tissue_mask_gm = load_nii(f"masks/brainweb-{subject}-mask_gm_volume.nii.gz")
tissue_mask_csf = load_nii(f"masks/brainweb-{subject}-mask_csf_volume.nii.gz")


tissue_masks_arr = np.stack([tissue_mask_wm, tissue_mask_gm, tissue_mask_csf], axis=0)  # (3, H, W) — WM, GM, CSF
tissue_masks_arr = tissue_masks_arr.astype(np.float32)
tissue_names_masks = ["WM", "GM", "CSF"]


def mae_tissue(im1, im2, mask):
    im1 = np.squeeze(im1)
    im2 = np.squeeze(im2)
    if im1.ndim == 3:
        im1 = get_slice(im1)
    if im2.ndim == 3:
        im2 = get_slice(im2)
    if mask.ndim == 3:
        mask = get_slice(mask)
    valid = (mask > 0) & (im1 > 0) & (im2 > 0)
    if valid.sum() == 0:
        return float("nan")
    return np.mean(np.abs(im1[valid] - im2[valid]))


n = len(refs)
n_tissues = len(tissue_names_masks)
mae_matrices = np.full((n_tissues, n, n), np.nan)
for t in range(n_tissues):
    for i in range(n):
        for j in range(n):
            if i != j:
                mae_matrices[t, i, j] = mae_tissue(
                    refs[i], refs[j], tissue_masks_arr[t]
                )

col_w = 12
print(f"\nPer-tissue ADC MAE (×10⁻³ mm²/s)")
print(f"{'Pair':<45}" + "".join(f"{name:>{col_w}}" for name in tissue_names_masks))
print("-" * (45 + col_w * n_tissues))
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        label = f"{ref_names[i]} vs {ref_names[j]}"
        print(
            f"{label:<45}"
            + "".join(f"{mae_matrices[t, i, j]:>{col_w}.4f}" for t in range(n_tissues))
        )

fig, axes = plt.subplots(1, n_tissues, figsize=(7 * n_tissues, 6))
for t, (ax_t, tissue) in enumerate(zip(axes, tissue_names_masks)):
    mat = mae_matrices[t]
    masked = np.ma.masked_invalid(mat)
    cmap = plt.cm.plasma.copy()
    cmap.set_bad(color="lightgray")
    im = ax_t.imshow(masked, cmap=cmap, aspect="auto")
    ax_t.set_xticks(range(n))
    ax_t.set_yticks(range(n))
    ax_t.set_xticklabels(ref_names, rotation=45, ha="right", fontsize=8)
    ax_t.set_yticklabels(ref_names, fontsize=8)
    ax_t.set_title(f"{tissue} — ADC MAE (×10⁻³ mm²/s)")
    fig.colorbar(im, ax=ax_t, fraction=0.046, pad=0.04)
    vmin, vmax = np.nanmin(mat), np.nanmax(mat)
    mid = (vmin + vmax) / 2
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            if not np.isnan(val):
                ax_t.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if val < mid else "black",
                )
plt.tight_layout()
plt.show()

for t, tissue in enumerate(tissue_names_masks):
    mask = tissue_masks_arr[t]
    if mask.ndim == 3:
        mask = get_slice(mask)
    refs_masked = [np.where(mask > 0, get_slice(np.squeeze(r)), -1) for r in refs]

    fig_g, ax_g = plt.subplots(n + 1, n + 1, figsize=(5 * (n + 1), 5 * (n + 1)))
    fig_g.suptitle(f"All-pairs comparison — {tissue}", fontsize=14)
    ax_g[0, 0].axis("off")
    for i in range(n):
        for ax_rc, lbl in [
            (ax_g[0, i + 1], ref_names[i]),
            (ax_g[i + 1, 0], ref_names[i]),
        ]:
            ax_rc.imshow(refs_masked[i], cmap="gray")
            ax_rc.set_title(lbl, fontsize=7)
            ax_rc.axis("off")
    for i in range(n):
        for j in range(n):
            ax_g[i + 1, j + 1].axis("off")
            if i == j:
                continue
            diff = np.abs(refs_masked[i] - refs_masked[j])
            ax_g[i + 1, j + 1].imshow(diff, cmap="plasma")
            ax_g[i + 1, j + 1].set_title(
                f"{ref_names[i]} vs {ref_names[j]}\nMAE={mae_matrices[t, i, j]:.4f} ×10⁻³ mm²/s",
                fontsize=7,
            )
    plt.tight_layout()
    plt.show()

# %% ==============================================================================
#   MAE vs Reference per tissue — MSE EPI, Multishot SE
# =================================================================================
methods = {
    "Multishot SE": (adcmultishot_se_reg, None),
    "MSE EPI (noise)": (adcMESE_blipup_reg, adcMESE_blipdown_reg),
    "MSE EPI Corrected (noise)": (adcMESE_blipup_corrected_reg, adcMESE_blipdown_corrected_reg),
}

mae_per_method = {}
for method, (img_a, img_b) in methods.items():
    vals = []
    for t in range(n_tissues):
        mask = tissue_masks_arr[t]
        m_a = mae_tissue(img_a, adcRef_reg, mask)
        if img_b is not None:
            m_b = mae_tissue(img_b, adcRef_reg, mask)
            vals.append((m_a + m_b) / 2)
        else:
            vals.append(m_a)
    mae_per_method[method] = vals

x = np.arange(n_tissues)
n_methods = len(mae_per_method)
width = 0.6 / n_methods
offsets = np.arange(n_methods) * width - (n_methods - 1) * width / 2

fig, ax_bar = plt.subplots(figsize=(8, 5))
cmap = plt.get_cmap("plasma")
colors = cmap(np.linspace(0, 1, n_methods))
for idx, (method, vals) in enumerate(mae_per_method.items()):
    bars = ax_bar.bar(x + offsets[idx], vals, width, label=method, color=colors[idx])
    for bar, v in zip(bars, vals):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(tissue_names_masks)
ax_bar.set_ylabel("MAE (×10⁻³ mm²/s)")
ax_bar.set_title("ADC MAE per tissue (noise)\n(MSE EPI averaged over blip-up and blip-down)")
ax_bar.legend()
plt.tight_layout()
plt.show()


# %% ==============================================================================
#   Mean ADC per tissue — Reference vs acquisition methods
# =================================================================================
def mean_tissue(img, mask):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = get_slice(img)
    if mask.ndim == 3:
        mask = get_slice(mask)
    valid = (mask > 0) & (img > 0)
    if valid.sum() == 0:
        return float("nan")
    return np.mean(img[valid])


all_acqs = {
    "Reference": (adcRef_reg, None),
    "Multishot SE": (adcmultishot_se_reg, None),
    "MSE EPI (noise)": (adcMESE_blipup_reg, adcMESE_blipdown_reg),
    "MSE EPI Corrected (noise)": (adcMESE_blipup_corrected_reg, adcMESE_blipdown_corrected_reg),
}

mean_per_acq = {}
for acq, (img_a, img_b) in all_acqs.items():
    vals = []
    for t in range(n_tissues):
        mask = tissue_masks_arr[t]
        m_a = mean_tissue(img_a, mask)
        if img_b is not None:
            m_b = mean_tissue(img_b, mask)
            vals.append((m_a + m_b) / 2)
        else:
            vals.append(m_a)
    mean_per_acq[acq] = vals

n_acqs = len(mean_per_acq)
width_m = 0.6 / n_acqs
offsets_m = np.arange(n_acqs) * width_m - (n_acqs - 1) * width_m / 2
colors_m = plt.get_cmap("plasma")(np.linspace(0, 1, n_acqs))

fig, ax_mean = plt.subplots(figsize=(9, 5))
for idx, (acq, vals) in enumerate(mean_per_acq.items()):
    bars = ax_mean.bar(
        x + offsets_m[idx], vals, width_m, label=acq, color=colors_m[idx]
    )
    for bar, v in zip(bars, vals):
        ax_mean.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
ax_mean.set_xticks(x)
ax_mean.set_xticklabels(tissue_names_masks)
ax_mean.set_ylabel("Mean ADC (×10⁻³ mm²/s)")
ax_mean.set_title(
    "Mean ADC per tissue by acquisition (noise)\n(MSE EPI averaged over blip-up and blip-down)"
)
ax_mean.legend()
plt.tight_layout()
plt.show()

# %% ==============================================================================
#   Reference comparison — per-tissue MAE + binary shape difference
# =================================================================================
blipdown_methods = {
    "MSE EPI blip-down (noise)": adcMESE_blipdown_reg,
    "MSE EPI blip-down corrected (noise)": adcMESE_blipdown_corrected_reg,
}

blipup_methods = {
    "MSE EPI blip-up (noise)": adcMESE_blipup_reg,
    "MSE EPI blip-up corrected (noise)": adcMESE_blipup_corrected_reg,
}
# ===========================================================
# Blip-down
# ===========================================================
# ---- Per-tissue MAE vs Reference -------------------------------------------------
mae_blipdown = {
    name: [mae_tissue(img, adcRef_reg, tissue_masks_arr[t]) for t in range(n_tissues)]
    for name, img in blipdown_methods.items()
}

x = np.arange(n_tissues)
n_bd = len(mae_blipdown)
width_bd = 0.6 / n_bd
offsets_bd = np.arange(n_bd) * width_bd - (n_bd - 1) * width_bd / 2
colors_bd = plt.get_cmap("plasma")(np.linspace(0, 1, n_bd))

fig, ax_bd = plt.subplots(figsize=(8, 5))
for idx, (name, vals) in enumerate(mae_blipdown.items()):
    bars = ax_bd.bar(x + offsets_bd[idx], vals, width_bd, label=name, color=colors_bd[idx])
    for bar, v in zip(bars, vals):
        ax_bd.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
ax_bd.set_xticks(x)
ax_bd.set_xticklabels(tissue_names_masks)
ax_bd.set_ylabel("MAE (×10⁻³ mm²/s)")
ax_bd.set_title("Blip-down ADC MAE per tissue vs Reference (noise)")
ax_bd.legend()
plt.tight_layout()
plt.show()

# ---- Binary shape + intensity difference vs Reference ----------------------------
def binarize(img):
    return get_slice(np.squeeze(img)) > 1e-4

def dice(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    denom = mask_a.sum() + mask_b.sum()
    return 2.0 * inter / denom if denom > 0 else float("nan")

ref_mask = binarize(adcRef_reg)
ref_slice = get_slice(adcRef_reg)

from matplotlib.colors import ListedColormap
diff_cmap = ListedColormap(["#000000", "#1f9e89", "#fde725"])  # bg, ref-only, method-only

fig, ax_sh = plt.subplots(len(blipdown_methods), 4, figsize=(20, 5 * len(blipdown_methods)))
if len(blipdown_methods) == 1:
    ax_sh = ax_sh[np.newaxis, :]

for i, (name, img) in enumerate(blipdown_methods.items()):
    m_mask = binarize(img)
    m_slice = get_slice(np.squeeze(img))

    # shape diff: 0=agreement, 1=reference only, 2=method only
    diff_label = np.zeros_like(m_mask, dtype=np.uint8)
    diff_label[ref_mask & ~m_mask] = 1
    diff_label[m_mask & ~ref_mask] = 2

    # intensity diff (×10⁻³ mm²/s), only where both have signal
    valid = ref_mask & m_mask
    intensity_diff = np.where(valid, np.abs(m_slice - ref_slice), 0.0)
    mae_value = mae(img, adcRef_reg, non_zero=True)

    d = dice(m_mask, ref_mask)
    n_disagree = int((diff_label > 0).sum())

    ax_sh[i, 0].imshow(m_mask, cmap="gray")
    ax_sh[i, 0].set_title(f"{name}\n(binary mask)")

    ax_sh[i, 1].imshow(ref_mask, cmap="gray")
    ax_sh[i, 1].set_title("Reference\n(binary mask)")

    ax_sh[i, 2].imshow(diff_label, cmap=diff_cmap, vmin=0, vmax=2)
    ax_sh[i, 2].set_title(
        f"Shape diff\nDice={d:.4f}, disagree={n_disagree} px\n"
        "yellow = method only, teal = reference only"
    )

    im_int = ax_sh[i, 3].imshow(intensity_diff, cmap="plasma")
    ax_sh[i, 3].set_title(f"Intensity diff |method - ref|\nMAE={mae_value:.4f} ×10⁻³ mm²/s")
    fig.colorbar(im_int, ax=ax_sh[i, 3], fraction=0.046, pad=0.04)

    for j in range(4):
        ax_sh[i, j].axis("off")

plt.tight_layout()
plt.show()


# ===========================================================
# Blip-up
# ===========================================================
# ---- Per-tissue MAE vs Reference -------------------------------------------------
mae_blipup = {
    name: [mae_tissue(img, adcRef_reg, tissue_masks_arr[t]) for t in range(n_tissues)]
    for name, img in blipup_methods.items()
}

x = np.arange(n_tissues)
n_bd = len(mae_blipup)
width_bd = 0.6 / n_bd
offsets_bd = np.arange(n_bd) * width_bd - (n_bd - 1) * width_bd / 2
colors_bd = plt.get_cmap("plasma")(np.linspace(0, 1, n_bd))

fig, ax_bd = plt.subplots(figsize=(8, 5))
for idx, (name, vals) in enumerate(mae_blipup.items()):
    bars = ax_bd.bar(x + offsets_bd[idx], vals, width_bd, label=name, color=colors_bd[idx])
    for bar, v in zip(bars, vals):
        ax_bd.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
ax_bd.set_xticks(x)
ax_bd.set_xticklabels(tissue_names_masks)
ax_bd.set_ylabel("MAE (×10⁻³ mm²/s)")
ax_bd.set_title("Blip-up ADC MAE per tissue vs Reference (noise)")
ax_bd.legend()
plt.tight_layout()
plt.show()

# ---- Binary shape + intensity difference vs Reference ----------------------------
ref_mask = binarize(adcRef_reg)
ref_slice = get_slice(adcRef_reg)

from matplotlib.colors import ListedColormap
diff_cmap = ListedColormap(["#000000", "#1f9e89", "#fde725"])  # bg, ref-only, method-only

fig, ax_sh = plt.subplots(len(blipup_methods), 4, figsize=(20, 5 * len(blipup_methods)))
if len(blipup_methods) == 1:
    ax_sh = ax_sh[np.newaxis, :]

for i, (name, img) in enumerate(blipup_methods.items()):
    m_mask = binarize(img)
    m_slice = get_slice(np.squeeze(img))

    # shape diff: 0=agreement, 1=reference only, 2=method only
    diff_label = np.zeros_like(m_mask, dtype=np.uint8)
    diff_label[ref_mask & ~m_mask] = 1
    diff_label[m_mask & ~ref_mask] = 2

    # intensity diff (×10⁻³ mm²/s), only where both have signal
    valid = ref_mask & m_mask
    intensity_diff = np.where(valid, np.abs(m_slice - ref_slice), 0.0)
    mae_value = mae(img, adcRef_reg, non_zero=True)

    d = dice(m_mask, ref_mask)
    n_disagree = int((diff_label > 0).sum())

    ax_sh[i, 0].imshow(m_mask, cmap="gray")
    ax_sh[i, 0].set_title(f"{name}\n(binary mask)")

    ax_sh[i, 1].imshow(ref_mask, cmap="gray")
    ax_sh[i, 1].set_title("Reference\n(binary mask)")

    ax_sh[i, 2].imshow(diff_label, cmap=diff_cmap, vmin=0, vmax=2)
    ax_sh[i, 2].set_title(
        f"Shape diff\nDice={d:.4f}, disagree={n_disagree} px\n"
        "yellow = method only, teal = reference only"
    )

    im_int = ax_sh[i, 3].imshow(intensity_diff, cmap="plasma")
    ax_sh[i, 3].set_title(f"Intensity diff |method - ref|\nMAE={mae_value:.4f} ×10⁻³ mm²/s")
    fig.colorbar(im_int, ax=ax_sh[i, 3], fraction=0.046, pad=0.04)

    for j in range(4):
        ax_sh[i, j].axis("off")

plt.tight_layout()
plt.show()

# %%
