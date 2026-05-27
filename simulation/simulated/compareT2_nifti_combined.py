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
from cmcrameri import cm

cmap = cm.navia

filepath = os.path.dirname(os.path.abspath(__file__))
corrected_dir = os.path.join(filepath, "volumes_corrected")
noised_dir = os.path.join(filepath, "volumes_noised")
figs_dir = os.path.join(filepath, "figs")

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
    """MAE over 3D volume restricted to tissue mask (values in seconds)."""
    im1 = np.squeeze(im1)
    im2 = np.squeeze(im2)
    mask = np.squeeze(mask)
    valid = (mask > 0) & (im1 > 0) & (im2 > 0)
    if valid.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(im1[valid] - im2[valid])))


# Discover noise stamp from most recent run in volumes_noised/
_t2_noise_matches = sorted(
    glob.glob(os.path.join(noised_dir, f"brainweb-{subject}_T2_blipup_noise_*.nii.gz"))
)
if not _t2_noise_matches:
    raise FileNotFoundError(
        f"No noisy T2 maps found in {noised_dir}. "
        "Run process_dist_corrected_t2_noise.py first."
    )
_stamp_m = re.search(r"_noise_(SNR\d+_seed\d+_reg\w+)\.nii\.gz$", _t2_noise_matches[-1])
t2_stamp = _stamp_m.group(1) if _stamp_m else "SNR10_seed420_regON"
snr = t2_stamp.split("_")[0]
print(f"T2 noise stamp: {t2_stamp}")

# Noiseless volumes
t2Ref = load_nii(f"volumes/brainweb-{subject}-T2_ref_volume.nii.gz")
t2MESE_blipup = load_nii(f"volumes/brainweb-{subject}-T2_NLLS_t2_triple_blipup_volume.nii.gz")
t2MESE_blipdown = load_nii(f"volumes/brainweb-{subject}-T2_NLLS_t2_triple_blipdown_volume.nii.gz")
t2MESE_blipup_corrected = load_nii(
    os.path.join(corrected_dir, f"brainweb-{subject}-T2w_t2_blipup_map_corrected.nii.gz")
)
t2MESE_blipdown_corrected = load_nii(
    os.path.join(corrected_dir, f"brainweb-{subject}-T2w_t2_blipdown_map_corrected.nii.gz")
)

# Noisy volumes (SNR10)
t2_blipup_noise = load_nii(
    os.path.join(noised_dir, f"brainweb-{subject}_T2_blipup_noise_{t2_stamp}.nii.gz")
)
t2_blipdown_noise = load_nii(
    os.path.join(noised_dir, f"brainweb-{subject}_T2_blipdown_noise_{t2_stamp}.nii.gz")
)
t2_blipup_corrected_noise = load_nii(
    os.path.join(noised_dir, f"brainweb-{subject}_T2_blipup_corrected_noise_{t2_stamp}.nii.gz")
)
t2_blipdown_corrected_noise = load_nii(
    os.path.join(noised_dir, f"brainweb-{subject}_T2_blipdown_corrected_noise_{t2_stamp}.nii.gz")
)

# %% ==============================================================================
#   Registration — all images to Reference (3D rigid, ANTs)
# =================================================================================
t2Ref_reg = np.squeeze(t2Ref)

if REGISTER:
    import ants

    _ref_ants = ants.image_read(f"volumes/brainweb-{subject}-T2_ref_volume.nii.gz")

    def _reg(path: str) -> np.ndarray:
        return ants.registration(
            _ref_ants,
            ants.image_read(path),
            type_of_transform="Rigid",
        )["warpedmovout"].numpy()

    t2MESE_blipup_reg = _reg(
        f"volumes/brainweb-{subject}-T2_NLLS_t2_triple_blipup_volume.nii.gz"
    )
    t2MESE_blipdown_reg = _reg(
        f"volumes/brainweb-{subject}-T2_NLLS_t2_triple_blipdown_volume.nii.gz"
    )
    t2MESE_blipup_corrected_reg = _reg(
        os.path.join(corrected_dir, f"brainweb-{subject}-T2w_t2_blipup_map_corrected.nii.gz")
    )
    t2MESE_blipdown_corrected_reg = _reg(
        os.path.join(corrected_dir, f"brainweb-{subject}-T2w_t2_blipdown_map_corrected.nii.gz")
    )
    t2_blipup_noise_reg = _reg(
        os.path.join(noised_dir, f"brainweb-{subject}_T2_blipup_noise_{t2_stamp}.nii.gz")
    )
    t2_blipdown_noise_reg = _reg(
        os.path.join(noised_dir, f"brainweb-{subject}_T2_blipdown_noise_{t2_stamp}.nii.gz")
    )
    t2_blipup_corrected_noise_reg = _reg(
        os.path.join(noised_dir, f"brainweb-{subject}_T2_blipup_corrected_noise_{t2_stamp}.nii.gz")
    )
    t2_blipdown_corrected_noise_reg = _reg(
        os.path.join(
            noised_dir, f"brainweb-{subject}_T2_blipdown_corrected_noise_{t2_stamp}.nii.gz"
        )
    )
    print("Registration complete.")
else:
    t2MESE_blipup_reg = np.squeeze(t2MESE_blipup)
    t2MESE_blipdown_reg = np.squeeze(t2MESE_blipdown)
    t2MESE_blipup_corrected_reg = np.squeeze(t2MESE_blipup_corrected)
    t2MESE_blipdown_corrected_reg = np.squeeze(t2MESE_blipdown_corrected)
    t2_blipup_noise_reg = np.squeeze(t2_blipup_noise)
    t2_blipdown_noise_reg = np.squeeze(t2_blipdown_noise)
    t2_blipup_corrected_noise_reg = np.squeeze(t2_blipup_corrected_noise)
    t2_blipdown_corrected_noise_reg = np.squeeze(t2_blipdown_corrected_noise)
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
# conditions = {
#     "MSE blip-down": t2MESE_blipdown_reg,
#     "MSE blip-up": t2MESE_blipup_reg,
#     "MSE blip-down\n(noise SNR10)": t2_blipdown_noise_reg,
#     "MSE blip-up\n(noise SNR10)": t2_blipup_noise_reg,
#     "MSE blip-down\ncorrected": t2MESE_blipdown_corrected_reg,
#     "MSE blip-up\ncorrected": t2MESE_blipup_corrected_reg,
#     "MSE blip-down\ncorrected (noise)": t2_blipdown_corrected_noise_reg,
#     "MSE blip-up\ncorrected (noise)": t2_blipup_corrected_noise_reg,
# }

# mae_per_condition = {
#     name: [mae_tissue(img, t2Ref_reg, tissue_masks_arr[t]) * 1000 for t in range(n_tissues)]
#     for name, img in conditions.items()
# }

# # Print table
col_w = 12
# cond_names_flat = [n.replace("\n", " ") for n in conditions]
# print(f"\nPer-tissue T2 MAE vs Reference (ms) — 3D registered")
# print(f"{'Condition':<45}" + "".join(f"{t:>{col_w}}" for t in tissue_names))
# print("-" * (45 + col_w * n_tissues))
# for name, vals in zip(cond_names_flat, mae_per_condition.values()):
#     print(f"{name:<45}" + "".join(f"{v:>{col_w}.4f}" for v in vals))

x = np.arange(n_tissues)
# n_cond = len(conditions)
# width = 0.8 / n_cond
# offsets = np.arange(n_cond) * width - (n_cond - 1) * width / 2
# colors = plt.get_cmap("tab10")(np.linspace(0, 0.9, n_cond))

# fig, ax = plt.subplots(figsize=(14, 6))
# for idx, (name, vals) in enumerate(mae_per_condition.items()):
#     bars = ax.bar(x + offsets[idx], vals, width, label=name.replace("\n", " "), color=colors[idx])
#     for bar, v in zip(bars, vals):
#         if not np.isnan(v):
#             ax.text(
#                 bar.get_x() + bar.get_width() / 2,
#                 bar.get_height(),
#                 f"{v:.2f}",
#                 ha="center",
#                 va="bottom",
#                 fontsize=6,
#                 rotation=90,
#             )
# ax.set_xticks(x)
# ax.set_xticklabels(tissue_names, fontsize=12)
# ax.set_ylabel("MAE (ms)")
# ax.set_title("T2 MAE per tissue vs Reference — all 8 conditions (3D registered)")
# ax.legend(fontsize=7, ncol=2, loc="upper right")
# plt.tight_layout()
# plt.savefig("t2_per_tissue_mae_all_conditions.png", dpi=300, bbox_inches="tight")
# plt.show()

# %% ==============================================================================
#   Averaged blip-up / blip-down MAE per tissue — 4 grouped methods
# =================================================================================
avg_methods = {
    "MSE EPI": (t2MESE_blipup_reg, t2MESE_blipdown_reg),
    f"MSE EPI (noise {snr})": (t2_blipup_noise_reg, t2_blipdown_noise_reg),
    "MSE EPI corrected": (t2MESE_blipup_corrected_reg, t2MESE_blipdown_corrected_reg),
    f"MSE EPI corrected\n(noise {snr})": (
        t2_blipup_corrected_noise_reg,
        t2_blipdown_corrected_noise_reg,
    ),
}

mae_avg = {}
for name, (img_up, img_down) in avg_methods.items():
    vals = []
    for t in range(n_tissues):
        mask = tissue_masks_arr[t]
        m_up = mae_tissue(img_up, t2Ref_reg, mask)
        m_down = mae_tissue(img_down, t2Ref_reg, mask)
        vals.append((m_up + m_down) / 2 * 1000)
    mae_avg[name] = vals

print(f"\nPer-tissue T2 MAE vs Reference (ms) — blip-up/blip-down averaged")
print(f"{'Method':<40}" + "".join(f"{t:>{col_w}}" for t in tissue_names))
print("-" * (40 + col_w * n_tissues))
for name, vals in mae_avg.items():
    print(f"{name.replace(chr(10), ' '):<40}" + "".join(f"{v:>{col_w}.4f}" for v in vals))

n_avg = len(mae_avg)
width_a = 0.6 / n_avg
offsets_a = np.arange(n_avg) * width_a - (n_avg - 1) * width_a / 2
colors_a = cmap(np.linspace(0.1, 0.9, n_avg))

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
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
ax_a.set_xticks(x)
ax_a.set_xticklabels(tissue_names, fontsize=12)
ax_a.set_ylabel("MAE (ms)")
ax_a.set_title(
    "T2 MAE per tissue vs Reference\n(blip-up / blip-down averaged, 3D registered)"
)
ax_a.legend(fontsize=8, ncol=1, loc="upper left")
plt.tight_layout()
plt.savefig(rf"{figs_dir}/t2_per_tissue_mae_averaged_blipupdown.png", dpi=300, bbox_inches="tight")
plt.show()

# %% ==============================================================================
#   Mean T2 per tissue — Reference + all conditions (blip-up/blip-down averaged)
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
    "Reference": (t2Ref_reg, None),
    "MSE EPI": (t2MESE_blipup_reg, t2MESE_blipdown_reg),
    "MSE EPI\n(noise SNR10)": (t2_blipup_noise_reg, t2_blipdown_noise_reg),
    "MSE EPI\ncorrected": (t2MESE_blipup_corrected_reg, t2MESE_blipdown_corrected_reg),
    "MSE EPI\ncorrected (noise)": (
        t2_blipup_corrected_noise_reg,
        t2_blipdown_corrected_noise_reg,
    ),
}

mean_per_acq = {}
for name, (img_a, img_b) in all_acqs.items():
    vals = []
    for t in range(n_tissues):
        mask = tissue_masks_arr[t]
        m_a = mean_tissue(img_a, mask) * 1000
        if img_b is not None:
            m_b = mean_tissue(img_b, mask) * 1000
            vals.append((m_a + m_b) / 2)
        else:
            vals.append(m_a)
    mean_per_acq[name] = vals

# Print table
print(f"\nMean T2 per tissue (ms) — 3D registered, blip-up/blip-down averaged")
print(f"{'Approach':<40}" + "".join(f"{t:>{col_w}}" for t in tissue_names))
print("-" * (40 + col_w * n_tissues))
for name, vals in mean_per_acq.items():
    print(f"{name.replace(chr(10), ' '):<40}" + "".join(f"{v:>{col_w}.4f}" for v in vals))

n_acqs = len(mean_per_acq)
width_m = 0.6 / n_acqs
offsets_m = np.arange(n_acqs) * width_m - (n_acqs - 1) * width_m / 2

# Reference gets a distinct colour (dark grey); rest use plasma
ref_color = np.array([[0.15, 0.15, 0.15, 1.0]])
method_colors = cmap(np.linspace(0.1, 0.9, n_acqs - 1))
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
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

ax_m.set_xticks(x)
ax_m.set_xticklabels(tissue_names, fontsize=12)
ax_m.set_ylabel("Mean T2 (ms)")
ax_m.set_title(
    "Mean T2 per tissue — Reference vs all conditions\n(blip-up / blip-down averaged, 3D registered)"
)
ax_m.legend(fontsize=8, ncol=1, loc="upper left")
plt.tight_layout()
plt.savefig(rf"{figs_dir}/t2_per_tissue_mean_all_conditions.png", dpi=300, bbox_inches="tight")
plt.show()

# %% ==============================================================================
#   Mean T2 per tissue — Reference + all conditions — violin + box plots
# =================================================================================
import matplotlib.patches as mpatches
from scipy.ndimage import binary_erosion


def get_tissue_voxels(img, mask, thresh=1e-3) -> np.ndarray:
    """Return T2 values (ms) within tissue mask, strictly above threshold."""
    img  = np.squeeze(img)
    mask = np.squeeze(mask)
    valid = (mask > 0) & (img > thresh)
    return img[valid] * 1000

def get_masked_zero_voxels(img, mask) -> np.ndarray:
    """Return a boolean mask of voxels that are inside the tissue mask and zero-valued."""
    img = np.squeeze(img)
    mask = np.squeeze(mask)
    return (mask > 0) & (img <= 0)


all_acqs = {
    "Reference": (t2Ref_reg, None),
    "MSE EPI": (t2MESE_blipup_reg, t2MESE_blipdown_reg),
    f"MSE EPI\n(noise {snr})": (t2_blipup_noise_reg, t2_blipdown_noise_reg),
    "MSE EPI\ncorrected": (t2MESE_blipup_corrected_reg, t2MESE_blipdown_corrected_reg),
    f"MSE EPI\ncorrected (noise {snr})": (t2_blipup_corrected_noise_reg, t2_blipdown_corrected_noise_reg),
}

# Build per-tissue voxel arrays: dict[tissue] -> list of arrays (one per method)
# For blip-up/down pairs, concatenate voxels from both
voxels_per_tissue = {t: [] for t in tissue_names}
method_labels = []

for name, (img_a, img_b) in all_acqs.items():
    method_labels.append(name.replace("\n", " "))
    for ti, tname in enumerate(tissue_names):
        mask = tissue_masks_arr[ti]
        
        # eroded_mask = binary_erosion(mask, structure=np.ones((4,4,4)))
        # mask = eroded_mask.astype(np.float32)
        
        v_a = get_tissue_voxels(img_a, mask)
        if img_b is not None:
            v_b = get_tissue_voxels(img_b, mask)
            voxels = np.concatenate([v_a, v_b])
        else:
            voxels = v_a
        voxels_per_tissue[tname].append(voxels)

# Print mean ± std table
print(f"\nMean ± SD T2 per tissue (ms) — 3D registered, blip-up/blip-down pooled")
print(f"{'Approach':<40}" + "".join(f"{t:>{col_w*2}}" for t in tissue_names))
print("-" * (40 + col_w * 2 * n_tissues))
for i, label in enumerate(method_labels):
    row = f"{label:<40}"
    for tname in tissue_names:
        v = voxels_per_tissue[tname][i]
        row += f"{np.mean(v):>{col_w}.2f} ± {np.std(v):<{col_w}.2f}"
    print(row)

# Colors: reference dark grey, rest plasma
ref_color = np.array([0.15, 0.15, 0.15, 1.0])
method_colors_arr = cmap(np.linspace(0.1, 0.9, len(all_acqs) - 1))
colors_v = [ref_color] + [method_colors_arr[i] for i in range(len(all_acqs) - 1)]

n_methods = len(method_labels)
positions_base = np.arange(n_tissues)  # one group per tissue
spacing = 0.18  # gap between violins/boxes within a group
offsets_v = np.linspace(-(n_methods - 1) / 2 * spacing, (n_methods - 1) / 2 * spacing, n_methods)

# Violin plot
fig_v, ax_v = plt.subplots(figsize=(12, 6))
for ti, tname in enumerate(tissue_names):
    for mi, (label, color) in enumerate(zip(method_labels, colors_v)):
        pos = positions_base[ti] + offsets_v[mi]
        data = voxels_per_tissue[tname][mi]

        # --- Violin ---
        parts = ax_v.violinplot(data, positions=[pos], widths=spacing * 0.9,
                                showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(1.5)

ax_v.set_xticks(positions_base)
ax_v.set_xticklabels(tissue_names, fontsize=12)
ax_v.set_ylabel("T2 (ms)")
ax_v.set_title("Violin plot — T2 per tissue\n(blip-up/down pooled, 3D registered)")
ax_v.grid(axis="y", alpha=0.3)

# Box plot
fig_b, ax_b = plt.subplots(figsize=(12, 6))
for ti, tname in enumerate(tissue_names):
    for mi, (label, color) in enumerate(zip(method_labels, colors_v)):
        pos = positions_base[ti] + offsets_v[mi]
        data = voxels_per_tissue[tname][mi]

        # --- Box ---
        bp = ax_b.boxplot(data, positions=[pos], widths=spacing * 0.8,
                          patch_artist=True, showfliers=False,
                          medianprops=dict(color="white", linewidth=1.5),
                          whiskerprops=dict(color=color),
                          capprops=dict(color=color),
                          boxprops=dict(facecolor=(*color[:3], 0.7), edgecolor=color))
        # whisker caps colour
        for cap in bp["caps"]:
            cap.set_color(color)

ax_b.set_xticks(positions_base)
ax_b.set_xticklabels(tissue_names, fontsize=12)
ax_b.set_ylabel("T2 (ms)")
ax_b.set_title("Box plot — T2 per tissue\n(blip-up/down pooled, 3D registered)")
ax_b.grid(axis="y", alpha=0.3)

# Shared legend
legend_patches = [
    mpatches.Patch(facecolor=colors_v[i], alpha=0.8, label=method_labels[i])
    for i in range(n_methods)
]

fig_v.legend(handles=legend_patches, fontsize=8, ncol=1,
             loc="center right", bbox_to_anchor=(1.0, 0.5))
plt.figure(fig_v)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig(rf"{figs_dir}/t2_per_tissue_violin.png", dpi=300, bbox_inches="tight")
plt.show()

fig_b.legend(handles=legend_patches, fontsize=8, ncol=1,
             loc="center right", bbox_to_anchor=(1.0, 0.5))
plt.figure(fig_b)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig(rf"{figs_dir}/t2_per_tissue_box.png", dpi=300, bbox_inches="tight")
plt.show()
# %%
# %% ==============================================================================
#   Diagnostic — zero-voxel overlay on center slice per tissue × condition
# =================================================================================
T2_ZERO_THRESH = 1e-3  # seconds; voxels below this are treated as "zero"

diag_conditions = {
    "Reference":              t2Ref_reg,
    "MSE EPI blip-up":        t2MESE_blipup_reg,
    "MSE EPI blip-down":      t2MESE_blipdown_reg,
    f"Noise blip-up {snr}":   t2_blipup_noise_reg,
    f"Noise blip-down {snr}": t2_blipdown_noise_reg,
    "Corrected blip-up":      t2MESE_blipup_corrected_reg,
    "Corrected blip-down":    t2MESE_blipdown_corrected_reg,
    f"Corr+noise blip-up":    t2_blipup_corrected_noise_reg,
    f"Corr+noise blip-down":  t2_blipdown_corrected_noise_reg,
}

n_diag_cond = len(diag_conditions)
fig_d, axes_d = plt.subplots(
    n_tissues, n_diag_cond,
    figsize=(n_diag_cond * 2.2, n_tissues * 2.5),
)

for ti, tname in enumerate(tissue_names):
    mask_3d = np.squeeze(tissue_masks_arr[ti])
    # center slice index along DISPLAY_AXIS
    sl = mask_3d.shape[DISPLAY_AXIS] // 2

    mask_sl = get_slice(mask_3d, axis=DISPLAY_AXIS, idx=sl)

    for ci, (cname, img) in enumerate(diag_conditions.items()):
        ax = axes_d[ti, ci]
        img_sq = np.squeeze(img)
        img_sl  = get_slice(img_sq,  axis=DISPLAY_AXIS, idx=sl)

        # show T2 map clipped for display (seconds → ms for readability)
        im = ax.imshow(
            img_sl * 1000,
            cmap="gray",
            vmin=0,
            vmax=np.nanpercentile(img_sl[mask_sl > 0] * 1000, 99) if mask_sl.any() else 200,
            interpolation="nearest",
        )

        # overlay: inside mask AND below threshold → red
        zero_overlay = (mask_sl > 0) & (img_sl < T2_ZERO_THRESH)
        rgba = np.zeros((*img_sl.shape, 4))
        rgba[zero_overlay] = [1, 0, 0, 0.8]
        ax.imshow(rgba, interpolation="nearest")

        # mask boundary contour
        ax.contour(mask_sl, levels=[0.5], colors="cyan", linewidths=0.5)

        n_zero = int(zero_overlay.sum())
        ax.set_title(f"{cname}\n({tname}) zeros={n_zero}", fontsize=6)
        ax.axis("off")

fig_d.suptitle(
    f"Zero-voxel diagnostic (T2 < {T2_ZERO_THRESH*1000:.1f} ms) — slice {sl}\n"
    "Red = zero inside mask  |  Cyan = mask boundary",
    fontsize=9,
)
plt.tight_layout()
# plt.savefig("t2_zero_voxel_diagnostic.png", dpi=200, bbox_inches="tight")
plt.show()
print(
    f"\nDiagnostic saved. Zero threshold = {T2_ZERO_THRESH*1000:.1f} ms. "
    "Adjust T2_ZERO_THRESH if needed."
)
# %%
# %% ==============================================================================
#   Mean T2 per tissue — Reference + all conditions — violin + box plots
# =============================================================================
import matplotlib.patches as mpatches

T2_ZERO_THRESH = 1e-1  # seconds

def get_tissue_voxels(img, mask, thresh=T2_ZERO_THRESH) -> np.ndarray:
    """Return T2 values (ms) within tissue mask, strictly above threshold."""
    img  = np.squeeze(img)
    mask = np.squeeze(mask)
    valid = (mask > 0) & (img > thresh)
    return img[valid] * 1000


all_acqs = {
    "Reference": (t2Ref_reg, None),
    # "MSE EPI": (t2MESE_blipup_reg, t2MESE_blipdown_reg),
    f"MSE EPI\n(noise {snr})": (t2_blipup_noise_reg, t2_blipdown_noise_reg),
    # "MSE EPI\ncorrected": (t2MESE_blipup_corrected_reg, t2MESE_blipdown_corrected_reg),
    f"MSE EPI\ncorrected (noise {snr})": (t2_blipup_corrected_noise_reg, t2_blipdown_corrected_noise_reg),
}

voxels_per_tissue = {t: [] for t in tissue_names}
method_labels = []

for name, (img_a, img_b) in all_acqs.items():
    method_labels.append(name.replace("\n", " "))
    for ti, tname in enumerate(tissue_names):
        mask = tissue_masks_arr[ti]
        v_a = get_tissue_voxels(img_a, mask)
        if img_b is not None:
            v_b = get_tissue_voxels(img_b, mask)
            voxels = np.concatenate([v_a, v_b])
        else:
            voxels = v_a
        voxels_per_tissue[tname].append(voxels)

# Print mean ± std table
print(f"\nMean ± SD T2 per tissue (ms) — 3D registered, blip-up/blip-down pooled")
print(f"{'Approach':<40}" + "".join(f"{t:>{col_w*2}}" for t in tissue_names))
print("-" * (40 + col_w * 2 * n_tissues))
for i, label in enumerate(method_labels):
    row = f"{label:<40}"
    for tname in tissue_names:
        v = voxels_per_tissue[tname][i]
        row += f"{np.mean(v):>{col_w}.2f} ± {np.std(v):<{col_w}.2f}"
    print(row)

# Colors
ref_color = np.array([0.15, 0.15, 0.15, 1.0])
method_colors_arr = cmap(np.linspace(0.1, 0.9, len(all_acqs) - 1))
colors_v = [ref_color] + [method_colors_arr[i] for i in range(len(all_acqs) - 1)]

n_methods = len(method_labels)
positions_base = np.arange(n_tissues)
spacing = 0.18
offsets_v = np.linspace(-(n_methods - 1) / 2 * spacing, (n_methods - 1) / 2 * spacing, n_methods)

# Violin plot
fig_v, ax_v = plt.subplots(figsize=(12, 6))
for ti, tname in enumerate(tissue_names):
    for mi, (label, color) in enumerate(zip(method_labels, colors_v)):
        pos = positions_base[ti] + offsets_v[mi]
        data = voxels_per_tissue[tname][mi]
        parts = ax_v.violinplot(data, positions=[pos], widths=spacing * 0.9,
                                showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(1.5)

ax_v.set_xticks(positions_base)
ax_v.set_xticklabels(tissue_names, fontsize=12)
ax_v.set_ylabel("T2 (ms)")
ax_v.set_title("Violin plot — T2 per tissue\n(blip-up/down pooled, 3D registered)")
ax_v.grid(axis="y", alpha=0.3)

# Box plot
fig_b, ax_b = plt.subplots(figsize=(12, 6))
for ti, tname in enumerate(tissue_names):
    for mi, (label, color) in enumerate(zip(method_labels, colors_v)):
        pos = positions_base[ti] + offsets_v[mi]
        data = voxels_per_tissue[tname][mi]
        bp = ax_b.boxplot(data, positions=[pos], widths=spacing * 0.8,
                          patch_artist=True, showfliers=False,
                          medianprops=dict(color="white", linewidth=1.5),
                          whiskerprops=dict(color=color),
                          capprops=dict(color=color),
                          boxprops=dict(facecolor=(*color[:3], 0.7), edgecolor=color))
        for cap in bp["caps"]:
            cap.set_color(color)

ax_b.set_xticks(positions_base)
ax_b.set_xticklabels(tissue_names, fontsize=12)
ax_b.set_ylabel("T2 (ms)")
ax_b.set_title("Box plot — T2 per tissue\n(blip-up/down pooled, 3D registered)")
ax_b.grid(axis="y", alpha=0.3)

# Shared legend
legend_patches = [
    mpatches.Patch(facecolor=colors_v[i], alpha=0.8, label=method_labels[i])
    for i in range(n_methods)
]

fig_v.legend(handles=legend_patches, fontsize=8, ncol=1,
             loc="center right", bbox_to_anchor=(1.0, 0.5))
plt.figure(fig_v)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("t2_per_tissue_violin.png", dpi=300, bbox_inches="tight")
plt.show()

fig_b.legend(handles=legend_patches, fontsize=8, ncol=1,
             loc="center right", bbox_to_anchor=(1.0, 0.5))
plt.figure(fig_b)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("t2_per_tissue_box.png", dpi=300, bbox_inches="tight")
plt.show()
# %%
