# IQ-BRAIN is funded by the European Union (MSCA Doctoral Network,
# December 2024–November 2028, Grant Agreement No. 101169519).
# %% ==============================================================================
#   Imports & data loading
# =================================================================================
import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

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


t2SSE_blipup = load_nii(
    f"volumes/brainweb-{subject}-T2_NLLS_t2_sse_blipup_volume.nii.gz"
)
t2SSE_blipdown = load_nii(
    f"volumes/brainweb-{subject}-T2_NLLS_t2_sse_blipdown_volume.nii.gz"
)
t2MESE_blipup = load_nii(
    f"volumes/brainweb-{subject}-T2_NLLS_t2_triple_blipup_volume.nii.gz"
)
t2MESE_blipdown = load_nii(
    f"volumes/brainweb-{subject}-T2_NLLS_t2_triple_blipdown_volume.nii.gz"
)
t2Ref = load_nii(f"volumes/brainweb-{subject}-T2_ref_volume.nii.gz")
t2multishot_se = load_nii(
    f"volumes/brainweb-{subject}-T2_NLLS_t2_multishot_volume.nii.gz"
)

ims = [
    t2SSE_blipup,
    t2SSE_blipdown,
    t2MESE_blipup,
    t2MESE_blipdown,
    t2multishot_se,
    t2Ref,
]
titles = [
    "T2 SSE Blip-Up",
    "T2 SSE Blip-Down",
    "T2 MESE Blip-Up",
    "T2 MESE Blip-Down",
    "T2 Multishot SE",
    "T2 Reference",
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
#   Registration — all T2 images to Reference
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

    t2SSE_blipup_reg = _reg(
        f"volumes/brainweb-{subject}-T2_NLLS_t2_sse_blipup_volume.nii.gz"
    )
    t2SSE_blipdown_reg = _reg(
        f"volumes/brainweb-{subject}-T2_NLLS_t2_sse_blipdown_volume.nii.gz"
    )
    t2MESE_blipup_reg = _reg(
        f"volumes/brainweb-{subject}-T2_NLLS_t2_triple_blipup_volume.nii.gz"
    )
    t2MESE_blipdown_reg = _reg(
        f"volumes/brainweb-{subject}-T2_NLLS_t2_triple_blipdown_volume.nii.gz"
    )
    t2multishot_se_reg = _reg(
        f"volumes/brainweb-{subject}-T2_NLLS_t2_multishot_volume.nii.gz"
    )
    print("Registration complete.")
else:
    t2SSE_blipup_reg = np.squeeze(t2SSE_blipup)
    t2SSE_blipdown_reg = np.squeeze(t2SSE_blipdown)
    t2MESE_blipup_reg = np.squeeze(t2MESE_blipup)
    t2MESE_blipdown_reg = np.squeeze(t2MESE_blipdown)
    t2multishot_se_reg = np.squeeze(t2multishot_se)
    print("Registration skipped — using raw images.")

fig.savefig("t2_comparison_matrix.png", dpi=300, bbox_inches="tight")


# %% ==============================================================================
#   Pairwise comparison: SSE vs MESE, and multishot SE vs reference (registered)
# =================================================================================
pairs = [
    (
        t2SSE_blipup_reg,
        t2MESE_blipup_reg,
        "T2 SSE Blip-Up (registered)",
        "T2 MESE Blip-Up (registered)",
        "Difference Blip-Up",
    ),
    (
        t2SSE_blipdown_reg,
        t2MESE_blipdown_reg,
        "T2 SSE Blip-Down (registered)",
        "T2 MESE Blip-Down (registered)",
        "Difference Blip-Down",
    ),
    (
        t2multishot_se_reg,
        t2Ref_reg,
        "T2 Multishot SE (registered)",
        "T2 Reference",
        "Difference Multishot SE vs Ref",
    ),
]

fig, ax = plt.subplots(len(pairs), 3, figsize=(15, 5 * len(pairs)))
for i, (img_a, img_b, title_a, title_b, title_diff) in enumerate(pairs):
    sl_a = get_slice(img_a)
    sl_b = get_slice(img_b)
    diff = np.abs(sl_a - sl_b)
    mae_value = mae(img_a, img_b, non_zero=True) * 1000
    ax[i, 0].imshow(sl_a, cmap="gray")
    ax[i, 0].set_title(title_a)
    ax[i, 1].imshow(sl_b, cmap="gray")
    ax[i, 1].set_title(title_b)
    ax[i, 2].imshow(diff, cmap="plasma")
    ax[i, 2].set_title(f"{title_diff}\nMAE={mae_value:.4f} ms")
    for j in range(3):
        ax[i, j].axis("off")
        fig.colorbar(ax[i, j].images[0], ax=ax[i, j], fraction=0.046, pad=0.04)
plt.tight_layout()

# %% ==============================================================================
#   All-pairs MAE matrix
# =================================================================================
refs = [
    t2Ref_reg,
    t2multishot_se_reg,
    t2SSE_blipup_reg,
    t2SSE_blipdown_reg,
    t2MESE_blipup_reg,
    t2MESE_blipdown_reg,
]
ref_names = [
    "Reference",
    "Multishot SE",
    "SSE EPI blip-up",
    "SSE EPI blip-down",
    "MSE EPI blip-up",
    "MSE EPI blip-down",
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
        mae_value = mae(ref, target, non_zero=True) * 1000
        ax[i + 1, j + 1].imshow(im, cmap="plasma")
        ax[i + 1, j + 1].set_title(
            f"{ref_name} vs {target_name}\nMAE={mae_value:.4f} ms"
        )
plt.show()

# %% ==============================================================================
#   First 2 rows of MAE matrix (Reference row only)
# =================================================================================
naming = lambda name: "T2 " + name + " [ms]"

fig, ax = plt.subplots(2, size, figsize=(30 * size / 6, 30 * 2 / 6))

ax[0, 0].axis("off")
ax[0, 0].text(
    0.5,
    0.5,
    "T2",
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
    mae_value = mae(ref, target, non_zero=True) * 1000
    ax[1, j].imshow(im, cmap="plasma")
    ax[1, j].set_title(f"{ref_name} vs {target_name}\nMAE={mae_value:.4f} ms")

plt.show()

# %% ==============================================================================
#   Per-tissue MAE (WM / GM / CSF)
# =================================================================================
tissue_masks_arr = np.load(
    f"masks/brainweb-{subject}-tissue_masks.npy"
)  # (3, H, W) — WM, GM, CSF
tissue_masks_arr = tissue_masks_arr  # np.rot90(tissue_masks_arr, k=1, axes=(1, 2))
tissue_names_masks = ["WM", "GM", "CSF"]

# Plot center tissue mask for each tissue
fig_masks, axs_masks = plt.subplots(
    1, len(tissue_names_masks), figsize=(6 * len(tissue_names_masks), 6)
)
for t, name in enumerate(tissue_names_masks):
    mask = tissue_masks_arr[t]
    # if mask is 3D, pick central slice; otherwise assume 2D
    if mask.ndim == 3:
        z = mask.shape[0] // 2
        disp = mask[z]
    else:
        disp = mask
    ax = axs_masks[t] if len(tissue_names_masks) > 1 else axs_masks
    ax.imshow(disp, cmap="gray")
    ax.set_title(f"Center mask - {name}")
    ax.axis("off")
plt.tight_layout()
plt.show()


def mae_tissue(im1, im2, mask):
    im1 = np.squeeze(im1)
    im2 = np.squeeze(im2)
    # flatten 3D to 2D if needed by taking the display slice
    if im1.ndim == 3:
        im1 = get_slice(im1)
    if im2.ndim == 3:
        im2 = get_slice(im2)
    valid = mask & (im1 > 0) & (im2 > 0)
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
                mae_matrices[t, i, j] = (
                    mae_tissue(refs[i], refs[j], tissue_masks_arr[t]) * 1000
                )

col_w = 12
print(f"\nPer-tissue T2 MAE (ms)")
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
    ax_t.set_title(f"{tissue} - T2 MAE (ms)")
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
    refs_masked = [np.where(mask, get_slice(np.squeeze(r)), -1) for r in refs]

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
                f"{ref_names[i]} vs {ref_names[j]}\nMAE={mae_matrices[t, i, j]:.4f} ms",
                fontsize=7,
            )
    plt.tight_layout()
    plt.show()

# %% ==============================================================================
#   MAE vs Reference per tissue — SSE, MSE, Multishot SE
# =================================================================================
methods = {
    "Multishot SE": (t2multishot_se_reg, None),
    "SSE EPI": (t2SSE_blipup_reg, t2SSE_blipdown_reg),
    "MSE EPI": (t2MESE_blipup_reg, t2MESE_blipdown_reg),
}

mae_per_method = {}
for method, (img_a, img_b) in methods.items():
    vals = []
    for t in range(n_tissues):
        mask = tissue_masks_arr[t]
        m_a = mae_tissue(img_a, t2Ref_reg, mask)
        if img_b is not None:
            m_b = mae_tissue(img_b, t2Ref_reg, mask)
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
ax_bar.set_ylabel("MAE (ms)")
ax_bar.set_title("T2 MAE per tissue\n(SSE / MSE averaged over blip-up and blip-down)")
ax_bar.legend()
plt.tight_layout()
plt.show()


# %% ==============================================================================
#   Mean T2 per tissue — Reference vs acquisition methods
# =================================================================================
def mean_tissue(img, mask):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = get_slice(img)
    valid = mask & (img > 0)
    if valid.sum() == 0:
        return float("nan")
    return np.mean(img[valid])


all_acqs = {
    "Reference": (t2Ref_reg, None),
    "Multishot SE": (t2multishot_se_reg, None),
    "SSE EPI": (t2SSE_blipup_reg, t2SSE_blipdown_reg),
    "MSE EPI": (t2MESE_blipup_reg, t2MESE_blipdown_reg),
}

mean_per_acq = {}
for acq, (img_a, img_b) in all_acqs.items():
    vals = []
    for t in range(n_tissues):
        mask = tissue_masks_arr[t]
        m_a = mean_tissue(img_a, mask) * 1000
        if img_b is not None:
            m_b = mean_tissue(img_b, mask) * 1000
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
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
ax_mean.set_xticks(x)
ax_mean.set_xticklabels(tissue_names_masks)
ax_mean.set_ylabel("Mean T2 (ms)")
ax_mean.set_title(
    "Mean T2 per tissue by acquisition\n(SSE / MSE averaged over blip-up and blip-down)"
)
ax_mean.legend()
plt.tight_layout()
plt.show()

# %%
