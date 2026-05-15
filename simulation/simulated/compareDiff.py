# IQ-BRAIN is funded by the European Union (MSCA Doctoral Network,
# December 2024–November 2028, Grant Agreement No. 101169519).
# %% ==============================================================================
#   Imports & data loading
# =================================================================================
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

SUBJECT_ID = 0
subjects = np.unique([f.split('-')[1] for f in os.listdir("brainmaps/") if f.startswith("brainweb-subj")])
subject = subjects[SUBJECT_ID]
print(f"Selected subject: {subject}")

REGISTER = True

def mae(im1, im2, non_zero=True)-> float:
    im1 = np.squeeze(im1)
    im2 = np.squeeze(im2)
    if non_zero:
        non_zero_mask = (im1 > 0) & (im2 > 0)
        return np.mean(np.abs(im1[non_zero_mask] - im2[non_zero_mask]))
    else:
        return np.mean(np.abs(im1 - im2))
    

adcSSE_blipup = np.load(f"brainmaps/brainweb-{subject}-ADC_SSE_blipup.npy")
adcSSE_blipdown = np.load(f"brainmaps/brainweb-{subject}-ADC_SSE_blipdown.npy")
adcMESE_blipup = np.load(f"brainmaps/brainweb-{subject}-ADC_MSE_blipup.npy")
adcMESE_blipdown = np.load(f"brainmaps/brainweb-{subject}-ADC_MSE_blipdown.npy")
adcRef = np.load(f"brainmaps/brainweb-{subject}-ADC_Ref.npy")
adcmultishot_se = np.load(f"brainmaps/brainweb-{subject}-ADC_multishot_se.npy")
# t2SSE_distcorr_blipup = np.load("brainmaps/T2_SSE_distcorr_blipup.npy")

ims = [adcSSE_blipup, adcSSE_blipdown, adcMESE_blipup, adcMESE_blipdown, adcmultishot_se, adcRef]
titles = ["ADC SSE Blip-Up", "ADC SSE Blip-Down", "ADC MESE Blip-Up", "ADC MESE Blip-Down", "ADC Multishot SE", "ADC Reference"]

n_images = len(ims)
n_cols = 2
n_rows = (n_images + n_cols - 1) // n_cols
fig, ax = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
ax = ax.flatten()
for i, (im, title) in enumerate(zip(ims, titles)):
    ax[i].imshow(im, cmap='gray')
    ax[i].set_title(title)
    ax[i].axis('off')

# Turn off unused subplots
for i in range(n_images, len(ax)):
    ax[i].axis('off')
    
plt.tight_layout()


# %% ==============================================================================
#   Registration — all ADC images to Reference
# =================================================================================
adcRef_reg = np.squeeze(adcRef)

if REGISTER:
    import ants
    _ref_ants = ants.from_numpy(adcRef_reg.astype(np.float32))

    def _reg(img):
        return ants.registration(
            _ref_ants,
            ants.from_numpy(np.squeeze(img).astype(np.float32)),
            type_of_transform='Rigid',
        )['warpedmovout'].numpy()

    adcSSE_blipup_reg    = _reg(adcSSE_blipup)
    adcSSE_blipdown_reg  = _reg(adcSSE_blipdown)
    adcMESE_blipup_reg   = _reg(adcMESE_blipup)
    adcMESE_blipdown_reg = _reg(adcMESE_blipdown)
    adcmultishot_se_reg  = _reg(adcmultishot_se)
    print("Registration complete.")
else:
    adcSSE_blipup_reg    = np.squeeze(adcSSE_blipup)
    adcSSE_blipdown_reg  = np.squeeze(adcSSE_blipdown)
    adcMESE_blipup_reg   = np.squeeze(adcMESE_blipup)
    adcMESE_blipdown_reg = np.squeeze(adcMESE_blipdown)
    adcmultishot_se_reg  = np.squeeze(adcmultishot_se)
    print("Registration skipped — using raw images.")

# %% ==============================================================================
#   Pairwise comparison: SSE vs MESE, and multishot SE vs reference (registered)
# =================================================================================
pairs = [
    (adcSSE_blipup_reg,   adcMESE_blipup_reg,   "ADC SSE Blip-Up (registered)",   "ADC MESE Blip-Up (registered)",   "Difference Blip-Up"),
    (adcSSE_blipdown_reg, adcMESE_blipdown_reg, "ADC SSE Blip-Down (registered)", "ADC MESE Blip-Down (registered)", "Difference Blip-Down"),
    (adcmultishot_se_reg, adcRef_reg,            "ADC Multishot SE (registered)",  "ADC Reference",             "Difference Multishot SE vs Ref"),
]

fig, ax = plt.subplots(len(pairs), 3, figsize=(15, 5 * len(pairs)))
for i, (img_a, img_b, title_a, title_b, title_diff) in enumerate(pairs):
    diff = np.abs(img_a - img_b)
    mae_value = mae(img_a, img_b, non_zero=True)
    ax[i, 0].imshow(img_a, cmap='gray')
    ax[i, 0].set_title(title_a)
    ax[i, 1].imshow(img_b, cmap='gray')
    ax[i, 1].set_title(title_b)
    ax[i, 2].imshow(diff, cmap='plasma')
    ax[i, 2].set_title(f"{title_diff}\nMAE={mae_value:.4f} ×10⁻³ mm²/s")
    for j in range(3):
        ax[i, j].axis('off')
        fig.colorbar(ax[i, j].images[0], ax=ax[i, j], fraction=0.046, pad=0.04)
plt.tight_layout()

# %% ==============================================================================
#   All-pairs MAE matrix
# =================================================================================
refs = [adcRef_reg, adcmultishot_se_reg, adcSSE_blipup_reg, adcSSE_blipdown_reg, adcMESE_blipup_reg, adcMESE_blipdown_reg]
ref_names = ["Reference", "Multishot SE", "SSE EPI blip-up", "SSE EPI blip-down", "MSE EPI blip-up", "MSE EPI blip-down"]
size = len(refs)

fig, ax = plt.subplots(size + 1, size + 1, figsize=(30 * (size + 1) / 6, 30 * (size + 1) / 6))
ax[0, 0].axis('off')
for i in range(len(refs)):
    ax[0,i+1].imshow(refs[i], cmap="gray")
    ax[0,i+1].axis('off')
    ax[0,i+1].set_title(ref_names[i])
    ax[i+1,0].imshow(refs[i], cmap="gray")
    ax[i+1,0].axis('off')
    ax[i+1,0].set_title(ref_names[i])
    
for i in range(len(refs)):
    ref = refs[i]
    ref_name = ref_names[i]
    
    nonzero_mask_ref = ref > 0
    
    for j in range(len(refs)):
        print(f"i,j shape: {ref.shape}, {refs[j].shape}")
        ax[i+1,j+1].axis('off')
        if i == j:
            continue
        
        target = refs[j]
        target_name = ref_names[j]
        nonzero_mask_target = target > 0
        
        im = np.abs(ref - target)
        mae_value = mae(ref, target, non_zero=True)

        ax[i+1,j+1].imshow(im, cmap='plasma')
        ax[i+1,j+1].axis('off')
        ax[i+1,j+1].set_title(f"{ref_name} vs {target_name}\nMAE={mae_value:.4f} ×10⁻³ mm²/s")
# plt.tight_layout()
plt.show()

# %% ==============================================================================
#   ADC MAE matrix - Reference row only
# =================================================================================
refs = [adcRef_reg, adcmultishot_se_reg, adcSSE_blipup_reg, adcSSE_blipdown_reg, adcMESE_blipup_reg, adcMESE_blipdown_reg]
ref_names = ["Reference", "Multishot SE", "SSE EPI blip-up", "SSE EPI blip-down", "MSE EPI blip-up", "MSE EPI blip-down"]
size = len(refs)

fig, ax = plt.subplots(2, size, figsize=(30 * size / 6, 30 * 2 / 6))

naming = lambda name: "ADC " + name + " [10⁻³ mm²/s]"

# Header row: empty col 0, images in cols 1..size-1
ax[0, 0].axis('off')
ax[0, 0].text(0.5, 0.5, "ADC", fontsize=36, fontweight='bold',
              ha='center', va='center', transform=ax[0, 0].transAxes)
for i in range(size - 1):
    ax[0, i+1].imshow(refs[i+1], cmap="gray")
    ax[0, i+1].axis('off')
    ax[0, i+1].set_title(naming(ref_names[i+1]))

# Bottom row: reference thumbnail in col 0, diff maps in cols 1..size-1
for j in range(size):
    ax[1, j].axis('off')

ax[1, 0].imshow(refs[0], cmap="gray")
ax[1, 0].set_title(naming(ref_names[0]))

ref = refs[0]
ref_name = ref_names[0]
for j in range(1, size):
    target = refs[j]
    target_name = ref_names[j]
    im = np.abs(ref - target)
    mae_value = mae(ref, target, non_zero=True)
    ax[1, j].imshow(im, cmap='plasma')
    ax[1, j].set_title(f"{ref_name} vs {target_name}\nMAE={mae_value:.4f} ×10⁻³ mm²/s")

plt.show()
# %% ==============================================================================
#   Per-tissue MAE (WM / GM / CSF)
# =================================================================================
tissue_masks_arr = np.load(f"masks/brainweb-{subject}-tissue_masks.npy") # (3, H, W) — WM, GM, CSF
tissue_masks_arr = np.rot90(tissue_masks_arr, k=1, axes=(1, 2))
tissue_names_masks = ["WM", "GM", "CSF"]


def mae_tissue(im1, im2, mask):
    im1 = np.squeeze(im1)
    im2 = np.squeeze(im2)
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
                mae_matrices[t, i, j] = mae_tissue(refs[i], refs[j], tissue_masks_arr[t])

col_w = 12
print(f"\nPer-tissue ADC MAE (×10⁻³ mm²/s)")
print(f"{'Pair':<45}" + "".join(f"{name:>{col_w}}" for name in tissue_names_masks))
print("-" * (45 + col_w * n_tissues))
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        label = f"{ref_names[i]} vs {ref_names[j]}"
        print(f"{label:<45}" + "".join(f"{mae_matrices[t, i, j]:>{col_w}.4f}" for t in range(n_tissues)))

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
    ax_t.set_yticklabels(ref_names, fontsize=8)
    ax_t.set_title(f"{tissue} — ADC MAE (×10⁻³ mm²/s)")
    fig.colorbar(im, ax=ax_t, fraction=0.046, pad=0.04)
    vmin, vmax = np.nanmin(mat), np.nanmax(mat)
    mid = (vmin + vmax) / 2
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            if not np.isnan(val):
                ax_t.text(j, i, f"{val:.2f}", ha="center", va="center",
                          fontsize=7, color="white" if val < mid else "black")
plt.tight_layout()
plt.show()

for t, tissue in enumerate(tissue_names_masks):
    mask = tissue_masks_arr[t]
    refs_masked = [np.where(mask, np.squeeze(r), -0.1) for r in refs]

    fig_g, ax_g = plt.subplots(n + 1, n + 1, figsize=(5 * (n + 1), 5 * (n + 1)))
    fig_g.suptitle(f"All-pairs comparison — {tissue}", fontsize=14)
    ax_g[0, 0].axis('off')
    for i in range(n):
        for ax_rc, lbl in [(ax_g[0, i+1], ref_names[i]), (ax_g[i+1, 0], ref_names[i])]:
            ax_rc.imshow(refs_masked[i], cmap='gray')
            ax_rc.set_title(lbl, fontsize=7)
            ax_rc.axis('off')
    for i in range(n):
        for j in range(n):
            ax_g[i+1, j+1].axis('off')
            if i == j:
                continue
            diff = np.abs(refs_masked[i] - refs_masked[j])
            ax_g[i+1, j+1].imshow(diff, cmap='plasma')
            ax_g[i+1, j+1].set_title(
                f"{ref_names[i]} vs {ref_names[j]}\nMAE={mae_matrices[t, i, j]:.4f} ×10⁻³ mm²/s",
                fontsize=7)
    plt.tight_layout()
    plt.show()

# %% ==============================================================================
#   MAE vs Reference per tissue — SSE, MSE, Multishot SE
# =================================================================================
methods = {
    "Multishot SE": (adcmultishot_se_reg,  None),
    "SSE EPI":      (adcSSE_blipup_reg,   adcSSE_blipdown_reg),
    "MSE EPI":      (adcMESE_blipup_reg,  adcMESE_blipdown_reg),
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
cmap = plt.get_cmap('plasma')
colors = cmap(np.linspace(0, 1, n_methods))
for idx, (method, vals) in enumerate(mae_per_method.items()):
    bars = ax_bar.bar(x + offsets[idx], vals, width, label=method, color=colors[idx])
    for bar, v in zip(bars, vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.4f}", ha='center', va='bottom', fontsize=7)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(tissue_names_masks)
ax_bar.set_ylabel("MAE (×10⁻³ mm²/s)")
ax_bar.set_title("ADC MAE per tissue\n(SSE / MSE averaged over blip-up and blip-down)")
ax_bar.legend()
plt.tight_layout()
plt.show()

# %%
