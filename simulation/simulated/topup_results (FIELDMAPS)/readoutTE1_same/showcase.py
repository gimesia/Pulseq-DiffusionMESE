#%%
import os

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
#%%

def load_nii(path: str) -> np.ndarray:
    return nib.load(path).get_fdata()

dir_path = os.path.abspath(os.path.dirname(__file__))

TE1bd = load_nii(f'{dir_path}\\brainweb-subj04-T2w_TE65_blipdown.nii.gz')
TE1bu = load_nii(f'{dir_path}\\brainweb-subj04-T2w_TE65_blipup.nii.gz')
TE2bd = load_nii(f'{dir_path}\\brainweb-subj04-T2w_TE123_blipdown.nii.gz')
TE2bu = load_nii(f'{dir_path}\\brainweb-subj04-T2w_TE123_blipup.nii.gz')
TE3bd = load_nii(f'{dir_path}\\brainweb-subj04-T2w_TE181_blipdown.nii.gz')
TE3bu = load_nii(f'{dir_path}\\brainweb-subj04-T2w_TE181_blipup.nii.gz')

TE1bd_dist_corr = load_nii(f'{dir_path}\\brainweb-subj04-T2w_TE65_blipdown_corrected.nii.gz')
TE1bu_dist_corr = load_nii(f'{dir_path}\\brainweb-subj04-T2w_TE65_blipup_corrected.nii.gz')
TE2bd_dist_corr = load_nii(f'{dir_path}\\brainweb-subj04-T2w_TE123_blipdown_corrected.nii.gz')
TE2bu_dist_corr = load_nii(f'{dir_path}\\brainweb-subj04-T2w_TE123_blipup_corrected.nii.gz')
TE3bd_dist_corr = load_nii(f'{dir_path}\\brainweb-subj04-T2w_TE181_blipdown_corrected.nii.gz')
TE3bu_dist_corr = load_nii(f'{dir_path}\\brainweb-subj04-T2w_TE181_blipup_corrected.nii.gz')

# %%
# display images in 2 rows

imgs = [
    (TE1bd, 'TE1 blip-down'), (TE2bd, 'TE2 blip-up'), (TE3bd, 'TE3 blip-down'),
    (TE1bd_dist_corr, 'TE1 after TOPUP'), (TE2bd_dist_corr, 'TE2 after TOPUP'), (TE3bd_dist_corr, 'TE3 after TOPUP'),
    (TE1bu, 'TE1 blip-up'), (TE2bu, 'TE2 blip-down'), (TE3bu, 'TE3 blip-up'),
]

n = len(imgs)
rows = 3
cols = n // rows
fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
for i, (img, title) in enumerate(imgs):
    r = i // cols
    c = i % cols
    ax = axes[r, c] if rows > 1 else axes[c]
    # show central slice in the last axis (z)
    if img.ndim == 3:
        slice_idx = img.shape[2] // 2
        im = img[:, :, slice_idx]
    else:
        im = img
    ax.imshow(np.rot90(im), cmap='gray')
    ax.axis('off')
    ax.set_title(title, fontsize=8)

plt.tight_layout()
plt.show()

fig.savefig('same_showcase.png', dpi=500)

# %%
