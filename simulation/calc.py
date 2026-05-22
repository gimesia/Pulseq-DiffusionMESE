#%%
import os
from pathlib import Path

from matplotlib import pyplot as plt
import nibabel as nib
import numpy as np

from utils_relaxometry import create_t2_map


def read_nifti(filepath):
	"""Read a 4D NIfTI image and return the loaded image and data array."""
	filepath = Path(filepath)
	img = nib.load(str(filepath))
	data = img.get_fdata()

	return img, data


path = r"C:\Users\User\Desktop\t2_vol\merged_blipdown_corrected.nii.gz"
image, array = read_nifti(path)
print(f"Loaded {array.ndim}D NIfTI: shape={array.shape}, dtype={array.dtype}")


topupt_dir = "readoutTE1_same"

paths_blipdown = [
    rf"C:\Users\User\Desktop\topup_results\{topupt_dir}\brainweb-subj04-T2w_TE65_blipdown_corrected.nii.gz",
    rf"C:\Users\User\Desktop\topup_results\{topupt_dir}\brainweb-subj04-T2w_TE122_blipdown_corrected.nii.gz",
    rf"C:\Users\User\Desktop\topup_results\{topupt_dir}\brainweb-subj04-T2w_TE180_blipdown_corrected.nii.gz",
]
paths_blipup = [
    rf"C:\Users\User\Desktop\topup_results\{topupt_dir}\brainweb-subj04-T2w_TE65_blipup_corrected.nii.gz",
    rf"C:\Users\User\Desktop\topup_results\{topupt_dir}\brainweb-subj04-T2w_TE123_blipup_corrected.nii.gz",
    rf"C:\Users\User\Desktop\topup_results\{topupt_dir}\brainweb-subj04-T2w_TE181_blipup_corrected.nii.gz",
]

blipup_ims = []
for path in paths_blipup:
    img, data = read_nifti(path)
    blipup_ims.append(data)
blipdown_ims = []
for path in paths_blipdown:
    img, data = read_nifti(path)
    blipdown_ims.append(data)
    
blipdown_ims = np.stack(blipdown_ims, axis=-1)
blipup_ims = np.stack(blipup_ims, axis=-1)

# %%
slices=[]
slices_s0=[]

arr = blipdown_ims

for i in range(arr.shape[2]):
    if i < 30 or i > 50:
        pass
    print(f"Processing slice {i+1}/{arr.shape[2]}...", flush=True, end="\r")
    ims = np.transpose(arr[:,:,i,:], (2,0,1))
    t2, s0 = create_t2_map(ims,[0.065, 0.122, 0.180], t2_bounds=(0.0, 2))
    slices.append(t2)
    slices_s0.append(s0)
slices_stacked = np.stack(slices, axis=2)
slices_s0_stacked = np.stack(slices_s0, axis=2)



#%%
# Plot all slices in subplots with 8 columns
num_slices = slices_stacked.shape[2]
num_cols = 8
num_rows = int(np.ceil(num_slices / num_cols))
fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, num_rows * 2))
axes = axes.flatten()
for i in range(num_slices):
    axes[i].imshow(slices_stacked[:, :, i], cmap='seismic')
    axes[i].set_title(f'Slice {i}')
    axes[i].axis('off')

for i in range(num_slices, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.show()
# %%
output_dir = Path(r"C:\Users\User\Desktop\t2_vol\output")
os.makedirs(output_dir, exist_ok=True)

# Save T2 and S0 maps as NIfTI files using the affine from the input image
t2_img = nib.Nifti1Image(np.asarray(slices_stacked, dtype=np.float32), affine=image.affine)
s0_img = nib.Nifti1Image(np.asarray(slices_s0_stacked, dtype=np.float32), affine=image.affine)

t2_path = output_dir / f't2_map_{topupt_dir}.nii.gz'
s0_path = output_dir / f's0_map_{topupt_dir}.nii.gz'

nib.save(t2_img, str(t2_path))
nib.save(s0_img, str(s0_path))

print(f"Saved T2 map to: {t2_path}")
print(f"Saved S0 map to: {s0_path}")

# %%
