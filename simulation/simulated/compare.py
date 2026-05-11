# IQ-BRAIN is funded by the European Union (MSCA Doctoral Network,
# December 2024–November 2028, Grant Agreement No. 101169519).
# %%
import matplotlib.pyplot as plt
import numpy as np
import torch

t2SSE_blipup = np.load("vol/T2_SSE_blipup.npy")
t2SSE_blipdown = np.load("vol/T2_SSE_blipdown.npy")
t2MESE_blipup = np.load("vol/T2_MESE_blipup.npy")
t2MESE_blipdown = np.load("vol/T2_MESE_blipdown.npy")
t2Ref = np.rot90(np.load("vol/T2_Ref.npy"), -1)
t2multishot_se = np.load("vol/T2_multishot_se.npy")

fig, ax = plt.subplots(3,2, figsize=(10,15))
ax = ax.flatten()
ax[0].imshow(t2SSE_blipup)
ax[0].set_title("T2 SSE Blip-Up")
ax[1].imshow(t2SSE_blipdown)
ax[1].set_title("T2 SSE Blip-Down")
ax[2].imshow(t2MESE_blipup)
ax[2].set_title("T2 MESE Blip-Up")
ax[3].imshow(t2MESE_blipdown)
ax[3].set_title("T2 MESE Blip-Down")
ax[4].imshow(t2multishot_se)
ax[4].set_title("T2 Multishot SE")
ax[5].imshow(t2Ref)
ax[5].set_title("T2 Reference")


# %%
# Compare MESE vs SSE for blip-up and blip-down
diff_up = np.abs(t2MESE_blipup - t2SSE_blipup)
diff_down = np.abs(t2MESE_blipdown - t2SSE_blipdown)

fig, ax = plt.subplots(3, 3, figsize=(15, 15))
# Blip-up row
ax[0, 0].imshow(t2SSE_blipup)
ax[0, 0].set_title("T2 SSE Blip-Up")
ax[0, 1].imshow(t2MESE_blipup)
ax[0, 1].set_title("T2 MESE Blip-Up")
ax[0, 2].imshow(diff_up)
ax[0, 2].set_title("Diff Blip-Up")
# Blip-down row
ax[1, 0].imshow(t2SSE_blipdown)
ax[1, 0].set_title("T2 SSE Blip-Down")
ax[1, 1].imshow(t2MESE_blipdown)
ax[1, 1].set_title("T2 MESE Blip-Down")
ax[1, 2].imshow(diff_down)
ax[1, 2].set_title("Diff Blip-Down")
# Refs row
ax[2, 0].imshow(t2multishot_se)
ax[2, 0].set_title("T2 Multishot SE")
ax[2, 1].imshow(t2Ref)
ax[2, 1].set_title("T2 Reference")
ax[2,2].imshow(np.abs(t2multishot_se - t2Ref.squeeze()))
ax[2,2].set_title("Diff Multishot SE vs Ref")

for a in ax.flatten():
    a.axis('off')
plt.tight_layout()
# %%
fig, ax = plt.subplots(6, 6, figsize=(30, 30))
ax[0, 0].axis('off')

refs = [t2multishot_se, t2SSE_blipup, t2SSE_blipdown, t2MESE_blipup, t2MESE_blipdown]
ref_names = ["Multishot SE", "SE-SE EPI Blip-Up", "SE-SE EPI Blip-Down", "ME-SE EPI Blip-Up", "ME-SE EPI Blip-Down"]
for i in range(5):
    ax[0,i+1].imshow(refs[i], cmap="gray")
    ax[0,i+1].axis('off')
    ax[0,i+1].set_title(ref_names[i])
    ax[i+1,0].imshow(refs[i], cmap="gray")
    ax[i+1,0].axis('off')
    ax[i+1,0].set_title(ref_names[i])
    
for i in range(5):
    ref = refs[i]
    ref_name = ref_names[i]
    
    nonzero_mask_ref = ref > 0
    
    for j in range(5):
        ax[i+1,j+1].axis('off')
        if i == j:
            continue
        
        target = refs[j]
        target_name = ref_names[j]
        nonzero_mask_target = target > 0
        
        im = np.abs(ref - target)
        mae = np.mean(im[nonzero_mask_ref | nonzero_mask_target]) # only consider pixels where either ref or target is nonzero
        mae = mae * 1000 # convert to ms
        ax[i+1,j+1].imshow(im, cmap='seismic')
        ax[i+1,j+1].axis('off')
        ax[i+1,j+1].set_title(f"{ref_name} vs {target_name}\nMAE={mae:.4f} ms")
# plt.tight_layout()
plt.show()
# %%

# %%
from EPI_MRI.EPIMRIDistortionCorrection import *
from optimization.GaussNewton import *
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.float32  # single precision is fast on GPU and accurate enough

# 1. Load the blip-up/blip-down pair
# data = DataObject(
#     image_plus_path  = "",|
#     image_minus_path = "DiffTripleSE-TE-80-blipdown.nii.gz",
#     phase_encoding_direction = 3,   # 1, 2, or 3
#     device = device,
#     dtype  = dtype,
# )
path1 = './TE/DiffTripleSE-TE-80-blipup.nii.gz'
path2 = './TE/DiffTripleSE-TE-80-blipdown.nii.gz'

# Sanity check: load the images and print their shapes and dtypes
nib1 = nib.load(path1)
nib2 = nib.load(path2)
print("Shape of img1:", nib1.shape)
print("Dtype of img1:", nib1.get_fdata().dtype)
print("Shape of img2:", nib2.shape)
print("Dtype of img2:", nib2.get_fdata().dtype)

data = DataObject(
    img1=path1, 
    img2=path2,
    phase_encoding_direction=2,  # 1, 2, or 3
    device=device,
    dtype=dtype
)

# initialize the field map
loss_func = EPIMRIDistortionCorrection(data, 1000, 1e-7, regularizer=myLaplacian1D, PC=JacobiCG)
B0 = loss_func.initialize(blur_result=False)
opt = GaussNewton(loss_func, max_iter=1500, verbose=True, path='topup/')
opt.run_correction(B0)
opt.apply_correction()
# %%

path1_res = rf'topup\-im1Corrected.nii.gz'
path2_res = rf'topup\-im2Corrected.nii.gz'
fieldmap_res = rf'topup\-EstFieldMap.nii.gz'

# Sanity check: load the images and print their shapes and dtypes
nib3 = nib.load(path1_res)
nib4 = nib.load(path2_res)
nib5 = nib.load(fieldmap_res)
print("Shape of CORRECTED img1:", nib3.shape)
print("Dtype of CORRECTED img1:", nib3.get_fdata().dtype)
print("Shape of CORRECTED img2:", nib4.shape)
print("Dtype of CORRECTED img2:", nib4.get_fdata().dtype)
print("Shape of fieldmap:", nib5.shape)
print("Dtype of fieldmap:", nib5.get_fdata().dtype)

fig, ax = plt.subplots(1, 4, figsize=(20, 5))
ax[0].imshow(nib3.get_fdata()[:, :, nib3.shape[2]//2], cmap='gray')
ax[0].set_title("Corrected Image 1 (Blip-Up)")
ax[1].imshow(nib4.get_fdata()[:, :, nib4.shape[2]//2], cmap='gray')
ax[1].set_title("Corrected Image 2 (Blip-Down)")
ax[2].imshow(nib5.get_fdata()[:, :, nib5.shape[2]//2], cmap='seismic')
ax[2].set_title("Estimated Field Map")
ax[3].imshow(nib3.get_fdata() - nib4.get_fdata(), cmap='seismic')
ax[3].set_title("Difference Image")
# %%
