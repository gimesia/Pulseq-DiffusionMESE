# %%
import os
import sys

sim_path = r"C:\Users\User\OneDrive\PhD\Sumbission\ESMRMB26\Pulseq-DiffusionMESE\simulation"
if sim_path not in sys.path:
    sys.path.append(sim_path)

from EPI_MRI.EPIMRIDistortionCorrection import *
from optimization.GaussNewton import *
import torch
from utils_relaxometry import create_t2_map


lookup_tag = "DiffTripleSE"

filenames = [f for f in os.listdir('./TE/') if lookup_tag in f]

mseTEs = np.unique([int(f.split('-')[-2].replace('.nii.gz', '').replace('TE', '')) for f in filenames])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.float32  # single precision is fast on GPU and accurate enough

# store read-in paths for MSE (DiffTripleSE)
mse_paths = []
mse_imgs_bu = []
mse_imgs_bd = []

for TE in mseTEs:
    print(f"\nProcessing TE={TE} ms...", end="/r", flush=True)
    # 1. Load the blip-up/blip-down pair
    path1 = f'./TE/DiffTripleSE-TE-{TE}-blipup.nii.gz'
    path2 = f'./TE/DiffTripleSE-TE-{TE}-blipdown.nii.gz'

    mse_paths.append((path1, path2))

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
    opt = GaussNewton(loss_func, max_iter=1500, verbose=True, path='topupT2/')
    opt.run_correction(B0)
    opt.apply_correction()
    pass

    path1_res = rf'topupT2\-im1Corrected.nii.gz'
    path2_res = rf'topupT2\-im2Corrected.nii.gz'
    fieldmap_res = rf'topupT2\-EstFieldMap.nii.gz'

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
    fig.suptitle(f"MSE Distortion Correction Results (TE={TE} ms)")
    ax[0].imshow(nib3.get_fdata()[:, :, nib3.shape[2]//2], cmap='gray')
    ax[0].set_title(f"MSE Corrected Image 1\nTE={TE} ms")
    ax[1].imshow(nib4.get_fdata()[:, :, nib4.shape[2]//2], cmap='gray')
    ax[1].set_title(f"MSE Corrected Image 2\nTE={TE} ms")
    ax[2].imshow(nib5.get_fdata()[:, :, nib5.shape[2]//2], cmap='seismic')
    ax[2].set_title(f"MSE Estimated Field Map\nTE={TE} ms")
    ax[3].imshow(nib3.get_fdata() - nib4.get_fdata(), cmap='seismic')
    ax[3].set_title(f"MSE Difference Image\nTE={TE} ms")
    
    new_name_path1_res = rf'topupT2\{lookup_tag}-TE{TE}-im1Corrected.nii.gz'
    new_name_path2_res = rf'topupT2\{lookup_tag}-TE{TE}-im2Corrected.nii.gz'
    new_name_fieldmap_res = rf'topupT2\{lookup_tag}-TE{TE}-EstFieldMap.nii.gz'
    
    if os.path.exists(new_name_path1_res):
        os.remove(new_name_path1_res)

    if os.path.exists(new_name_path2_res):
        os.remove(new_name_path2_res)

    if os.path.exists(new_name_fieldmap_res):
        os.remove(new_name_fieldmap_res)

    os.rename(path1_res, new_name_path1_res)
    os.rename(path2_res, new_name_path2_res)
    os.rename(fieldmap_res, new_name_fieldmap_res)
    
    mse_imgs_bu.append(nib.load(new_name_path1_res).get_fdata())
    mse_imgs_bd.append(nib.load(new_name_path2_res).get_fdata())

mse_imgs_bu = np.array(mse_imgs_bu)   # (n_TEs, Ny, Nx)
print(f"Images stack: {mse_imgs_bu.shape}  TEs: {len(mseTEs)}")

mse_imgs_bd = np.array(mse_imgs_bd)   # (n_TEs, Ny, Nx)
print(f"Images stack: {mse_imgs_bd.shape}  TEs: {len(mseTEs)}")

# %%
lookup_tag = "DiffSE"

filenames = [f for f in os.listdir('./TE/') if lookup_tag in f]

sseTEs = np.unique([int(f.split('-')[-2].replace('.nii.gz', '').replace('TE', '')) for f in filenames])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.float32  # single precision is fast on GPU and accurate enough

# store read-in paths for SSE (DiffSE)
sse_paths = []
sse_imgs_bu = []
sse_imgs_bd = []

for TE in sseTEs:
    print(f"\nProcessing TE={TE} ms...", end="/r", flush=True)
    # 1. Load the blip-up/blip-down pair
    path1 = f'./TE/DiffSE-TE{TE}-blipup.nii.gz'
    path2 = f'./TE/DiffSE-TE{TE}-blipdown.nii.gz'

    sse_paths.append((path1, path2))

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
    opt = GaussNewton(loss_func, max_iter=1500, verbose=True, path='topupT2/')
    opt.run_correction(B0)
    opt.apply_correction()
    pass

    path1_res = rf'topupT2\-im1Corrected.nii.gz'
    path2_res = rf'topupT2\-im2Corrected.nii.gz'
    fieldmap_res = rf'topupT2\-EstFieldMap.nii.gz'

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
    fig.suptitle(f"SSE Distortion Correction Results (TE={TE} ms)")
    ax[0].imshow(nib3.get_fdata()[:, :], cmap='gray')
    ax[0].set_title(f"SSE Corrected Image 1\nTE={TE} ms")
    ax[1].imshow(nib4.get_fdata()[:, :], cmap='gray')
    ax[1].set_title(f"SSE Corrected Image 2\nTE={TE} ms")
    ax[2].imshow(nib5.get_fdata()[:, :], cmap='seismic')
    ax[2].set_title(f"SSE Estimated Field Map\nTE={TE} ms")
    ax[3].imshow(nib3.get_fdata() - nib4.get_fdata(), cmap='seismic')
    ax[3].set_title(f"SSE Difference Image\nTE={TE} ms")
    
    new_name_path1_res = rf'topupT2\{lookup_tag}-TE{TE}-im1Corrected.nii.gz'
    new_name_path2_res = rf'topupT2\{lookup_tag}-TE{TE}-im2Corrected.nii.gz'
    new_name_fieldmap_res = rf'topupT2\{lookup_tag}-TE{TE}-EstFieldMap.nii.gz'
    
    if os.path.exists(new_name_path1_res):
        os.remove(new_name_path1_res)

    if os.path.exists(new_name_path2_res):
        os.remove(new_name_path2_res)

    if os.path.exists(new_name_fieldmap_res):
        os.remove(new_name_fieldmap_res)

    os.rename(path1_res, new_name_path1_res)
    os.rename(path2_res, new_name_path2_res)
    os.rename(fieldmap_res, new_name_fieldmap_res)

    sse_imgs_bu.append(nib.load(new_name_path1_res).get_fdata())
    sse_imgs_bd.append(nib.load(new_name_path2_res).get_fdata())

sse_imgs_bu = np.array(sse_imgs_bu)   # (n_TEs, Ny, Nx)
print(f"Images stack: {sse_imgs_bu.shape}  TEs: {len(sseTEs)}")

sse_imgs_bd = np.array(sse_imgs_bd)   # (n_TEs, Ny, Nx)
print(f"Images stack: {sse_imgs_bd.shape}  TEs: {len(sseTEs)}")

# %%
relaxometry_results = {'mse_bu': create_t2_map(np.squeeze(mse_imgs_bu), mseTEs, method='nlls'),
                       'sse_bu': create_t2_map(np.squeeze(sse_imgs_bu), sseTEs, method='nlls'),
                       'mse_bd': create_t2_map(np.squeeze(mse_imgs_bd), mseTEs, method='nlls'),
                       'sse_bd': create_t2_map(np.squeeze(sse_imgs_bd), sseTEs, method='nlls')}

fig, ax = plt.subplots(2, 2, figsize=(12, 12))
ax = ax.flatten()
titles = ["MSE (blipup) T2 Map (NLLS Fit)", "SSE (blipup) T2 Map (NLLS Fit)", "MSE (blipdown) T2 Map (NLLS Fit)", "SSE (blipdown) T2 Map (NLLS Fit)"]
for i, key in enumerate(relaxometry_results.keys()):
    im = ax[i].imshow(np.rot90(relaxometry_results[key][0], -1), cmap='viridis')
    ax[i].set_title(titles[i])
fig.suptitle(f"Distortion-Corrected T2 Maps (NLLS Fit) for MSE and SSE Blip-Up/Blip-Down Images")
plt.tight_layout()
# save results
results_dir = r'C:\Users\User\OneDrive\PhD\Sumbission\ESMRMB26\Pulseq-DiffusionMESE\simulation\simulated\vol'

# save numeric outputs (uncompressed .npy)
np.save(os.path.join(results_dir, 'T2_MSE_distcorr_blipup.npy'), np.rot90(relaxometry_results['mse_bu'][0]/1000, -1)) # save AP and in s
np.save(os.path.join(results_dir, 'T2_SSE_distcorr_blipup.npy'), np.rot90(relaxometry_results['sse_bu'][0]/1000, -1)) # save AP and in s
np.save(os.path.join(results_dir, 'T2_MSE_distcorr_blipdown.npy'), np.rot90(relaxometry_results['mse_bd'][0]/1000, -1)) # save AP and in s
np.save(os.path.join(results_dir, 'T2_SSE_distcorr_blipdown.npy'), np.rot90(relaxometry_results['sse_bd'][0]/1000, -1)) # save AP and in s

# save relaxometry figure
print(f"Saved results to: {results_dir}")

# %%
