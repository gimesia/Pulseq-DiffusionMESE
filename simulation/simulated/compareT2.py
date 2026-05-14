# IQ-BRAIN is funded by the European Union (MSCA Doctoral Network,
# December 2024–November 2028, Grant Agreement No. 101169519).
# %%
import matplotlib.pyplot as plt
import numpy as np
import torch

def mae(im1, im2, non_zero=True)-> float:
    im1 = np.squeeze(im1)
    im2 = np.squeeze(im2)
    if non_zero:
        non_zero_mask = (im1 > 0) & (im2 > 0)
        return np.mean(np.abs(im1[non_zero_mask] - im2[non_zero_mask]))
    else:
        return np.mean(np.abs(im1 - im2))
    

t2SSE_blipup = np.load("brainmaps/T2_SSEblipup.npy")
t2SSE_blipdown = np.load("brainmaps/T2_SSEblipdown.npy")
t2MESE_blipup = np.load("brainmaps/T2_MESEblipup.npy")
t2MESE_blipdown = np.load("brainmaps/T2_MESEblipdown.npy")
t2Ref = (np.load("brainmaps/T2_Ref.npy"))
t2multishot_se = (np.load("brainmaps/T2_multishot_se.npy"))
# t2MESE_distcorr_blipdown = np.load("brainmaps/T2_MSE_distcorr_blipdown.npy")
# t2SSE_distcorr_blipdown = np.load("brainmaps/T2_SSE_distcorr_blipdown.npy")
# t2MESE_distcorr_blipup = np.load("brainmaps/T2_MSE_distcorr_blipup.npy")
# t2SSE_distcorr_blipup = np.load("brainmaps/T2_SSE_distcorr_blipup.npy")

ims = [t2SSE_blipup, t2SSE_blipdown, t2MESE_blipup, t2MESE_blipdown, t2multishot_se, t2Ref]
titles = ["T2 SSE Blip-Up", "T2 SSE Blip-Down", "T2 MESE Blip-Up", "T2 MESE Blip-Down", "T2 Multishot SE", "T2 Reference"]

n_images = len(ims)
n_cols = 2
n_rows = (n_images + n_cols - 1) // n_cols
fig, ax = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
ax = ax.flatten()
for i, (im, title) in enumerate(zip(ims, titles)):
    ax[i].imshow(im)
    ax[i].set_title(title)
    ax[i].axis('off')

# Turn off unused subplots
for i in range(n_images, len(ax)):
    ax[i].axis('off')
    
plt.tight_layout()


# %%
import ants
# Compare MESE vs SSE for blip-up and blip-down
plot_ims = [
    [t2SSE_blipup, t2MESE_blipup, np.abs(t2MESE_blipup - t2SSE_blipup)],
    [t2SSE_blipdown, t2MESE_blipdown, np.abs(t2MESE_blipdown - t2SSE_blipdown)],
    # [t2SSE_distcorr_blipdown, t2MESE_distcorr_blipdown, np.abs(t2MESE_distcorr_blipdown - t2SSE_distcorr_blipdown)],
    [t2multishot_se, t2Ref, np.abs(t2multishot_se - t2Ref.squeeze())]
]

plot_titles = [
    ["T2 SSE Blip-Up", "T2 MSE Blip-Up", "Diff Blip-Up"],
    ["T2 SSE Blip-Down", "T2 MSE Blip-Down", "Diff Blip-Down"],
    # ["T2 SSE Dist-Corr", "T2 MSE Dist-Corr", "Diff Dist-Corr"],
    ["T2 Multishot SE", "T2 Reference", "Diff Multishot SE vs Ref"]
]

for idx, i in enumerate(plot_ims):
    for j in range(2):
        print(f'{plot_titles[idx][j]}\tMin: {i[j].min()}, Max: {i[j].max()}, Mean: {i[j].mean()}')

rows = len(plot_ims)

fig, ax = plt.subplots(rows, 3, figsize=(15, 5 * rows))

for i in range(rows):
    print(f"Processing row {i+1}/{rows}")
    
    moving_img = np.squeeze(plot_ims[i][0])
    fixed_img = np.squeeze(plot_ims[i][1])
    assert fixed_img.shape == moving_img.shape, f"Shape mismatch: {fixed_img.shape} vs {moving_img.shape}"
    registration = ants.registration(ants.from_numpy(fixed_img), ants.from_numpy(moving_img), type_of_transform='Rigid')
    warped_moving = moving_img #registration['warpedmovout'].numpy()
    difference_img = np.abs(fixed_img - warped_moving)
    mae_value = mae(fixed_img, warped_moving, non_zero=True) * 1000 # convert to ms
    
    # mae_title = plot_titles[i][j] + f"\nMAE vs Ref: {mae_value:.4f} ms"
    # ax[i, j].set_title(mae_title)
    # ax[i, j].imshow(difference_img)
    # ax[i, j].imshow(plot_ims[i][j])
    # ax[i, j].set_title(plot_titles[i][j])
    ax[i, 0].imshow(fixed_img, cmap='gray')
    ax[i, 0].set_title(plot_titles[i][0])
    ax[i, 1].imshow(warped_moving, cmap='gray')
    ax[i, 1].set_title(plot_titles[i][1])
    ax[i, 2].imshow(difference_img, cmap='seismic')
    ax[i, 2].set_title(f"{plot_titles[i][2]}\nMAE={mae_value:.4f} ms")
    for j in range(3):
        fig.colorbar(ax[i, j].images[0], ax=ax[i, j], fraction=0.046, pad=0.04)
plt.tight_layout()
# %%

refs = [np.squeeze(t2Ref), np.squeeze(t2multishot_se), np.squeeze(t2SSE_blipup), np.squeeze(t2SSE_blipdown), np.squeeze(t2MESE_blipup), np.squeeze(t2MESE_blipdown)]
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
        mae_value = mae(ref, target, non_zero=True) # only consider pixels where either ref or target is nonzero
        mae_value = mae_value * 1000 # convert to ms
        ax[i+1,j+1].imshow(im, cmap='seismic')
        ax[i+1,j+1].axis('off')
        ax[i+1,j+1].set_title(f"{ref_name} vs {target_name}\nMAE={mae_value:.4f} ms")
# plt.tight_layout()
plt.show()

# %%
