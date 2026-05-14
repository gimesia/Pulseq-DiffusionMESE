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
t2Ref = (np.load("vol/T2_Ref.npy"))
t2multishot_se = np.fliplr(np.load("vol/T2_multishot_se.npy"))
t2MESE_distcorr_blipdown = np.load("vol/T2_MSE_distcorr_blipdown.npy")
t2SSE_distcorr_blipdown = np.load("vol/T2_SSE_distcorr_blipdown.npy")
t2MESE_distcorr_blipup = np.load("vol/T2_MSE_distcorr_blipup.npy")
t2SSE_distcorr_blipup = np.load("vol/T2_SSE_distcorr_blipup.npy")

ims = [t2SSE_blipup, t2SSE_blipdown, t2MESE_blipup, t2MESE_blipdown, t2MESE_distcorr_blipup, t2MESE_distcorr_blipdown, t2multishot_se, t2Ref]
titles = ["T2 SSE Blip-Up", "T2 SSE Blip-Down", "T2 MESE Blip-Up", "T2 MESE Blip-Down", "T2 MESE Distortion-Corrected Blip-Up", "T2 MESE Distortion-Corrected Blip-Down", "T2 Multishot SE", "T2 Reference"]

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
# Compare MESE vs SSE for blip-up and blip-down
plot_ims = [
    [t2SSE_blipup, t2MESE_blipup, np.abs(t2MESE_blipup - t2SSE_blipup)],
    [t2SSE_blipdown, t2MESE_blipdown, np.abs(t2MESE_blipdown - t2SSE_blipdown)],
    [t2SSE_distcorr_blipdown, t2MESE_distcorr_blipdown, np.abs(t2MESE_distcorr_blipdown - t2SSE_distcorr_blipdown)],
    [t2multishot_se, t2Ref, np.abs(t2multishot_se - t2Ref.squeeze())]
]

plot_titles = [
    ["T2 SSE Blip-Up", "T2 MSE Blip-Up", "Diff Blip-Up"],
    ["T2 SSE Blip-Down", "T2 MSE Blip-Down", "Diff Blip-Down"],
    ["T2 SSE Dist-Corr", "T2 MSE Dist-Corr", "Diff Dist-Corr"],
    ["T2 Multishot SE", "T2 Reference", "Diff Multishot SE vs Ref"]
]

for idx, i in enumerate(plot_ims):
    for j in range(2):
        print(f'{plot_titles[idx][j]}\tMin: {i[j].min()}, Max: {i[j].max()}, Mean: {i[j].mean()}')


fig, ax = plt.subplots(4, 3, figsize=(15, 20))

for i in range(4):
    for j in range(3):
        ax[i, j].imshow(plot_ims[i][j])
        ax[i, j].set_title(plot_titles[i][j])
        ax[i, j].axis('off')

plt.tight_layout()
# %%


refs = [t2Ref, t2multishot_se, t2SSE_blipup, t2SSE_blipdown, t2MESE_blipup, t2MESE_blipdown, t2SSE_distcorr_blipdown, t2MESE_distcorr_blipdown]
ref_names = ["Reference", "Multishot SE", "SSE EPI blip-up", "SSE EPI blip-down", "MSE EPI blip-up", "MSE EPI blip-down", "SSE Dist-Corr", "MSE Dist-Corr"]
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
