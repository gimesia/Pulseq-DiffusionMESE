# IQ-BRAIN is funded by the European Union (MSCA Doctoral Network,
# December 2024–November 2028, Grant Agreement No. 101169519).
# %%
import matplotlib.pyplot as plt
import numpy as np
import torch

t2SSE = np.load("vol/T2_SSE.npy")
t2MESE = np.load("vol/T2_MESE.npy")
t2Ref = np.load("vol/T2_Ref.npy")

fig, ax = plt.subplots(1,3, figsize=(15,5))
ax[0].imshow(t2SSE)
ax[0].set_title("T2 SSE")
ax[1].imshow(t2MESE)
ax[1].set_title("T2 MESE")
ax[2].imshow(t2Ref)
ax[2].set_title("T2 Reference")

# %%
diff_im = np.abs(t2MESE-t2SSE)
fig, ax = plt.subplots(1,3, figsize=(15,5))
ax[0].imshow(t2SSE)
ax[0].set_title("T2 SSE")
ax[1].imshow(t2MESE)
ax[1].set_title("T2 MESE")
ax[2].imshow(diff_im)
ax[2].set_title("Diff")

# %%
