"""
pipeline_showcase.py
--------------------
One-figure visual summary of the Pulseq-DiffusionMESE simulation pipeline:

    [ D  ]                [ T2w TE_1 ] [ DWI b_1 ]               [ T2 map  ]
    [ T2 ]    ── MRzero ──>[    ⋮    ] [   ⋮    ]── NLLS fit ──> [ ADC map ]
    [ B0 ]                [ T2w TE_N ] [ DWI b_M ]               [ B0 map  ]

Left column   : phantom ground-truth maps (D, T2, B0) — pulled from the
                BrainWeb phantom loader so they match the simulation grid.
Middle block  : 2 columns of acquisitions (T2w over TEs, DWI over b-values),
                4 rows with a row of dots between to indicate "many more".
Right column  : qMRI maps recovered by the per-pipeline NLLS fit (T2, ADC)
                plus the topup-style estimated field map (B0).

Author      : Aron Gimesi <aron.gimesi@tecnico.ulisboa.pt>
Affiliation : Instituto Superior Técnico | MSCA-DN IQ-BRAIN
Date        : 2026
Context     : ESMRMB 2026 — Pulseq DiffusionMESE showcase

Funding acknowledgement (mandatory):
    IQ-BRAIN is funded by the European Union (MSCA Doctoral Network,
    December 2024–November 2028, Grant Agreement No. 101169519).
"""
#%%
from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.patches import FancyArrow
from matplotlib.gridspec import GridSpec

from phantom_loader import load_phantom

# ──────────────────────────────────────────────────────────────────────────────
#  Paths and selection
# ──────────────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).resolve().parent
SIM_DIR    = HERE / "simulated"
T2W_DIR    = SIM_DIR / "topup_results (FIELDMAPS)" / "t2_volumes_corrected_same"  # all T2w files are here
DIFF_DIR   = SIM_DIR / "topup_results (FIELDMAPS)" / "diff_volumes_corrected_same"
VOL_DIR    = SIM_DIR / "volumes"
PHANTOM_JSON = (
    HERE.parent
    / "brainweb_phantoms" / "brainweb-subj04" / "brainweb-subj04-3T.json"
)

# Representative TEs / b-values shown in the middle block (top, top, bottom, bottom)
T2_TES_MS   = [80, 120, 158, 236]
DIFF_BVALS  = [100, 300, 500, 700]
DIFF_DIR_ID = 0
DIFF_TE_MS  = 100        # all DWI files exist at TE100/158/216 — pick one
BLIP        = "blipdown"   # which polarity to show

SLICE_FRAC  = 0.55       # which axial slice to show (fraction of nz)

# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_slice(path: Path) -> np.ndarray:
    """Load a NIfTI and return the z-th axial slice, rotated for display."""
    arr = nib.load(str(path)).get_fdata()
    if arr.ndim == 2:
        return np.rot90(arr)
    return np.rot90(arr[:, :, arr.shape[2] // 2])


def show(ax, img, *, title=None, cmap="gray", vmin=None, vmax=None):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    if title:
        ax.set_title(title, fontsize=9)


def dots_row(axes):
    """Draw three centred dots in each axis of a row and hide the frame."""
    for ax in axes:
        ax.axis("off")
        ax.text(0.5, 0.5, r"$\cdots$",
                ha="center", va="center", fontsize=22,
                transform=ax.transAxes)


def big_arrow(fig, x0, x1, y, *, label=None):
    """Draw an arrow in figure coordinates from (x0,y) to (x1,y)."""
    ax = fig.add_axes((x0, y - 0.06, x1 - x0, 0.12), frame_on=False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyArrow(
        0.05, 0.5, 0.9, 0.0,
        width=0.12, length_includes_head=True,
        head_width=0.45, head_length=0.18,
        facecolor="white", edgecolor="black", linewidth=1.5,
    ))
    if label:
        ax.text(0.5, 1.05, label, ha="center", va="bottom", fontsize=10)


# ──────────────────────────────────────────────────────────────────────────────
#  Load data
# ──────────────────────────────────────────────────────────────────────────────
print("[showcase] loading phantom for ground-truth D / T2 / B0 ...")
# phantom_loader prints Unicode arrows that crash on Windows' cp1252 console;
# capture its stdout so the showcase runs in any terminal.
with contextlib.redirect_stdout(io.StringIO()):
    phantom, _, _ = load_phantom(
        json_path=str(PHANTOM_JSON),
        resolution_mm=224 / 96,
        fov_mm=224.0,
        slice_idx=None,
    )
D_in  = np.rot90(phantom.D.squeeze().numpy())
T2_in = np.rot90(phantom.T2.squeeze().numpy())
B0_in = np.rot90(phantom.B0.squeeze().numpy())

# Slice index used everywhere on the saved volumes
nz_vol = nib.load(str(VOL_DIR / "brainweb-subj04-D_ref_volume.nii.gz")).shape[2]
Z      = int(SLICE_FRAC * nz_vol)
print(f"[showcase] axial slice z = {Z}/{nz_vol}")

# T2-weighted images at the chosen TEs
t2w_imgs = [
    load_slice(T2W_DIR / f"brainweb-subj04-T2w_TE{te}_{BLIP}_corrected.nii.gz")
    for te in T2_TES_MS
]

# Diffusion-weighted images at the chosen b-values
dwi_imgs = [
    load_slice(
        DIFF_DIR
        / f"brainweb-subj04-ADCw_b{b}_dir{DIFF_DIR_ID}_TE{DIFF_TE_MS}_{BLIP}_corrected.nii.gz",
    )
    for b in DIFF_BVALS
]

# Recovered qMRI maps
T2_out  = load_slice(
    r"C:\Users\User\OneDrive\PhD\Sumbission\ESMRMB26\Pulseq-DiffusionMESE\simulation\simulated\volumes_corrected\t2_blipup_map_corrected.nii.gz"
    # VOL_DIR / f"brainweb-subj04-T2_NLLS_t2_triple_{BLIP}_volume.nii.gz", Z
)
ADC_out = load_slice(
    r"C:\Users\User\OneDrive\PhD\Sumbission\ESMRMB26\Pulseq-DiffusionMESE\simulation\simulated\volumes_corrected\adc_blipup_dir1.nii.gz"
    # VOL_DIR / f"brainweb-subj04-ADC_NLLS_adc_triple_{BLIP}_volume.nii.gz", Z
) *1000
# B0 "output" — show the same field map: in this study the reference B0 is
# what topup is asked to recover, so it doubles as the recovered field map.
B0_out  = load_slice(
    r"C:\Users\User\OneDrive\PhD\Sumbission\ESMRMB26\Pulseq-DiffusionMESE\simulation\simulated\topup_results (FIELDMAPS)\readoutTE1_same\fieldmap_Hz_blipdown.nii.gz",
)

# ──────────────────────────────────────────────────────────────────────────────
#  Figure layout
# ──────────────────────────────────────────────────────────────────────────────
#  Three blocks side by side, separated by arrows drawn in figure coordinates.
#  Each block is its own GridSpec so the inner cells can have independent sizes
#  without forcing alignment across blocks.
fig = plt.figure(figsize=(14, 7), facecolor="white")

#  Block 1: phantom inputs (3 rows × 1 col) — left=0.04, right=0.22
gs_in  = GridSpec(3, 1, left=0.04, right=0.22, top=0.92, bottom=0.06,
                  hspace=0.18)
ax_D  = fig.add_subplot(gs_in[0])
ax_T2 = fig.add_subplot(gs_in[1])
ax_B0 = fig.add_subplot(gs_in[2])

show(ax_D,  D_in,        title="D  [10$^{-3}$ mm²/s]",
     cmap="magma",   vmin=0,    vmax=3.0)
show(ax_T2, T2_in * 1e3, title="T2  [ms]",
     cmap="viridis", vmin=0,    vmax=300)
show(ax_B0, B0_in,        title="B0  [Hz]",
     cmap="seismic", vmin=-50,  vmax=50)
ax_D.set_ylabel("Phantom\ninputs", fontsize=11, rotation=0,
                ha="right", va="center", labelpad=20)

#  Block 2: acquisitions (5 rows × 2 cols, middle row is the dots)
gs_acq = GridSpec(5, 2, left=0.32, right=0.68, top=0.92, bottom=0.06,
                  hspace=0.18, wspace=0.10,
                  height_ratios=[1, 1, 0.35, 1, 1])
ax_t2w = [fig.add_subplot(gs_acq[r, 0]) for r in (0, 1, 3, 4)]
ax_dwi = [fig.add_subplot(gs_acq[r, 1]) for r in (0, 1, 3, 4)]
ax_dots = [fig.add_subplot(gs_acq[2, 0]), fig.add_subplot(gs_acq[2, 1])]

# Column headers — place above the top row of images
ax_t2w[0].set_title("T2-weighted", fontsize=11, pad=10)
ax_dwi[0].set_title("Diffusion-weighted", fontsize=11, pad=10)

# Common display scale across the column for fair visual comparison
t2w_vmax = max(im.max() for im in t2w_imgs)
dwi_vmax = max(im.max() for im in dwi_imgs)

for ax, im, te in zip(ax_t2w, t2w_imgs, T2_TES_MS):
    show(ax, im, vmin=0, vmax=t2w_vmax)
    ax.text(0.04, 0.95, f"TE = {te} ms",
            color="white", fontsize=8, ha="left", va="top",
            transform=ax.transAxes,
            bbox=dict(facecolor="black", alpha=0.45,
                      edgecolor="none", boxstyle="round,pad=0.15"))

for ax, im, b in zip(ax_dwi, dwi_imgs, DIFF_BVALS):
    show(ax, im, vmin=0, vmax=dwi_vmax)
    ax.text(0.04, 0.95, f"b = {b} s/mm²",
            color="white", fontsize=8, ha="left", va="top",
            transform=ax.transAxes,
            bbox=dict(facecolor="black", alpha=0.45,
                      edgecolor="none", boxstyle="round,pad=0.15"))

dots_row(ax_dots)

#  Block 3: qMRI outputs (3 rows × 1 col)
gs_out = GridSpec(3, 1, left=0.78, right=0.96, top=0.92, bottom=0.06,
                  hspace=0.18)
ax_ADC = fig.add_subplot(gs_out[0])
ax_T2m = fig.add_subplot(gs_out[1])
ax_B0m = fig.add_subplot(gs_out[2])

show(ax_ADC, ADC_out,        title="ADC  [10$^{-3}$ mm²/s]",
     cmap="magma",   vmin=0,    vmax=3.0)
show(ax_T2m, T2_out  * 1e3, title="T2  [ms]",
     cmap="viridis", vmin=0,    vmax=300)
show(ax_B0m, B0_out,         title="B0  [Hz]",
     cmap="seismic", vmin=-50,  vmax=50)
ax_ADC.set_ylabel("qMRI\nmaps", fontsize=11, rotation=0,
                  ha="right", va="center", labelpad=20)

#  Arrows between blocks
big_arrow(fig, 0.23, 0.31, 0.50, label="MRzero")
big_arrow(fig, 0.69, 0.77, 0.50, label="NLLS fit / TOPUP")

# fig.suptitle(
#     "Pulseq-DiffusionMESE simulation pipeline",
#     fontsize=14, y=0.985,
# )

out_path = SIM_DIR / "figs" / "pipeline_showcase.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=500, bbox_inches="tight", facecolor="white")
print(f"[showcase] saved -> {out_path}")

plt.show()
# %%
