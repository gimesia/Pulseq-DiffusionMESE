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
from matplotlib.patches import FancyArrowPatch
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LinearLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cmcrameri import cm

from phantom_loader import load_phantom

# ──────────────────────────────────────────────────────────────────────────────
#  Paths and selection
# ──────────────────────────────────────────────────────────────────────────────
SUBJ_ID    = "subj04"  # which BrainWeb phantom subject to show

HERE       = Path(__file__).resolve().parent
SIM_DIR    = HERE / "simulated"
T2W_DIR    = SIM_DIR / "t2_vol"     # uncorrected — blip direction is visible
DIFF_DIR   = SIM_DIR / "diff_vol"   # uncorrected — blip direction is visible
VOL_DIR    = SIM_DIR / "volumes"
PHANTOM_JSON = (
    HERE.parent
    / "brainweb_phantoms" / f"brainweb-{SUBJ_ID}" / f"brainweb-{SUBJ_ID}-3T.json"
)

# Acquisitions shown in the middle block.
TES_MS  = [100, 158, 216]           # column → TE
B_VALS  = [0, 100, 500, 700]       # row    → b-value (dots inserted between)
DIR_ID  = 0                          # which diffusion direction to display
SWAP_TE = 158                        # this TE column is acquired in reversed
                                     # order (blipdown | blipup) instead of
                                     # (blipup | blipdown), for every b-value

SLICE_FRAC = 0.55  # which axial slice to show (fraction of nz)

D_max = 3.2
T2_max = 1800

# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_slice(path: Path, z = None) -> np.ndarray:
    """Load a NIfTI and return one axial slice, rotated for display.

    If ``z`` is None the middle slice is used.
    """
    arr = nib.load(str(path)).get_fdata()
    if arr.ndim == 2:
        return np.rot90(arr)
    z = arr.shape[2] // 2
    return np.rot90(arr[:, :, z])


def _t2w_path(te_ms: int, blip: str) -> Path:
    return T2W_DIR / f"brainweb-{SUBJ_ID}-T2w_TE{te_ms}_{blip}.nii.gz"


def _dwi_path(te_ms: int, b: int, blip: str, dir_id: int) -> Path:
    return (
        DIFF_DIR
        / f"brainweb-{SUBJ_ID}-ADCw_b{b}_dir{dir_id}_TE{te_ms}_{blip}.nii.gz"
    )


def _blip_glyph(blip: str) -> str:
    """Compact polarity indicator used in image labels."""
    return r"$\uparrow$" if blip == "blipup" else r"$\downarrow$"


def show(ax, img, *, title=None, cmap="gray", vmin=None, vmax=None, cbar_label=None):
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    if title:
        ax.set_title(title, fontsize=11)
        
    # Append a correctly sized colorbar to the RIGHT of the axis
    if cbar_label is not None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.08)
        cbar = plt.colorbar(im, cax=cax)
        # Enforce exactly 5 ticks evenly distributed across the data range
        cbar.locator = LinearLocator(numticks=5)
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=9)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=10)


def dots_row(axes):
    """Draw three centred dots in each axis of a row and hide the frame."""
    for ax in axes:
        ax.axis("off")
        ax.text(0.5, 0.5, r"$\cdots$",
                ha="center", va="center", fontsize=22,
                transform=ax.transAxes)


def big_arrow(fig, x0, x1, y, *, label=None):
    """Draw an arrow in figure coordinates from (x0,y) to (x1,y) safely."""
    # Use FancyArrowPatch attached to the figure to avoid axes distortion
    arrow = FancyArrowPatch(
        posA=(x0, y), posB=(x1, y),
        transform=fig.transFigure,
        arrowstyle="simple,head_length=15,head_width=15,tail_width=5",
        facecolor="white", edgecolor="black", linewidth=1.2,
        zorder=10
    )
    fig.add_artist(arrow)
    if label:
        fig.text((x0 + x1) / 2, y + 0.03, label, 
                 ha="center", va="bottom", fontsize=12, fontweight="bold")


# ──────────────────────────────────────────────────────────────────────────────
#  Load data
# ──────────────────────────────────────────────────────────────────────────────
print("[showcase] loading phantom for ground-truth D / T2 / B0 ...")
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

nz_vol = nib.load(str(VOL_DIR / f"brainweb-{SUBJ_ID}-D_ref_volume.nii.gz")).shape[2]
Z      = int(SLICE_FRAC * nz_vol)
print(f"[showcase] axial slice z = {Z}/{nz_vol}")

acq_imgs: dict[tuple[int, int, str], np.ndarray] = {}
for _te in TES_MS:
    for _b in B_VALS:
        for _blip in ("blipup", "blipdown"):
            acq_imgs[(_te, _b, _blip)] = load_slice(
                _dwi_path(_te, _b, _blip, DIR_ID), Z
            )

T2_out  = load_slice(
    SIM_DIR / "volumes_corrected"
            / f"brainweb-{SUBJ_ID}-T2w_t2_blipup_map_corrected.nii.gz",
    Z,
)  
ADC_out = load_slice(
    SIM_DIR / "volumes_corrected"
            / f"brainweb-{SUBJ_ID}-ADCw_adc_blipup_corrected_trace.nii.gz",
    Z,
) * 1000  
B0_out  = load_slice(
    SIM_DIR / "topup_results (FIELDMAPS)" / "readoutTE1_same"
            / "fieldmap_Hz_blipdown.nii.gz",
    Z,
)

# ──────────────────────────────────────────────────────────────────────────────
#  Figure layout
# ──────────────────────────────────────────────────────────────────────────────
# We establish a top-level GridSpec for the columns to cleanly enforce that the 
# Left (Phantom) and Right (qMRI) columns take up larger space than the center ones.
fig = plt.figure(figsize=(19, 9), facecolor="white")
master_gs = GridSpec(1, 3, left=0.02, right=0.98, top=0.92, bottom=0.06, 
                     width_ratios=[1.4, 3.2, 1.4], wspace=0.22)

#  Block 1: phantom inputs (3 rows × 1 col)
gs_in  = master_gs[0].subgridspec(3, 1, hspace=0.18)
ax_D  = fig.add_subplot(gs_in[0])
ax_T2 = fig.add_subplot(gs_in[1])
ax_B0 = fig.add_subplot(gs_in[2])

# No hard vmax limits on quantitative maps to auto-scale safely
show(ax_D,  D_in,        title="D", cbar_label="10$^{-3}$ mm²/s",
     cmap=cm.lipari,   vmin=0, vmax=D_max)
show(ax_T2, T2_in * 1e3, title="T2", cbar_label="ms",
     cmap=cm.navia, vmin=0, vmax=T2_max)
show(ax_B0, B0_in,        title="B0", cbar_label="Hz",
     cmap="seismic", vmin=-50, vmax=50)

# Header and Footer Labels assigned strictly to the first and last axes
ax_D.text(0.5, 1.25, "Phantom inputs", transform=ax_D.transAxes,
          ha="center", va="bottom", fontsize=13, fontweight="bold")
ax_B0.text(0.5, -0.22, "Phantom inputs", transform=ax_B0.transAxes,
          ha="center", va="top", fontsize=11, color="gray")


#  Block 2: (TE × b) acquisition matrix
gs_acq = master_gs[1].subgridspec(5, 3, hspace=0.45, wspace=0.18,
                                  height_ratios=[1, 1, 0.25, 1, 1])

b_to_row = {b: r for b, r in zip(B_VALS, (0, 1, 3, 4))} 

def _cell_label(ax_left, te_ms: int, b: int) -> None:
    """Joint (TE, b) label, sitting cleanly below the cell's bottom-LEFT corner."""
    ax_left.text(
        0.0, -0.06,
        f"(TE, b) = ({te_ms} ms, {b} s/mm²)",
        ha="left", va="top", fontsize=9, fontweight="bold",
        transform=ax_left.transAxes,
    )

def _blip_header(ax, blip: str) -> None:
    ax.text(
        0.5, 1.18, f"blip {_blip_glyph(blip)}",
        ha="center", va="bottom", fontsize=11, fontweight="bold",
        transform=ax.transAxes,
    )

for c_idx, te in enumerate(TES_MS):
    if te == SWAP_TE:
        left_blip, right_blip = "blipdown", "blipup"
    else:
        left_blip, right_blip = "blipup", "blipdown"

    for b in B_VALS:
        r = b_to_row[b]
        inner = gs_acq[r, c_idx].subgridspec(1, 2, wspace=0.04)
        ax_L = fig.add_subplot(inner[0, 0])
        ax_R = fig.add_subplot(inner[0, 1])

        # vmax removed completely to auto-scale frames independently
        show(ax_L, acq_imgs[(te, b, left_blip)],  vmin=0)
        show(ax_R, acq_imgs[(te, b, right_blip)], vmin=0)
        _cell_label(ax_L, te, b)

        if b == B_VALS[0]:
            _blip_header(ax_L, left_blip)
            _blip_header(ax_R, right_blip)

for c_idx in range(len(TES_MS)):
    ax_d = fig.add_subplot(gs_acq[2, c_idx])
    ax_d.axis("off")
    ax_d.text(0.5, 0.5, r"$\vdots$", ha="center", va="center",
              fontsize=22, transform=ax_d.transAxes)


#  Block 3: qMRI outputs (3 rows × 1 col)
gs_out = master_gs[2].subgridspec(3, 1, hspace=0.18)
ax_ADC = fig.add_subplot(gs_out[0])
ax_T2m = fig.add_subplot(gs_out[1])
ax_B0m = fig.add_subplot(gs_out[2])

# Vmax bounds stripped completely to adhere to your instruction
show(ax_ADC, ADC_out,        title="ADC map", cbar_label="10$^{-3}$ mm²/s",
     cmap=cm.lipari,   vmin=0, vmax=D_max)
show(ax_T2m, T2_out  * 1e3, title="T2 map", cbar_label="ms",
     cmap=cm.navia, vmin=0, vmax=T2_max)
show(ax_B0m, B0_out,         title="B0 map", cbar_label="Hz",
     cmap="seismic", vmin=-50, vmax=50)

# Header and Footer Labels attached strictly to the first and last axes
ax_ADC.text(0.5, 1.25, "qMRI maps", transform=ax_ADC.transAxes,
            ha="center", va="bottom", fontsize=13, fontweight="bold")
ax_B0m.text(0.5, -0.22, "qMRI maps", transform=ax_B0m.transAxes,
            ha="center", va="top", fontsize=11, color="gray")


#  Arrows connecting workflows
# big_arrow(fig, 0.12, 0.16, 0.50, label="MRzero")
# big_arrow(fig, 0.82, 0.86, 0.50, label="TOPUP + Fit")

out_path = SIM_DIR / "figs" / "pipeline_showcase.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=500, bbox_inches="tight", facecolor="white")
print(f"[showcase] saved -> {out_path}")

plt.show()
# %%