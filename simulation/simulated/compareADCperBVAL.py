# %%
import os
import re
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt

# Do not force a non-interactive backend so plots can appear in Jupyter.
# matplotlib will select an appropriate backend (e.g. inline) when run in
# a Jupyter environment.
from scipy import stats
from collections import defaultdict
import matplotlib.cm as cm

# ---- Config ----
# Resolved relative to this file (simulation/simulated/) so the script
# works after any clone, regardless of where the repo lives on disk.
_HERE = Path(__file__).resolve().parent
IMG_DIR = _HERE / "diff_img"
FIGS_DIR = _HERE / "figs"

OUTPUT_PATH = os.path.join(FIGS_DIR, "ADC_estimation_comparison.png")

BVALS = list(range(0, 2001, 100))  # 0, 100, ..., 1000
N_DIRS = 6  # dir0..dir5
TE = 100
MASK_THRESH_FRAC = 0.2

SEQUENCES = ["DiffMultiShotSE", "DiffSE", "DiffTripleSE"]
SEQ_TITLES = ["MultiShot SE", "Single SE", "Triple SE"]

CMAP = cm.get_cmap("plasma")
COLORS = [CMAP(i / (N_DIRS - 1)) for i in range(N_DIRS)][
    ::-1
]  # Reverse for better visibility
DIR_LABELS = ["dir1", "dir2", "dir3", "dir4", "dir5", "dir6"]


# ---- Helper: load and average blip pairs for one sequence ----
def load_sequence(img_dir, seq_name, bvals, n_dirs, te):
    pattern = re.compile(
        rf"{seq_name}-b(\d+)-dir(\d+)-TE(\d+)(?:-blip(up|down))?\.nii\.gz"
    )
    raw = defaultdict(lambda: defaultdict(list))
    found_any = False

    for fname in os.listdir(img_dir):
        m = pattern.match(fname)
        if m:
            b, d, te_f = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if te_f == te and b in bvals and d < n_dirs:
                raw[b][d].append(nib.load(os.path.join(img_dir, fname)).get_fdata())
                found_any = True

    if not found_any:
        print(f"WARNING: No files found for sequence '{seq_name}'")
        return None

    averaged = {}
    for b in bvals:
        for d in range(n_dirs):
            imgs = raw[b][d]
            if len(imgs) > 0:
                averaged[(b, d)] = np.mean(imgs, axis=0)
            else:
                print(f"  WARNING: {seq_name} missing b={b} dir={d}")
    return averaged


# ---- Helper: build mask from mean b=0 ----
def build_mask(averaged, n_dirs, thresh_frac):
    b0_imgs = [averaged[(0, d)] for d in range(n_dirs) if (0, d) in averaged]
    if len(b0_imgs) == 0:
        raise ValueError("No b=0 images found for masking")
    mean_b0 = np.mean(b0_imgs, axis=0)
    threshold = thresh_frac * mean_b0.max()
    mask = mean_b0 > threshold
    print(
        f"  Mask: {mask.sum()} voxels (thresh={threshold:.2f}, max={mean_b0.max():.2f})"
    )
    return mask


# ---- Helper: plot one ADC subplot ----
def plot_adc(ax, averaged, mask, bvals, n_dirs, colors, dir_labels, seq_name):
    adc_all = []

    for d in range(n_dirs):
        log_means = []
        cv_errs = []
        bvals_used = []

        for b in bvals:
            if (b, d) not in averaged:
                continue
            vox = averaged[(b, d)][mask]
            vox = vox[vox > 0]
            if len(vox) < 10:
                continue
            mean_s = np.mean(vox)
            std_s = np.std(vox)
            log_means.append(np.log(mean_s))
            cv_errs.append(std_s / mean_s)
            bvals_used.append(b)

        if len(bvals_used) < 2:
            print(f"  {seq_name} dir{d}: not enough points, skipping")
            continue

        bvals_arr = np.array(bvals_used, dtype=float)
        log_arr = np.array(log_means)
        cv_arr = np.array(cv_errs)

        slope, intercept, *_ = stats.linregress(bvals_arr, log_arr)
        adc = -slope
        adc_all.append(adc)
        print(f"  {seq_name} dir{d}: ADC = {adc:.2e} mm2/s")

        c = colors[d]
        lbl = dir_labels[d]

        ax.errorbar(
            bvals_arr,
            log_arr,
            yerr=cv_arr,
            fmt="o",
            color=c,
            markersize=4,
            capsize=3,
            elinewidth=1,
            markeredgewidth=0,
            label=lbl,
            zorder=3,
        )

        b_fit = np.linspace(0, max(bvals), 300)
        ax.plot(b_fit, intercept + slope * b_fit, color=c, linewidth=1.5, zorder=2)

    if adc_all:
        mean_adc = np.mean(adc_all)
        # ax.text(0.50, 0.10,
        #         f"ADC$_{{x,y,z}}$={mean_adc:.1e} (mm$^2$/s)",
        #         transform=ax.transAxes, color='black', fontsize=9)

    ax.set_xlim(-20, max(bvals) + 50)
    ax.set_xlabel("b-value (s/mm²)", color="black", fontsize=10)
    ax.set_ylabel("log(S)", color="black", fontsize=10)
    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
    ax.set_facecolor("white")


# ---- Main ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor("white")
fig.suptitle("ADC Estimation", fontsize=14, fontweight="bold", color="black")

for ax, seq, title in zip(axes, SEQUENCES, SEQ_TITLES):
    print(f"\nLoading {seq}...")
    averaged = load_sequence(IMG_DIR, seq, BVALS, N_DIRS, TE)
    if averaged is None:
        ax.text(
            0.5,
            0.5,
            f"No data\nfor {seq}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="black",
        )
        ax.set_title(title, color="black", fontsize=11, fontweight="bold")
        continue

    mask = build_mask(averaged, N_DIRS, MASK_THRESH_FRAC)
    plot_adc(ax, averaged, mask, BVALS, N_DIRS, COLORS, DIR_LABELS, seq)
    ax.set_title(title, color="black", fontsize=11, fontweight="bold")
    ax.legend(
        fontsize=8,
        facecolor="white",
        labelcolor="black",
        framealpha=1.0,
        edgecolor="black",
        loc="upper right",
    )

plt.tight_layout()
plt.show()

plt.savefig(OUTPUT_PATH, dpi=250, facecolor="white")
print(f"\nSaved: {OUTPUT_PATH}")
# %%
