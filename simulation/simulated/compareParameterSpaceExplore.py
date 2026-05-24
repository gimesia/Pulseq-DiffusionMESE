# IQ-BRAIN is funded by the European Union (MSCA Doctoral Network,
# December 2024–November 2028, Grant Agreement No. 101169519).
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

FIGS_PATH = r"C:\Users\gimes\OneDrive\PhD\Sumbission\ESMRMB26\Pulseq-DiffusionMESE\simulation\simulated\figs"

# ── Acquisition parameters ────────────────────────────────────────────────────
Nx, Ny = 96, 96
FOV = 224e-3  # m
res = FOV / Nx  # 2.333 mm
TR = 5.0  # s
ESP = 58  # ms, inter-echo spacing

# Parameter space
b_values = np.arange(0, 2001, 100, dtype=int)  # 21 b-values
te1s = np.arange(65, 125, 5, dtype=int)  # 18 TE1s
te2s = te1s + ESP
te3s = te1s + 2 * ESP

N_b = len(b_values)  # 21
N_TE1 = len(te1s)  # 18
N_echoes = 3
N_dirs = 6


# ── Timing helpers ────────────────────────────────────────────────────────────
def acquisition_time(n_shots_per_volume: int, n_volumes: int, TR: float) -> float:
    return n_shots_per_volume * n_volumes * TR


def fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    parts = []
    if h:
        parts.append(f"{h}h")
    if m or h:
        parts.append(f"{m}min")
    parts.append(f"{s:.0f}s")
    return " ".join(parts)


n_volumes_single = N_b * N_TE1 * N_echoes * N_dirs
n_volumes_triple = N_b * N_TE1 * N_dirs

t_multishot = acquisition_time(Ny, n_volumes_single, TR)
t_singleshot_single = acquisition_time(1, n_volumes_single, TR)
t_singleshot_triple = acquisition_time(1, n_volumes_triple, TR)

sequences = [
    ("Multishot\nsingle-SE EPI", t_multishot, "tab:purple"),
    ("Singleshot\nsingle-SE EPI", t_singleshot_single, "tab:green"),
    ("Singleshot\ntriple-SE EPI", t_singleshot_triple, "tab:orange"),
]


# ── Plot ──────────────────────────────────────────────────────────────────────
plt.set_cmap("plasma")
cmap_plasma = plt.cm.plasma

fig, axes = plt.subplots(1, 2, figsize=(13, 7))
fig.suptitle(
    f"MESE-EPI acquisition overview  —  {Nx}×{Ny}, FOV {FOV*1e3:.0f} mm, "
    f"TR {TR:.0f} s, ESP {ESP} ms, {N_dirs} dirs",
    fontsize=11,
    y=1.01,
)
colors = cmap_plasma(np.linspace(0, 1, 3))


# ── Left: parameter space scatter ────────────────────────────────────────────
ax = axes[0]
B, T1 = np.meshgrid(b_values, te1s)
B, T2 = np.meshgrid(b_values, te2s)
B, T3 = np.meshgrid(b_values, te3s)

cmap = plt.cm.plasma
ax.scatter(B.ravel(), T1.ravel(), c=colors[0], label="TE1", s=18, marker="o")
ax.scatter(B.ravel(), T2.ravel(), c=colors[1], label="TE2", s=18, marker="X")
ax.scatter(B.ravel(), T3.ravel(), c=colors[2], label="TE3", s=18, marker="D")

ax.set_xlabel("b-value [s/mm²]")
ax.set_ylabel("TE [ms]")
ax.set_title("Acquired parameter space (b-val vs TE)\n")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)

# ── Right: acquisition time (log scale) ──────────────────────────────────────
ax2 = axes[1]

labels = [s[0] for s in sequences]
times = np.array([s[1] for s in sequences])

x = np.arange(len(labels))
bars = ax2.bar(x, times / 3600, color=colors, width=0.5, alpha=0.9, zorder=3)

ax2.set_yscale("log")
ax2.set_ylabel("Acquisition time [hours]")
ax2.set_title("Total acquisition time by sequence type\n")

ax2.set_xticks(x)
ax2.set_xticklabels(labels, ha="center")

ax2.set_yticks([0.1, 1, 10, 100, 1000])
ax2.set_yticklabels([0.1, 1, 10, 100, 1000])


ax2.grid(True, axis="y", which="both", alpha=0.3, zorder=0)


# Annotate bars with human-readable time
for bar, t in zip(bars, times):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() * 1.1,
        fmt_time(t),
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )


plt.tight_layout()
plt.savefig(FIGS_PATH + r"\acquisition_overview.png", dpi=300, bbox_inches="tight")
plt.show()

# ── Console summary ───────────────────────────────────────────────────────────
print(f"\nAcquisition parameters")
print(f"  Matrix:      {Nx} × {Ny}")
print(f"  FOV:         {FOV*1e3:.0f} mm,  res = {res*1e3:.2f} mm")
print(f"  TR:          {TR:.1f} s")
print(f"  b-values:    {N_b}  ({b_values[0]}–{b_values[-1]} s/mm²)")
print(f"  TE1s:        {N_TE1}  ({te1s[0]}–{te1s[-1]} ms,  ESP={ESP} ms)")
print(f"  Echoes/shot: {N_echoes}  (TE1, TE2=TE1+{ESP}, TE3=TE1+{2*ESP})")
print(f"  Directions:  {N_dirs}")
print()
print(
    f"{'Sequence':<30} {'Shots/vol':>10} {'Volumes':>10} {'Total shots':>12} {'Time':>12}"
)
print("-" * 76)
print(
    f"{'Multishot single-SE EPI':<30} {Ny:>10} {n_volumes_single:>10} "
    f"{Ny*n_volumes_single:>12} {fmt_time(t_multishot):>12}"
)
print(
    f"{'Singleshot single-SE EPI':<30} {1:>10} {n_volumes_single:>10} "
    f"{n_volumes_single:>12} {fmt_time(t_singleshot_single):>12}"
)
print(
    f"{'Singleshot triple-SE EPI':<30} {1:>10} {n_volumes_triple:>10} "
    f"{n_volumes_triple:>12} {fmt_time(t_singleshot_triple):>12}"
)
print("\nSummary:")

print(f"Total parameters by acquisition: {len(b_values)*(len(te1s)*3)}")
# %%
