"""
phantom_loader.py
-----------------
Load a MRzeroCore NIfTI phantom (>= 0.4.3), match it exactly to a given
sequence FOV and resolution so simulation and acquisition share the same
spatial grid.

Usage
-----
    from phantom_loader import load_phantom

    phantom, phantom_data = load_phantom(
        json_path      = "phantoms/brainweb-subj04/brainweb-subj04-3T.json",
        fov_mm         = 224.0,   # must match seq definition FOV
        resolution_mm  = 224/96,  # = 2.333... mm  →  96×96 matrix
        slice_idx      = None,    # None → middle slice
    )
"""
#%%
from __future__ import annotations

import os
from copy import deepcopy
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import MRzeroCore as mr0


# ──────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _sanitize(phantom: mr0.VoxelGridPhantom) -> mr0.VoxelGridPhantom:
    """Replace NaN/Inf in all maps (real and complex) with 0.

    combine() produces NaN in parameter maps where all tissue PDs are zero
    (0/0 weighted average). Those voxels are masked out by build() via
    PD_threshold, but NaN in B1 propagates into calc_avg_B1_trig() which
    runs before any mask is applied, crashing compute_graph().
    """
    for attr in ("T1", "T2", "T2dash", "D", "B0"):
        t = getattr(phantom, attr)
        setattr(phantom, attr, torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0))
    for attr in ("B1", "coil_sens"):
        t = getattr(phantom, attr)
        real = torch.nan_to_num(t.real, nan=0.0, posinf=0.0, neginf=0.0)
        imag = torch.nan_to_num(t.imag, nan=0.0, posinf=0.0, neginf=0.0)
        setattr(phantom, attr, torch.complex(real, imag))
    return phantom


def _pad_to_square_hw(phantom: mr0.VoxelGridPhantom) -> mr0.VoxelGridPhantom:
    """Zero-pad H and W to a square, scaling physical FOV proportionally.

    BrainWeb native shape is (362, 434, 362) for highres or (181, 217, 181)
    for standard: y is always the wide dimension. Padding x to match y keeps
    voxel pitch identical in both in-plane directions so a subsequent
    interpolate() call introduces no geometric distortion.
    """
    sx, sy, sz = phantom.PD.shape
    target = max(sx, sy)

    pad_x0 = (target - sx) // 2
    pad_x1 = target - sx - pad_x0
    pad_y0 = (target - sy) // 2
    pad_y1 = target - sy - pad_y0

    # F.pad order (reversed dims): z_before, z_after, y_before, y_after, x_before, x_after
    pad = (0, 0, pad_y0, pad_y1, pad_x0, pad_x1)

    def _pad(t: torch.Tensor) -> torch.Tensor:
        return F.pad(t, pad, mode="constant", value=0.0)

    def _pad_mc(t: torch.Tensor) -> torch.Tensor:
        return torch.stack([_pad(t[c]) for c in range(t.shape[0])], dim=0)

    new_size = phantom.size.clone()
    new_size[0] = phantom.size[0] * target / sx
    new_size[1] = phantom.size[1] * target / sy

    p = deepcopy(phantom)
    p.PD = _pad(phantom.PD)
    p.T1 = _pad(phantom.T1)
    p.T2 = _pad(phantom.T2)
    p.T2dash = _pad(phantom.T2dash)
    p.D = _pad(phantom.D)
    p.B0 = _pad(phantom.B0)
    p.B1 = _pad_mc(phantom.B1)
    p.coil_sens = _pad_mc(phantom.coil_sens)
    p.size = new_size
    if phantom.tissue_masks:
        p.tissue_masks = {k: _pad(v) for k, v in phantom.tissue_masks.items()}

    print(
        f"[square pad]  ({sx}, {sy}, {sz}) → {tuple(p.PD.shape)}, "
        f"FOV {[round(v * 1e3, 1) for v in phantom.size.tolist()]} mm "
        f"→ {[round(v * 1e3, 1) for v in new_size.tolist()]} mm"
    )
    return p


def _match_fov(
    phantom: mr0.VoxelGridPhantom,
    fov_mm: float,
) -> mr0.VoxelGridPhantom:
    """Crop or zero-pad each in-plane dimension so phantom.size == fov_mm.

    After _pad_to_square_hw, size[0] == size[1] but may differ from the
    sequence FOV. This step adjusts the voxel grid (keeping voxel pitch
    constant) so that phantom.size[0] == phantom.size[1] == fov_mm exactly.

    - Phantom FOV > sequence FOV  →  centre-crop (remove border voxels)
    - Phantom FOV < sequence FOV  →  zero-pad    (add border voxels)

    size[2] (z) is left unchanged.
    """
    fov_m = fov_mm * 1e-3
    sx, sy, sz = phantom.PD.shape

    # Current voxel pitch (same in x and y after _pad_to_square_hw)
    vox = phantom.size[0].item() / sx  # metres per voxel

    target_n = round(fov_m / vox)  # voxels needed to span fov_m
    delta = target_n - sx  # positive → need to add, negative → crop

    if delta == 0:
        print(f"[FOV match]   FOV already {fov_mm:.1f} mm — no adjustment needed")
        return phantom

    if delta > 0:
        # --- pad ---
        p0 = delta // 2
        p1 = delta - p0
        pad = (0, 0, p0, p1, p0, p1)  # same padding applied to both x and y

        def _adj(t):
            return F.pad(t, pad, mode="constant", value=0.0)

        def _adj_mc(t):
            return torch.stack([_adj(t[c]) for c in range(t.shape[0])], dim=0)

        action = f"pad {sx} → {target_n}"
    else:
        # --- centre crop ---
        c0 = (-delta) // 2
        c1 = c0 + target_n

        def _adj(t):
            return t[c0:c1, c0:c1, :]

        def _adj_mc(t):
            return t[:, c0:c1, c0:c1, :]

        action = f"crop {sx} → {target_n}"

    new_size = phantom.size.clone()
    new_size[0] = fov_m
    new_size[1] = fov_m

    p = deepcopy(phantom)
    p.PD = _adj(phantom.PD)
    p.T1 = _adj(phantom.T1)
    p.T2 = _adj(phantom.T2)
    p.T2dash = _adj(phantom.T2dash)
    p.D = _adj(phantom.D)
    p.B0 = _adj(phantom.B0)
    p.B1 = _adj_mc(phantom.B1)
    p.coil_sens = _adj_mc(phantom.coil_sens)
    p.size = new_size
    if phantom.tissue_masks:
        p.tissue_masks = {k: _adj(v) for k, v in phantom.tissue_masks.items()}

    print(
        f"[FOV match]   {action}, "
        f"size → [{fov_mm:.1f}, {fov_mm:.1f}, "
        f"{round(phantom.size[2].item() * 1e3, 1)}] mm"
    )
    return p


# ──────────────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_phantom(
    json_path: str,
    fov_mm: float = 224.0,
    resolution_mm: float = 224.0 / 96,
    slice_idx: int | None = None,
) -> tuple[mr0.VoxelGridPhantom, object, dict[str, torch.Tensor]]:

    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Phantom JSON not found: {json_path}")

    # 1. Load and combine tissues
    tissue_dict = mr0.TissueDict.load(json_path)
    tissue_dict.pop("fat", None)
    tissue_dict.pop("vessels", None)
    tissue_names = list(tissue_dict.keys())
    print(f"[load]        tissues : {tissue_names}")
    print(f"[load]        native  : {next(iter(tissue_dict.values())).PD.shape}")

    phantom = tissue_dict.combine()

    # Piggyback per-tissue PDs into tissue_masks so they ride through
    # all spatial transforms (_pad_to_square_hw, _match_fov, interpolate,
    # slices) on the exact same grid as phantom.PD.
    for name, tissue in tissue_dict.items():
        phantom.tissue_masks[name] = tissue.PD.clone()

    # 2. Sanitize
    phantom = _sanitize(phantom)

    # 3. Pad to square
    phantom = _pad_to_square_hw(phantom)

    # 4. Match FOV
    phantom = _match_fov(phantom, fov_mm)

    # 5. Compute grid
    nx = ny = round(fov_mm / resolution_mm)
    vox_z = (
        phantom.size[2].item()
        * 1e3
        / round(phantom.size[2].item() / (resolution_mm * 1e-3))
    )
    nz = round(phantom.size[2].item() / (resolution_mm * 1e-3))
    print(
        f"[grid]        NX=NY={nx}, NZ={nz}  "
        f"(voxel {resolution_mm:.4f} × {resolution_mm:.4f} × {vox_z:.3f} mm)"
    )

    # 6. Interpolate
    phantom = phantom.interpolate(nx, ny, nz)
    vox_mm = (phantom.size / torch.tensor(phantom.D.shape) * 1e3).tolist()
    print(
        f"[interpolate] shape: {tuple(phantom.D.shape)}, "
        f"voxel: {[round(v, 4) for v in vox_mm]} mm"
    )

    # 7. Slice
    if slice_idx is None:
        slice_idx = nz // 2
    phantom = phantom.slices([slice_idx])
    print(f"[slice]       {slice_idx}/{nz}")

    # 8. Extract per-tissue masks from tissue_masks — perfectly aligned
    #    since they went through the same transforms as phantom.PD.
    #    Use argmax for mutually exclusive tissue assignment.
    stacked = torch.stack(
        [phantom.tissue_masks[n].squeeze(-1).float() for n in tissue_names], dim=0
    )  # (N_tissues, NX, NY)
    assignments = stacked.argmax(dim=0)  # (NX, NY)
    tissue_masks: dict[str, torch.Tensor] = {}
    for i, name in enumerate(tissue_names):
        tissue_masks[name] = (assignments == i) & (stacked[i] > 0.05)

    print(f"[masks]       {tissue_names} — shape {stacked.shape[1:]}")
    print(f"[masks]       counts: { {n: tissue_masks[n].sum().item() for n in tissue_names} }")

    # 9. Build
    phantom_data = phantom.build()
    print(
        f"[build]       {phantom_data.PD.shape[0]} voxels, "
        f"PD sum = {phantom_data.PD.sum().item():.1f}"
    )

    # 10. Sanitize B1
    def _fix_complex(t: torch.Tensor, default_real: float = 1.0) -> torch.Tensor:
        real = torch.nan_to_num(t.real, nan=default_real, posinf=default_real, neginf=0.0)
        imag = torch.nan_to_num(t.imag, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.complex(real, imag)

    b1_nans = phantom_data.B1.isnan().sum().item()
    if b1_nans > 0:
        phantom_data.B1 = _fix_complex(phantom_data.B1, default_real=1.0)
        phantom_data.coil_sens = _fix_complex(phantom_data.coil_sens, default_real=1.0)
        from MRzeroCore.phantom.sim_data import calc_avg_B1_trig
        phantom_data.avg_B1_trig = calc_avg_B1_trig(phantom_data.B1, phantom_data.PD)
        print(
            f"[sanitize B1] fixed {b1_nans} NaN values in SimData.B1, "
            f"avg_B1_trig nans remaining: "
            f"{phantom_data.avg_B1_trig.isnan().sum().item()}"
        )

    return phantom, phantom_data, tissue_masks

# %%
phantom, phantom_data, tissue_masks = load_phantom(
    json_path=r'C:\Users\User\OneDrive\PhD\Sumbission\ESMRMB26\Pulseq-DiffusionMESE\brainweb_phantoms\brainweb-subj04\brainweb-subj04-3T.json',
    resolution_mm=2.333333333333333,
    fov_mm=224.0,
    slice_idx=None,
)

fig, axes = plt.subplots(1, 4, figsize=(12, 4))
axes[0].imshow(tissue_masks['wm'])
axes[0].set_title('WM')
axes[1].imshow(tissue_masks['gm'])
axes[1].set_title('GM')
axes[2].imshow(tissue_masks['csf'])
axes[2].set_title('CSF')
axes[3].imshow(phantom.PD.squeeze())
axes[3].set_title('PD')
plt.tight_layout()
plt.show()
# %%
