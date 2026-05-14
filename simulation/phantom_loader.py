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

from __future__ import annotations

import os
from copy import deepcopy

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
) -> tuple[mr0.VoxelGridPhantom, object]:
    """Load a NIfTI BrainWeb phantom matched to a sequence FOV and resolution.

    The pipeline is:
        load → combine → sanitize → pad to square → match FOV → interpolate → slice

    Parameters
    ----------
    json_path : str
        Path to the ``phantom.json`` produced by ``mr0.generate_brainweb_phantoms()``.
    fov_mm : float
        In-plane field of view in mm, matching the sequence FOV (e.g. 224.0).
        The phantom is centre-cropped or zero-padded to this FOV before
        resampling so that phantom.size matches the sequence exactly.
    resolution_mm : float
        Isotropic voxel size in mm (e.g. 224/96 ≈ 2.333 for a 96-matrix).
        Together with fov_mm this determines the in-plane matrix:
        NX = NY = round(fov_mm / resolution_mm).
        NZ is chosen to give isotropic voxels over the full z extent.
    slice_idx : int | None
        Slice to select after interpolation (0-indexed).  None → middle slice.

    Returns
    -------
    phantom : mr0.VoxelGridPhantom
        Single-slice phantom whose .size matches the sequence FOV exactly.
    phantom_data : SimData
        Result of phantom.build().
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Phantom JSON not found: {json_path}")

    # 1. Load and combine tissues
    tissue_dict = mr0.TissueDict.load(json_path)
    tissue_dict.pop("fat", None)
    tissue_dict.pop("vessels", None)
    print(f"[load]        tissues : {list(tissue_dict.keys())}")
    print(f"[load]        native  : {next(iter(tissue_dict.values())).PD.shape}")

    phantom = tissue_dict.combine()

    # 2. Sanitize NaN/Inf from combine()'s 0/0 weighted average
    phantom = _sanitize(phantom)

    # 3. Pad in-plane to square (keeps voxel pitch equal in x and y)
    phantom = _pad_to_square_hw(phantom)

    # 4. Crop or pad to match sequence FOV exactly
    phantom = _match_fov(phantom, fov_mm)

    # 5. Compute matrix size from FOV and resolution
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

    # 6. Interpolate to simulation grid
    phantom = phantom.interpolate(nx, ny, nz)
    vox_mm = (phantom.size / torch.tensor(phantom.D.shape) * 1e3).tolist()
    print(
        f"[interpolate] shape: {tuple(phantom.D.shape)}, "
        f"voxel: {[round(v, 4) for v in vox_mm]} mm"
    )

    # 7. Select slice
    if slice_idx is None:
        slice_idx = nz // 2
    phantom = phantom.slices([slice_idx])
    print(f"[slice]       {slice_idx}/{nz}")

    phantom_data = phantom.build()
    print(
        f"[build]       {phantom_data.PD.shape[0]} voxels, "
        f"PD sum = {phantom_data.PD.sum().item():.1f}"
    )

    # Sanitize B1/coil_sens in SimData after build().
    # NaN can survive interpolation boundaries into brain-tissue voxels even
    # after pre-build sanitization. NaN real → 1.0 (nominal flip, no B1 info),
    # NaN imag → 0.0 (no phase offset). Then recompute avg_B1_trig.
    def _fix_complex(t: torch.Tensor, default_real: float = 1.0) -> torch.Tensor:
        real = torch.nan_to_num(
            t.real, nan=default_real, posinf=default_real, neginf=0.0
        )
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

    return phantom, phantom_data
