# Simulation output orientation

## Phantom frame

`phantom.D` (and all other parameter maps) have shape `(Nx, Ny, Nz)` after
`phantom.interpolate(nx, ny, nz)` in `phantom_loader.py`:

- axis 0 = **x** (L-R, readout direction, narrow dimension, ~181 mm)
- axis 1 = **y** (A-P, phase-encode direction, wide dimension, ~217 mm)
- axis 2 = **z** (slice / foot-head direction)

Per-slice access: `phantom.D[:, :, slice_idx]` → shape `(Nx, Ny)` — x-first.

Tissue masks from `phantom_loader.slice_preloaded_phantom` are also `(Nx, Ny)` after
`squeeze(-1)` — same x-first frame.

---

## Cartesian FFT reconstruction (multishot pipelines)

k-space is acquired row-by-row: Ny phase-encode lines × Nx readout samples.
After reshape to `(Ny, Nx)` and 2D IFFT:

```
image[y_row, x_col]   →   y-first (standard MRI display convention)
```

Relationship to phantom:

```
adc_nlls[iy, ix]  ≈  phantom.D[ix, iy, slice]
```

This is a transpose relative to the phantom frame, which is the standard
result of a Cartesian FFT (rows = PE = y, columns = RO = x).

---

## NUFFT reconstruction (EPI pipelines)

Trajectory is built as:

```python
traj = np.stack([kx_norm, ky_norm], axis=-1)   # shape (N_samples, 2)
op   = get_operator(samples=traj, shape=(Ny, Nx), ...)
```

`mrinufft` maps `samples[:, d]` to image dimension `d`. With `traj[:, 0] = kx`:

- image dim 0 (size Ny) ← **kx** (readout = **x** direction)
- image dim 1 (size Nx) ← **ky** (PE = **y** direction)

Raw `adj_op` output is therefore **x-first** (same frame as the phantom), the
opposite of the Cartesian FFT convention expected by the shared volume-save
pipeline.

**Fix applied** (all four EPI pipeline files):

```python
img_complex = op.adj_op(signal).squeeze().cpu().numpy().T
```

After `.T` (numpy transpose):

```
img_complex[iy, ix]  ≈  phantom.D[ix, iy, slice]
```

This matches the Cartesian FFT frame exactly, so both go through the same
volume-save pipeline without any additional per-pipeline branching.

---

## Volume NIfTI save pipeline (shared by all pipelines)

Input: `partial` of shape `(n_slices, Ny, Nx)` — the pre-allocated array
filled one slice at a time (y-first 2D maps).

```python
_m = np.rot90(partial, k=-1, axes=(1, 2))   # CW in image plane  →  (n_slices, Nx, Ny)
_m = np.transpose(_m, (1, 2, 0))            # slices last         →  (Nx, Ny, n_slices)
_save_volume_nifti(_m, path, res)
```

Saved with a diagonal affine `diag([res, res, res, 1])`:

| NIfTI axis | size | anatomical direction |
|-----------|------|----------------------|
| 0         | Nx   | x (L-R)              |
| 1         | Ny   | y (A-P)              |
| 2         | n_slices | z (I-S)          |

Net result:

```
nifti[ix, iy, iz]  =  phantom.D[ix, N-1-iy, slice_indices[iz]]
```

The y-axis is flipped relative to BrainWeb y. This matches real-scanner
conventions (anterior is at the top when viewed in ITK-SNAP's default axial
orientation).

---

## Reference volumes (D_ref, T2_ref)

Raw phantom shape: `(Nx, Ny, Nz)`.

**Fix applied** (`run_sim_volume.py`):

```python
_d_ref  = np.flip(preloaded.phantom.D.cpu().numpy(),  axis=1).copy()
_t2_ref = np.flip(preloaded.phantom.T2.cpu().numpy(), axis=1).copy()
```

After y-flip: `_d_ref[ix, iy, iz] = phantom.D[ix, N-1-iy, iz]`.
Saved directly as `(Nx, Ny, Nz)` NIfTI with the same diagonal affine.

Result matches the estimated-map NIfTI frame exactly — they overlay correctly
in ITK-SNAP.

---

## Tissue mask NIfTI volumes

Accumulated masks shape: `(n_slices, Nx, Ny)` — phantom x-first frame
(same as the raw 2D maps before the volume-save pipeline).

The same `rot90(k=-1, axes=(1,2))` + `transpose((1,2,0))` is applied before
saving, so:

```
mask_nifti[ix, iy, iz]  =  mask_phantom[ix, N-1-iy, slice_indices[iz]]
```

This matches the estimated-map NIfTI frame, so masks overlay correctly.

Output filenames: `{phantom_name}-mask_{tissue}_volume.nii.gz` in `paths.masks_dir`.
