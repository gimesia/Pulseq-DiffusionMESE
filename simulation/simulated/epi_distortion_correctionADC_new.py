# IQ-BRAIN is funded by the European Union (MSCA Doctoral Network,
# December 2024–November 2028, Grant Agreement No. 101169519).
# EPI distortion correction for ADC mapping — mirrors epi_distortion_correctionT2.py
#
# STRATEGY: field map estimated ONCE from the first b-value (b=0 or lowest b),
#           then applied to all remaining b-values without re-optimization.
#
# PLOTTING: one consolidated figure per sequence type, one column per b-value,
#           rows: corrected blip-up | corrected blip-down | field map | difference

# %%
import os
import sys

sim_path = r"C:\Users\User\OneDrive\PhD\Sumbission\ESMRMB26\Pulseq-DiffusionMESE\simulation"
if sim_path not in sys.path:
    sys.path.append(sim_path)

from EPI_MRI.EPIMRIDistortionCorrection import *
from optimization.GaussNewton import *
import torch
from utils_diffusion import create_adc_map

BVAL_DIR = r'.\diff_img'
results_dir = r'C:\Users\User\OneDrive\PhD\Sumbission\ESMRMB26\Pulseq-DiffusionMESE\simulation\simulated\brainmaps'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.float32
eps    = 1e-12

os.makedirs('topupADC', exist_ok=True)


def plot_correction_summary(records, seq_label):
    """
    Plot a summary figure for one sequence type.

    Parameters
    ----------
    records : list of dicts with keys:
        'b_val'    : int
        'im1'      : np.ndarray  corrected blip-up
        'im2'      : np.ndarray  corrected blip-down
        'B0'       : torch.Tensor or np.ndarray  field map
        'ref'      : bool  True if field map was estimated for this b-value
    seq_label : str  e.g. 'DiffTripleSE' or 'DiffSE'
    """
    n_b = len(records)
    row_labels = ['Corrected blip-up', 'Corrected blip-down', 'Field map', 'Difference (bu − bd)']
    n_rows = len(row_labels)

    fig, axes = plt.subplots(n_rows, n_b, figsize=(5 * n_b, 4 * n_rows))
    # Ensure axes is always 2-D
    if n_b == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(f"{seq_label} — Distortion Correction Summary\n"
                 f"(★ = reference b-value where field map was estimated)",
                 fontsize=13, y=1.01)

    for col, rec in enumerate(records):
        b_val = rec['b_val']
        im1   = rec['im1']
        im2   = rec['im2']
        B0    = rec['B0']
        is_ref = rec['ref']

        if hasattr(B0, 'cpu'):          # torch tensor
            B0_np = B0.squeeze().cpu().numpy()
        else:                           # already numpy (loaded from NIfTI)
            B0_np = np.squeeze(B0)

        col_title = f"b = {b_val} s/mm²" + (" ★" if is_ref else "")

        axes[0, col].imshow(im1, cmap='gray')
        axes[0, col].set_title(col_title, fontsize=10)

        axes[1, col].imshow(im2, cmap='gray')

        axes[2, col].imshow(B0_np, cmap='seismic')

        diff = im1 - im2
        axes[3, col].imshow(diff, cmap='seismic',
                            vmin=-np.percentile(np.abs(diff), 99),
                            vmax= np.percentile(np.abs(diff), 99))

        for row in range(n_rows):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=10, labelpad=6)

    plt.tight_layout()
    fig.savefig(os.path.join('topupADC', f'{seq_label}_correction_summary.png'),
                dpi=150, bbox_inches='tight')
    plt.show()


def apply_fixed_fieldmap(bu_path, bd_path, B0_fixed, b_val, prefix, device, dtype):
    """
    Apply a pre-estimated field map (B0_fixed) to a blip-up/down pair
    without re-running the Gauss-Newton optimisation.

    Returns (corrected_bu_array, corrected_bd_array).
    """
    data = DataObject(
        img1=bu_path,
        img2=bd_path,
        phase_encoding_direction=2,
        device=device,
        dtype=dtype,
    )
    loss_func = EPIMRIDistortionCorrection(data, 1000, 1e-7,
                                           regularizer=myLaplacian1D, PC=JacobiCG)
    loss_func.initialize(blur_result=False)

    # Forward pass with fixed field map — populates loss_func.corr1 / corr2
    loss_func.eval(B0_fixed, do_derivative=False)

    opt = GaussNewton(loss_func, max_iter=0, verbose=False, path=r'topupADC/')
    opt.B0 = B0_fixed.clone()
    opt.Bc = B0_fixed.clone()
    opt.apply_correction()

    path1_res    = r'topupADC\-im1Corrected.nii.gz'
    path2_res    = r'topupADC\-im2Corrected.nii.gz'
    fieldmap_res = r'topupADC\-EstFieldMap.nii.gz'

    new_path1    = rf'topupADC\{prefix}-b{b_val}-im1Corrected.nii.gz'
    new_path2    = rf'topupADC\{prefix}-b{b_val}-im2Corrected.nii.gz'
    new_fieldmap = rf'topupADC\{prefix}-b{b_val}-EstFieldMap.nii.gz'

    for p in [new_path1, new_path2, new_fieldmap]:
        if os.path.exists(p):
            os.remove(p)

    os.rename(path1_res,    new_path1)
    os.rename(path2_res,    new_path2)
    os.rename(fieldmap_res, new_fieldmap)

    im1 = nib.load(new_path1).get_fdata().squeeze()
    im2 = nib.load(new_path2).get_fdata().squeeze()
    print(f"  [{prefix} b={b_val}] applied fixed field map → shape: {im1.shape}")
    return im1, im2


def run_and_save_topup(bu_path, bd_path, b_val, prefix, device, dtype):
    """
    Run full Gauss-Newton field-map estimation + correction for one b-value.

    Returns (corrected_bu_array, corrected_bd_array, B0_tensor).
    """
    data = DataObject(
        img1=bu_path,
        img2=bd_path,
        phase_encoding_direction=2,
        device=device,
        dtype=dtype,
    )
    loss_func = EPIMRIDistortionCorrection(data, 1000, 1e-7,
                                           regularizer=myLaplacian1D, PC=JacobiCG)
    # Use blur_result=True (default) so the OT initialisation is smoothed —
    # critical for b=0 where bu/bd are nearly identical and the raw OT
    # estimate fits noise, producing a pathological all-red field map.
    B0 = loss_func.initialize(blur_result=False)
    opt = GaussNewton(loss_func, max_iter=1500, verbose=True, path='topupADC/')
    opt.run_correction(B0)
    opt.apply_correction()

    path1_res    = r'topupADC\-im1Corrected.nii.gz'
    path2_res    = r'topupADC\-im2Corrected.nii.gz'
    fieldmap_res = r'topupADC\-EstFieldMap.nii.gz'

    new_path1    = rf'topupADC\{prefix}-b{b_val}-im1Corrected.nii.gz'
    new_path2    = rf'topupADC\{prefix}-b{b_val}-im2Corrected.nii.gz'
    new_fieldmap = rf'topupADC\{prefix}-b{b_val}-EstFieldMap.nii.gz'

    for p in [new_path1, new_path2, new_fieldmap]:
        if os.path.exists(p):
            os.remove(p)

    os.rename(path1_res,    new_path1)
    os.rename(path2_res,    new_path2)
    os.rename(fieldmap_res, new_fieldmap)

    im1 = nib.load(new_path1).get_fdata().squeeze()
    im2 = nib.load(new_path2).get_fdata().squeeze()
    print(f"  [{prefix} b={b_val}] estimated field map → shape: {im1.shape}")

    # Return opt.Bc — the converged field map tensor — for reuse
    return im1, im2, opt.Bc.detach().clone()


# %%
# ============================================================================
# Triple SE (DiffTripleSE) — field map from first (lowest) b-value only
# ============================================================================
all_bval_files = [f for f in os.listdir(BVAL_DIR)
                  if f.endswith('.nii.gz')
                  and not f.startswith('DiffSE')
                  and not f.startswith('trace')]

triple_bvals = np.unique([
    int(f.split('b')[1].split('-')[0])
    for f in all_bval_files if 'blipup' in f
])

triple_bu     = []
triple_bd     = []
triple_B0_ref = None
triple_records = []   # collected for consolidated plot

for i, b_val in enumerate(triple_bvals):
    print(f"\nTriple SE: Processing b={b_val} s/mm²"
          + (" [REFERENCE — estimating field map]" if i == 0
             else " [applying fixed field map]"),
          flush=True)

    bu_files = sorted([f for f in all_bval_files
                       if f.startswith(f'DiffTripleSE-b{b_val}-') and 'blipup'   in f])
    bd_files = sorted([f for f in all_bval_files
                       if f.startswith(f'DiffTripleSE-b{b_val}-') and 'blipdown' in f])

    tes = np.unique([int(f.split('TE')[1].split('-')[0]) for f in bu_files])

    trace_bu_per_te, trace_bd_per_te = [], []
    for te_idx, te in enumerate(tes):
        if te_idx:
            break
        bu_te = np.array([nib.load(os.path.join(BVAL_DIR, f)).get_fdata().squeeze()
                          for f in bu_files if f'TE{te}' in f])
        bd_te = np.array([nib.load(os.path.join(BVAL_DIR, f)).get_fdata().squeeze()
                          for f in bd_files if f'TE{te}' in f])
        trace_bu_per_te.append(np.exp(np.mean(np.log(bu_te + eps), axis=0)))
        trace_bd_per_te.append(np.exp(np.mean(np.log(bd_te + eps), axis=0)))

    trace_bu = np.mean(trace_bu_per_te, axis=0)
    trace_bd = np.mean(trace_bd_per_te, axis=0)

    affine = np.eye(4)
    trace_bu_path = os.path.join(BVAL_DIR, f'trace-DiffTripleSE-b{b_val}-blipup.nii.gz')
    trace_bd_path = os.path.join(BVAL_DIR, f'trace-DiffTripleSE-b{b_val}-blipdown.nii.gz')
    nib.save(nib.Nifti1Image(trace_bu[:, :, np.newaxis].astype(np.float32), affine), trace_bu_path)
    nib.save(nib.Nifti1Image(trace_bd[:, :, np.newaxis].astype(np.float32), affine), trace_bd_path)

    if i == 0:
        im1, im2, triple_B0_ref = run_and_save_topup(
            trace_bu_path, trace_bd_path, b_val, 'DiffTripleSE', device, dtype)
        B0_for_record = triple_B0_ref
    else:
        im1, im2 = apply_fixed_fieldmap(
            trace_bu_path, trace_bd_path, triple_B0_ref, b_val, 'DiffTripleSE', device, dtype)
        B0_for_record = triple_B0_ref   # same field map shown for all non-ref b-values

    triple_bu.append(im1)
    triple_bd.append(im2)
    triple_records.append({'b_val': b_val, 'im1': im1, 'im2': im2,
                           'B0': B0_for_record, 'ref': (i == 0)})

triple_bu = np.array(triple_bu)
triple_bd = np.array(triple_bd)
print(f"\nTriple SE stack: {triple_bu.shape}  b-values: {triple_bvals}")

plot_correction_summary(triple_records, 'DiffTripleSE')


# %%
# ============================================================================
# Single SE (DiffSE) — field map from first (lowest) b-value only
# ============================================================================
single_se_files = [f for f in os.listdir(BVAL_DIR)
                   if f.endswith('.nii.gz') and f.startswith('DiffSE')]

single_bu      = []
single_bd      = []
single_bvals   = np.array([], dtype=int)
single_records = []

if single_se_files:
    single_bvals = np.unique([
        int(f.split('b')[1].split('-')[0])
        for f in single_se_files if 'blipup' in f
    ])

    single_B0_ref = None

    for i, b_val in enumerate(single_bvals):
        print(f"\nSingle SE: Processing b={b_val} s/mm²"
              + (" [REFERENCE — estimating field map]" if i == 0
                 else " [applying fixed field map]"),
              flush=True)

        bu_files = sorted([f for f in single_se_files
                           if f'b{b_val}-' in f and 'blipup'   in f])
        bd_files = sorted([f for f in single_se_files
                           if f'b{b_val}-' in f and 'blipdown' in f])

        bu_imgs = np.array([nib.load(os.path.join(BVAL_DIR, f)).get_fdata().squeeze()
                            for f in bu_files])
        bd_imgs = np.array([nib.load(os.path.join(BVAL_DIR, f)).get_fdata().squeeze()
                            for f in bd_files])

        trace_bu = np.exp(np.mean(np.log(bu_imgs + eps), axis=0))
        trace_bd = np.exp(np.mean(np.log(bd_imgs + eps), axis=0))

        affine = np.eye(4)
        trace_bu_path = os.path.join(BVAL_DIR, f'trace-DiffSE-b{b_val}-blipup.nii.gz')
        trace_bd_path = os.path.join(BVAL_DIR, f'trace-DiffSE-b{b_val}-blipdown.nii.gz')
        nib.save(nib.Nifti1Image(trace_bu[:, :, np.newaxis].astype(np.float32), affine), trace_bu_path)
        nib.save(nib.Nifti1Image(trace_bd[:, :, np.newaxis].astype(np.float32), affine), trace_bd_path)

        if i == 0:
            im1, im2, single_B0_ref = run_and_save_topup(
                trace_bu_path, trace_bd_path, b_val, 'DiffSE', device, dtype)
            B0_for_record = single_B0_ref
        else:
            im1, im2 = apply_fixed_fieldmap(
                trace_bu_path, trace_bd_path, single_B0_ref, b_val, 'DiffSE', device, dtype)
            B0_for_record = single_B0_ref

        single_bu.append(im1)
        single_bd.append(im2)
        single_records.append({'b_val': b_val, 'im1': im1, 'im2': im2,
                                'B0': B0_for_record, 'ref': (i == 0)})

    single_bu = np.array(single_bu)
    single_bd = np.array(single_bd)
    print(f"\nSingle SE stack: {single_bu.shape}  b-values: {single_bvals}")

    plot_correction_summary(single_records, 'DiffSE')
else:
    print("No DiffSE files found in BVAL — skipping single SE distortion correction.")
    print("Run qMRI_adc.py (with NIfTI saving) to generate DiffSE blip-up/down files first.")


# %%
# ============================================================================
# ADC fitting from distortion-corrected trace DWI
# ============================================================================
adc_results = {
    'triple_bu': create_adc_map(triple_bu, triple_bvals, method='nlls')[0],
    'triple_bd': create_adc_map(triple_bd, triple_bvals, method='nlls')[0],
}
if len(single_bu) > 0:
    adc_results['single_bu'] = create_adc_map(np.array(single_bu), single_bvals, method='nlls')[0]
    adc_results['single_bd'] = create_adc_map(np.array(single_bd), single_bvals, method='nlls')[0]

n_maps = len(adc_results)
fig, axes = plt.subplots(1, n_maps, figsize=(6 * n_maps, 6))
if n_maps == 1:
    axes = [axes]
titles = {
    'triple_bu': "Triple SE (blip-up) ADC Map (NLLS)",
    'triple_bd': "Triple SE (blip-down) ADC Map (NLLS)",
    'single_bu': "Single SE (blip-up) ADC Map (NLLS)",
    'single_bd': "Single SE (blip-down) ADC Map (NLLS)",
}
for ax, (key, adc_map) in zip(axes, adc_results.items()):
    im = ax.imshow(np.rot90(adc_map, -1) * 1e3, cmap='viridis')
    ax.set_title(titles[key])
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, label="ADC (×10⁻³ mm²/s)")
fig.suptitle("Distortion-Corrected ADC Maps (NLLS Fit, single field map per sequence)")
plt.tight_layout()


# %%
# ============================================================================
# Save outputs
# ============================================================================
np.save(os.path.join(results_dir, 'ADC_triple_se_distcorr_blipup.npy'),
        np.rot90(adc_results['triple_bu'], -1))
np.save(os.path.join(results_dir, 'ADC_triple_se_distcorr_blipdown.npy'),
        np.rot90(adc_results['triple_bd'], -1))

if 'single_bu' in adc_results:
    np.save(os.path.join(results_dir, 'ADC_single_se_distcorr_blipup.npy'),
            np.rot90(adc_results['single_bu'], -1))
    np.save(os.path.join(results_dir, 'ADC_single_se_distcorr_blipdown.npy'),
            np.rot90(adc_results['single_bd'], -1))

print(f"Saved results to: {results_dir}")


# %%
# ============================================================================
# ADC map field correction
#
# Two approaches applied to each sequence type:
#   A) Blip-up vs blip-down ADC map correction — treat the two ADC maps as a
#      bu/bd pair and run a fresh GN optimisation against each other.
#   B) Apply saved b=0 field map — load the NIfTI field map already estimated
#      from the b=0 DWI and warp both ADC maps with it directly, no re-optimisation.
# ============================================================================

def load_B0_from_nifti(fieldmap_path, device, dtype):
    """Load a saved field map NIfTI and return it as a torch tensor."""
    arr = nib.load(fieldmap_path).get_fdata().squeeze().astype(np.float32)
    return torch.tensor(arr, device=device, dtype=dtype)


def warp_adc_geometric(adc_path, B0, loss_func, polarity=+1):
    """
    Apply a purely geometric warp to a quantitative map — NO Jacobian modulation.

    For quantitative maps (ADC, T2, …) the Jacobian factor (1 + dbc) used in
    mp_transform must be omitted because it would rescale the values.  We
    replicate what mp_transform does internally but return FI (pure geometric
    interpolation) instead of TI = FI * Jac.

    Parameters
    ----------
    adc_path  : str           path to the ADC NIfTI to warp
    B0        : torch.Tensor  field map (size m_plus(m)), on the correct device
    loss_func : EPIMRIDistortionCorrection  already initialised with ANY image
                pair of the right size/domain — we only need its A operator and xc
    polarity  : +1 for blip-up (warp by +B0), -1 for blip-down (warp by -B0)

    Returns
    -------
    warped : np.ndarray  corrected ADC map, same shape as input
    """
    from EPI_MRI.ImageModels import Interp1D

    # Load ADC map and build an interpolation model on the same domain
    arr = nib.load(adc_path).get_fdata().squeeze().astype(np.float32)
    adc_tensor = torch.tensor(arr, device=B0.device, dtype=B0.dtype)
    I_adc = Interp1D(adc_tensor, loss_func.dataObj.omega,
                     loss_func.dataObj.m,
                     dtype=B0.dtype, device=B0.device)

    # Compute displacement: bc = A * (polarity * B0)
    bc = loss_func.A.mat_mul(polarity * B0)

    # Displaced grid — purely geometric, no Jacobian
    xt = loss_func.xc + bc

    # Interpolate ADC at displaced coordinates (FI only, no * Jac)
    FI = I_adc.eval(xt)

    # Permute back to original image orientation and return as numpy
    warped = FI.reshape(list(loss_func.dataObj.m)) \
               .permute(loss_func.dataObj.p) \
               .cpu().numpy()
    return warped


def correct_adc_maps_geometric(bu_path, bd_path, prefix, approach, device, dtype,
                                B0_fixed=None):
    """
    Geometrically warp a pair of ADC maps using a field map.

    Approach 'optimise': estimate field map by running GN on the ADC pair,
                         then apply geometric warp (no Jacobian modulation).
    Approach 'fixed'   : use a pre-estimated B0_fixed directly.

    Returns (adc_bu_corr, adc_bd_corr, B0_np) — all value-preserving.
    """
    out_dir = os.path.join(results_dir, 'adc_fieldcorr')
    os.makedirs(out_dir, exist_ok=True)

    data = DataObject(
        img1=bu_path,
        img2=bd_path,
        phase_encoding_direction=2,
        device=device,
        dtype=dtype,
    )
    loss_func = EPIMRIDistortionCorrection(data, 1000, 1e-7,
                                           regularizer=myLaplacian1D, PC=JacobiCG)

    if approach == 'optimise':
        # Estimate field map from the ADC pair itself
        B0_init = loss_func.initialize(blur_result=False)
        tmp_path = os.path.join(out_dir, f'_tmp_{prefix}_opt') + os.sep
        os.makedirs(tmp_path, exist_ok=True)
        opt = GaussNewton(loss_func, max_iter=1500, verbose=True, path=tmp_path)
        opt.run_correction(B0_init)
        B0_est = opt.Bc.detach().clone()
        tag = 'adc_optimised'
        # Save the estimated field map for reference
        fm_path = os.path.join(out_dir, f'{prefix}_{tag}_fieldmap.nii.gz')
        save_data(B0_est.reshape(list(m_plus(loss_func.dataObj.m)))
                       .permute(loss_func.dataObj.p), fm_path)

    elif approach == 'fixed':
        if B0_fixed is None:
            raise ValueError("B0_fixed must be provided for approach='fixed'")
        B0_est = B0_fixed
        tag = 'b0_fixed'
        fm_path = os.path.join(out_dir, f'{prefix}_{tag}_fieldmap.nii.gz')
        save_data(B0_est.reshape(list(m_plus(loss_func.dataObj.m)))
                       .permute(loss_func.dataObj.p), fm_path)
    else:
        raise ValueError(f"Unknown approach: {approach}")

    # Geometric warp only — values are preserved
    loss_func.initialize(blur_result=False)   # ensures xc and A are ready
    adc_bu_corr = warp_adc_geometric(bu_path, B0_est, loss_func, polarity=+1)
    adc_bd_corr = warp_adc_geometric(bd_path, B0_est, loss_func, polarity=-1)

    fm_np = nib.load(fm_path).get_fdata().squeeze()

    # Save corrected maps
    affine = nib.load(bu_path).affine
    nib.save(nib.Nifti1Image(adc_bu_corr.astype(np.float32), affine),
             os.path.join(out_dir, f'{prefix}_{tag}_blipup.nii.gz'))
    nib.save(nib.Nifti1Image(adc_bd_corr.astype(np.float32), affine),
             os.path.join(out_dir, f'{prefix}_{tag}_blipdown.nii.gz'))

    print(f"  [{prefix} | {tag}] geometric warp done, shape: {adc_bu_corr.shape}  "
          f"value range bu: [{adc_bu_corr.min():.4f}, {adc_bu_corr.max():.4f}]")
    return adc_bu_corr, adc_bd_corr, fm_np


def plot_adc_correction(adc_bu_raw, adc_bd_raw,
                        adc_bu_opt, adc_bd_opt, fm_opt,
                        adc_bu_fix, adc_bd_fix, fm_fix,
                        seq_label):
    """
    Summary figure: 3 rows × 6 columns.
      Row 0: ADC maps (shared viridis scale from raw data)
      Row 1: field maps (optimised col-2, fixed col-4; others blank)
      Row 2: difference bu − bd (shared seismic scale per group)
    """
    scale = 1e3   # → ×10⁻³ mm²/s

    def _rot(x): return np.rot90(x, -1)

    # Shared colour limits for ADC row — use raw data range so all cols match
    raw_finite = np.concatenate([adc_bu_raw[np.isfinite(adc_bu_raw)],
                                  adc_bd_raw[np.isfinite(adc_bd_raw)]])
    adc_vmin = np.percentile(raw_finite, 1)  * scale
    adc_vmax = np.percentile(raw_finite, 99) * scale

    col_specs = [
        ('Raw blip-up',          _rot(adc_bu_raw), _rot(adc_bd_raw)),
        ('Raw blip-down',        _rot(adc_bd_raw), _rot(adc_bd_raw)),   # placeholder
        ('Opt. corr. blip-up',   _rot(adc_bu_opt), _rot(adc_bd_opt)),
        ('Opt. corr. blip-down', _rot(adc_bd_opt), _rot(adc_bd_opt)),
        ('Fixed-B0 blip-up',     _rot(adc_bu_fix), _rot(adc_bd_fix)),
        ('Fixed-B0 blip-down',   _rot(adc_bd_fix), _rot(adc_bd_fix)),
    ]
    adc_maps  = [_rot(adc_bu_raw), _rot(adc_bd_raw),
                 _rot(adc_bu_opt), _rot(adc_bd_opt),
                 _rot(adc_bu_fix), _rot(adc_bd_fix)]
    diff_maps = [_rot(adc_bu_raw - adc_bd_raw), None,
                 _rot(adc_bu_opt - adc_bd_opt), None,
                 _rot(adc_bu_fix - adc_bd_fix), None]
    fm_maps   = [None, None, _rot(fm_opt), None, _rot(fm_fix), None]
    col_titles = ['Raw blip-up', 'Raw blip-down',
                  'Opt. corr. blip-up', 'Opt. corr. blip-down',
                  'Fixed-B0 blip-up',   'Fixed-B0 blip-down']

    n_cols, n_rows = 6, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))
    fig.suptitle(f"{seq_label} — ADC Map Field Correction  "
                 "(geometric warp, values preserved)\n"
                 "Raw  |  Optimised (ADC-pair GN)  |  Fixed B0 (from b=0 DWI)",
                 fontsize=12, y=1.02)

    row_labels = ['ADC (×10⁻³ mm²/s)', 'Field map', 'Difference bu−bd (×10⁻³)']

    for col in range(n_cols):
        # Row 0: ADC maps — shared scale
        ax = axes[0, col]
        im = ax.imshow(adc_maps[col] * scale, cmap='viridis',
                       vmin=adc_vmin, vmax=adc_vmax)
        ax.set_title(col_titles[col], fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([]); ax.set_yticks([])

        # Row 1: field maps — only cols 2 and 4
        ax = axes[1, col]
        if fm_maps[col] is not None:
            abs_max = np.percentile(np.abs(fm_maps[col][np.isfinite(fm_maps[col])]), 99)
            im = ax.imshow(fm_maps[col], cmap='seismic', vmin=-abs_max, vmax=abs_max)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks([]); ax.set_yticks([])
        else:
            ax.set_visible(False)

        # Row 2: difference maps — cols 0, 2, 4
        ax = axes[2, col]
        if diff_maps[col] is not None:
            d = diff_maps[col] * scale
            abs_max = np.percentile(np.abs(d[np.isfinite(d)]), 99)
            im = ax.imshow(d, cmap='seismic', vmin=-abs_max, vmax=abs_max)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks([]); ax.set_yticks([])
        else:
            ax.set_visible(False)

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=9, labelpad=6)

    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, 'adc_fieldcorr',
                             f'{seq_label}_adc_correction_summary.png'),
                dpi=150, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# Paths to the pre-computed ADC NIfTIs
# ---------------------------------------------------------------------------
adc_nifti = {
    'triple_bu': os.path.join(results_dir, 'ADC_triple_se_distcorr_blipup.nii.gz'),
    'triple_bd': os.path.join(results_dir, 'ADC_triple_se_distcorr_blipdown.nii.gz'),
    'single_bu': os.path.join(results_dir, 'ADC_single_se_distcorr_blipup.nii.gz'),
    'single_bd': os.path.join(results_dir, 'ADC_single_se_distcorr_blipdown.nii.gz'),
}

# Saved b=0 field maps from the DWI correction run earlier
b0_fieldmap_paths = {
    'triple': r'topupADC\DiffTripleSE-b0-EstFieldMap.nii.gz',
    'single': r'topupADC\DiffSE-b0-EstFieldMap.nii.gz',
}

# ---------------------------------------------------------------------------
# Triple SE ADC correction
# ---------------------------------------------------------------------------
if all(os.path.exists(adc_nifti[k]) for k in ('triple_bu', 'triple_bd')):

    # Approach A: optimise on the ADC pair itself (geometric warp, no Jacobian)
    adc_triple_bu_opt, adc_triple_bd_opt, fm_triple_opt = correct_adc_maps_geometric(
        adc_nifti['triple_bu'], adc_nifti['triple_bd'],
        prefix='ADC_triple_se', approach='optimise',
        device=device, dtype=dtype)

    # Approach B: apply saved b=0 field map (geometric warp, no Jacobian)
    B0_triple = load_B0_from_nifti(b0_fieldmap_paths['triple'], device, dtype)
    adc_triple_bu_fix, adc_triple_bd_fix, fm_triple_fix = correct_adc_maps_geometric(
        adc_nifti['triple_bu'], adc_nifti['triple_bd'],
        prefix='ADC_triple_se', approach='fixed',
        device=device, dtype=dtype, B0_fixed=B0_triple)

    adc_triple_bu_raw = nib.load(adc_nifti['triple_bu']).get_fdata().squeeze()
    adc_triple_bd_raw = nib.load(adc_nifti['triple_bd']).get_fdata().squeeze()

    plot_adc_correction(
        adc_triple_bu_raw, adc_triple_bd_raw,
        adc_triple_bu_opt, adc_triple_bd_opt, fm_triple_opt,
        adc_triple_bu_fix, adc_triple_bd_fix, fm_triple_fix,
        seq_label='DiffTripleSE')
else:
    print("Triple SE ADC NIfTIs not found — skipping ADC field correction.")

# ---------------------------------------------------------------------------
# Single SE ADC correction
# ---------------------------------------------------------------------------
if all(os.path.exists(adc_nifti[k]) for k in ('single_bu', 'single_bd')):

    adc_single_bu_opt, adc_single_bd_opt, fm_single_opt = correct_adc_maps_geometric(
        adc_nifti['single_bu'], adc_nifti['single_bd'],
        prefix='ADC_single_se', approach='optimise',
        device=device, dtype=dtype)

    B0_single = load_B0_from_nifti(b0_fieldmap_paths['single'], device, dtype)
    adc_single_bu_fix, adc_single_bd_fix, fm_single_fix = correct_adc_maps_geometric(
        adc_nifti['single_bu'], adc_nifti['single_bd'],
        prefix='ADC_single_se', approach='fixed',
        device=device, dtype=dtype, B0_fixed=B0_single)

    adc_single_bu_raw = nib.load(adc_nifti['single_bu']).get_fdata().squeeze()
    adc_single_bd_raw = nib.load(adc_nifti['single_bd']).get_fdata().squeeze()

    plot_adc_correction(
        adc_single_bu_raw, adc_single_bd_raw,
        adc_single_bu_opt, adc_single_bd_opt, fm_single_opt,
        adc_single_bu_fix, adc_single_bd_fix, fm_single_fix,
        seq_label='DiffSE')
else:
    print("Single SE ADC NIfTIs not found — skipping ADC field correction.")

# %%