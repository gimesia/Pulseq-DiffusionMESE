# IQ-BRAIN is funded by the European Union (MSCA Doctoral Network,
# December 2024–November 2028, Grant Agreement No. 101169519).
# EPI distortion correction for ADC mapping — mirrors epi_distortion_correctionT2.py

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

BVAL_DIR = './BVAL'
results_dir = r'C:\Users\User\OneDrive\PhD\Sumbission\ESMRMB26\Pulseq-DiffusionMESE\simulation\simulated\vol'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.float32
eps    = 1e-12

os.makedirs('topupADC', exist_ok=True)

# %%
# ============================================================================
# Triple SE (DiffTripleSE) — b-value loop, trace across directions × echoes
# Files in BVAL: b{b}-dir{d}-TE{te}-blipup/blipdown.nii.gz  (no sequence prefix)
# ============================================================================
all_bval_files = [f for f in os.listdir(BVAL_DIR)
                  if f.endswith('.nii.gz')
                  and not f.startswith('DiffSE')
                  and not f.startswith('trace')]

triple_bvals = np.unique([
    int(f.split('b')[1].split('-')[0])
    for f in all_bval_files if 'blipup' in f
])

triple_bu = []
triple_bd = []

for b_val in triple_bvals:
    print(f"\nTriple SE: Processing b={b_val} s/mm²...", flush=True)

    bu_files = sorted([f for f in all_bval_files if f.startswith(f'b{b_val}-') and 'blipup'   in f])
    bd_files = sorted([f for f in all_bval_files if f.startswith(f'b{b_val}-') and 'blipdown' in f])

    tes = np.unique([int(f.split('TE')[1].split('-')[0]) for f in bu_files])

    # Geometric mean across directions per echo, then arithmetic mean across echoes
    trace_bu_per_te = []
    trace_bd_per_te = []
    for te in tes:
        bu_te = np.array([nib.load(os.path.join(BVAL_DIR, f)).get_fdata().squeeze()
                          for f in bu_files if f'TE{te}' in f])
        bd_te = np.array([nib.load(os.path.join(BVAL_DIR, f)).get_fdata().squeeze()
                          for f in bd_files if f'TE{te}' in f])
        trace_bu_per_te.append(np.exp(np.mean(np.log(bu_te + eps), axis=0)))
        trace_bd_per_te.append(np.exp(np.mean(np.log(bd_te + eps), axis=0)))

    trace_bu = np.mean(trace_bu_per_te, axis=0)  # (Ny, Nx)
    trace_bd = np.mean(trace_bd_per_te, axis=0)

    affine = np.eye(4)
    trace_bu_path = os.path.join(BVAL_DIR, f'trace-DiffTripleSE-b{b_val}-blipup.nii.gz')
    trace_bd_path = os.path.join(BVAL_DIR, f'trace-DiffTripleSE-b{b_val}-blipdown.nii.gz')
    nib.save(nib.Nifti1Image(trace_bu[:, :, np.newaxis].astype(np.float32), affine), trace_bu_path)
    nib.save(nib.Nifti1Image(trace_bd[:, :, np.newaxis].astype(np.float32), affine), trace_bd_path)

    print("Shape of trace_bu:", nib.load(trace_bu_path).shape)
    print("Shape of trace_bd:", nib.load(trace_bd_path).shape)

    data = DataObject(
        img1=trace_bu_path,
        img2=trace_bd_path,
        phase_encoding_direction=2,
        device=device,
        dtype=dtype,
    )

    loss_func = EPIMRIDistortionCorrection(data, 1000, 1e-7, regularizer=myLaplacian1D, PC=JacobiCG)
    B0 = loss_func.initialize(blur_result=False)
    opt = GaussNewton(loss_func, max_iter=1500, verbose=True, path='topupADC/')
    opt.run_correction(B0)
    opt.apply_correction()

    path1_res    = rf'topupADC\-im1Corrected.nii.gz'
    path2_res    = rf'topupADC\-im2Corrected.nii.gz'
    fieldmap_res = rf'topupADC\-EstFieldMap.nii.gz'

    nib3 = nib.load(path1_res)
    nib4 = nib.load(path2_res)
    nib5 = nib.load(fieldmap_res)
    print("Shape of CORRECTED blip-up:",   nib3.shape)
    print("Shape of CORRECTED blip-down:", nib4.shape)
    print("Shape of field map:",           nib5.shape)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Triple SE Distortion Correction (b={b_val} s/mm²)")
    ax[0].imshow(nib3.get_fdata().squeeze(), cmap='gray')
    ax[0].set_title(f"Triple SE Corrected Blip-up\nb={b_val} s/mm²")
    ax[1].imshow(nib4.get_fdata().squeeze(), cmap='gray')
    ax[1].set_title(f"Triple SE Corrected Blip-down\nb={b_val} s/mm²")
    ax[2].imshow(nib5.get_fdata().squeeze(), cmap='seismic')
    ax[2].set_title(f"Triple SE Estimated Field Map\nb={b_val} s/mm²")
    ax[3].imshow(nib3.get_fdata().squeeze() - nib4.get_fdata().squeeze(), cmap='seismic')
    ax[3].set_title(f"Triple SE Difference\nb={b_val} s/mm²")
    plt.tight_layout()

    new_path1    = rf'topupADC\DiffTripleSE-b{b_val}-im1Corrected.nii.gz'
    new_path2    = rf'topupADC\DiffTripleSE-b{b_val}-im2Corrected.nii.gz'
    new_fieldmap = rf'topupADC\DiffTripleSE-b{b_val}-EstFieldMap.nii.gz'

    for p in [new_path1, new_path2, new_fieldmap]:
        if os.path.exists(p):
            os.remove(p)

    os.rename(path1_res,    new_path1)
    os.rename(path2_res,    new_path2)
    os.rename(fieldmap_res, new_fieldmap)

    triple_bu.append(nib.load(new_path1).get_fdata().squeeze())
    triple_bd.append(nib.load(new_path2).get_fdata().squeeze())

triple_bu = np.array(triple_bu)  # (n_b, Ny, Nx)
triple_bd = np.array(triple_bd)
print(f"Triple SE stack: {triple_bu.shape}  b-values: {len(triple_bvals)}")

# %%
# ============================================================================
# Single SE (DiffSE) — b-value loop, trace across directions (1 echo)
# Files in BVAL: DiffSE-b{b}-dir{d}-TE{te}-blipup/blipdown.nii.gz
# ============================================================================
single_se_files = [f for f in os.listdir(BVAL_DIR)
                   if f.endswith('.nii.gz') and f.startswith('DiffSE')]

single_bu = []
single_bd = []
single_bvals = np.array([], dtype=int)

if single_se_files:
    single_bvals = np.unique([
        int(f.split('b')[1].split('-')[0])
        for f in single_se_files if 'blipup' in f
    ])

    for b_val in single_bvals:
        print(f"\nSingle SE: Processing b={b_val} s/mm²...", flush=True)

        bu_files = sorted([f for f in single_se_files if f'b{b_val}-' in f and 'blipup'   in f])
        bd_files = sorted([f for f in single_se_files if f'b{b_val}-' in f and 'blipdown' in f])

        bu_imgs = np.array([nib.load(os.path.join(BVAL_DIR, f)).get_fdata().squeeze() for f in bu_files])
        bd_imgs = np.array([nib.load(os.path.join(BVAL_DIR, f)).get_fdata().squeeze() for f in bd_files])

        trace_bu = np.exp(np.mean(np.log(bu_imgs + eps), axis=0))
        trace_bd = np.exp(np.mean(np.log(bd_imgs + eps), axis=0))

        affine = np.eye(4)
        trace_bu_path = os.path.join(BVAL_DIR, f'trace-DiffSE-b{b_val}-blipup.nii.gz')
        trace_bd_path = os.path.join(BVAL_DIR, f'trace-DiffSE-b{b_val}-blipdown.nii.gz')
        nib.save(nib.Nifti1Image(trace_bu[:, :, np.newaxis].astype(np.float32), affine), trace_bu_path)
        nib.save(nib.Nifti1Image(trace_bd[:, :, np.newaxis].astype(np.float32), affine), trace_bd_path)

        print("Shape of trace_bu:", nib.load(trace_bu_path).shape)
        print("Shape of trace_bd:", nib.load(trace_bd_path).shape)

        data = DataObject(
            img1=trace_bu_path,
            img2=trace_bd_path,
            phase_encoding_direction=2,
            device=device,
            dtype=dtype,
        )

        loss_func = EPIMRIDistortionCorrection(data, 1000, 1e-7, regularizer=myLaplacian1D, PC=JacobiCG)
        B0 = loss_func.initialize(blur_result=False)
        opt = GaussNewton(loss_func, max_iter=1500, verbose=True, path='topupADC/')
        opt.run_correction(B0)
        opt.apply_correction()

        path1_res    = rf'topupADC\-im1Corrected.nii.gz'
        path2_res    = rf'topupADC\-im2Corrected.nii.gz'
        fieldmap_res = rf'topupADC\-EstFieldMap.nii.gz'

        nib3 = nib.load(path1_res)
        nib4 = nib.load(path2_res)
        nib5 = nib.load(fieldmap_res)
        print("Shape of CORRECTED blip-up:",   nib3.shape)
        print("Shape of CORRECTED blip-down:", nib4.shape)
        print("Shape of field map:",           nib5.shape)

        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f"Single SE Distortion Correction (b={b_val} s/mm²)")
        ax[0].imshow(nib3.get_fdata().squeeze(), cmap='gray')
        ax[0].set_title(f"Single SE Corrected Blip-up\nb={b_val} s/mm²")
        ax[1].imshow(nib4.get_fdata().squeeze(), cmap='gray')
        ax[1].set_title(f"Single SE Corrected Blip-down\nb={b_val} s/mm²")
        ax[2].imshow(nib5.get_fdata().squeeze(), cmap='seismic')
        ax[2].set_title(f"Single SE Estimated Field Map\nb={b_val} s/mm²")
        ax[3].imshow(nib3.get_fdata().squeeze() - nib4.get_fdata().squeeze(), cmap='seismic')
        ax[3].set_title(f"Single SE Difference\nb={b_val} s/mm²")
        plt.tight_layout()

        new_path1    = rf'topupADC\DiffSE-b{b_val}-im1Corrected.nii.gz'
        new_path2    = rf'topupADC\DiffSE-b{b_val}-im2Corrected.nii.gz'
        new_fieldmap = rf'topupADC\DiffSE-b{b_val}-EstFieldMap.nii.gz'

        for p in [new_path1, new_path2, new_fieldmap]:
            if os.path.exists(p):
                os.remove(p)

        os.rename(path1_res,    new_path1)
        os.rename(path2_res,    new_path2)
        os.rename(fieldmap_res, new_fieldmap)

        single_bu.append(nib.load(new_path1).get_fdata().squeeze())
        single_bd.append(nib.load(new_path2).get_fdata().squeeze())

    single_bu = np.array(single_bu)  # (n_b, Ny, Nx)
    single_bd = np.array(single_bd)
    print(f"Single SE stack: {single_bu.shape}  b-values: {len(single_bvals)}")
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
fig.suptitle("Distortion-Corrected ADC Maps (NLLS Fit)")
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
