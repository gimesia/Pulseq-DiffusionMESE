"""Encompassing runner that executes all six qMRI simulation pipelines.

Pipelines dispatched
--------------------
| Key            | Module               | Sequence                              | Quantity |
|----------------|----------------------|---------------------------------------|----------|
| adc_sse        | sim_adc_sse          | EPIDiffusionSEPulseqSeq               | ADC      |
| t2_sse         | sim_t2_sse           | EPIDiffusionSEPulseqSeq (b=0)         | T2       |
| adc_multishot  | sim_adc_multishot    | DiffusionSEMultishotPulseqSeq         | ADC      |
| t2_multishot   | sim_t2_multishot     | DiffusionSEMultishotPulseqSeq (b=0)   | T2       |
| adc_triple     | sim_adc_triple       | EPIDiffusionTripleSEPulseqSeq         | ADC      |
| t2_triple      | sim_t2_triple        | EPIDiffusionTripleSEPulseqSeq (b=0)   | T2       |

Author      : Aron Gimesi <aron.gimesi@tecnico.ulisboa.pt>
Affiliation : Instituto Superior Técnico | MSCA-DN IQ-BRAIN
Date        : 2026
Context     : ESMRMB 2026 — Pulseq DiffusionMESE showcase

Funding acknowledgement (mandatory):
    IQ-BRAIN is funded by the European Union (MSCA Doctoral Network,
    December 2024–November 2028, Grant Agreement No. 101169519).

Typical usage
-------------
    from run_sim import run_all_qmri_simulations, PathConfig

    paths = PathConfig(
        seq_lib_path = r"C:\\...\\Pulseq-DiffusionMESE\\pulseq_diffusion_mese",
        phantoms_dir = r"C:\\...\\Pulseq-DiffusionMESE\\brainweb_phantoms",
        phantom_name = "brainweb-subj04",
        sequences_dir = r".\\simulated\\seq",
        volumes_dir   = r".\\simulated\\brainmaps",
        masks_dir     = r".\\simulated\\masks",
        diff_img_dir  = r".\\simulated\\diff_img",
        t2_img_dir    = r".\\simulated\\t2_img",
    )
    results = run_all_qmri_simulations(paths, slice_idx=None)
"""
from __future__ import annotations

import os
import time
from typing import Iterable, Optional, Sequence

import numpy as np

from utils_sim_lib import (
    PathConfig,
    PreloadedPhantom,
    extract_phantom_slice,
    load_phantom_for_sim,
    mae_after_registration,
    phantom_map_to_2d,
    save_magnitude_nifti,
)
from sim_adc_sse import run_adc_sse
from sim_t2_sse import run_t2_sse
from sim_adc_multishot import run_adc_multishot
from sim_t2_multishot import run_t2_multishot
from sim_adc_triple import run_adc_triple
from sim_t2_triple import run_t2_triple


PIPELINES = ("adc_sse", "t2_sse", "adc_multishot", "t2_multishot", "adc_triple", "t2_triple")


# Default TE list shared by the T2 SSE and T2 multishot pipelines.
DEFAULT_T2_TES = (65,  70,  75,  80,  85,  90,  95, 100, 105, 110, 115, 120, 123,
       128, 133, 138, 143, 148, 153, 158, 163, 168, 173, 178, 181, 186,
       191, 196, 201, 206, 211, 216, 221, 226, 231, 236)

# Default TE1 list for the triple SE T2 pipeline (each TR yields 3 echoes).
DEFAULT_T2_TRIPLE_TE1 = tuple(range(65, 125, 5))
DEFAULT_ADC_BVALUES = tuple(range(0, 2001, 100))


def run_all_qmri_simulations(
    paths: PathConfig,
    *,
    slice_idx: Optional[int] = None,
    system_type: Optional[object] = None,
    fov: float = 224e-3,
    res: float = 2.33333333,
    b_values: Sequence[int] = DEFAULT_ADC_BVALUES,
    TEs_t2: Sequence[int] = DEFAULT_T2_TES,
    TE1_values_triple: Sequence[int] = DEFAULT_T2_TRIPLE_TE1,
    TE_adc: int = 100,
    TR: int = 5000,
    b_directions: int = 6,
    small_delta: Optional[float] = None,
    big_DELTA: Optional[float] = None,
    small_delta_t2: float = 0.018,
    big_DELTA_t2: float = 0.03,
    ETL: int = 1,
    blip_down: bool = True,
    alternating_blip_polarity: bool = False,
    use_gpu: Optional[bool] = None,
    pipelines: Optional[Iterable[str]] = None,
    save_slice_npy: bool = True,
    preloaded_phantom: Optional[PreloadedPhantom] = None,
) -> dict:
    """Run all qMRI simulations and return their results in a dict.

    Parameters
    ----------
    paths:
        :class:`PathConfig` instance with all directory paths (incl. the
        path to the ``pulseq_diffusion_mese`` library).
    slice_idx:
        Phantom slice index. ``None`` → centre slice.
    system_type:
        ``utils.SystemLimitType`` value. ``None`` → ``EXTREME``.
    fov, res:
        Field-of-view (m) and isotropic in-plane resolution (mm).
    b_values:
        Diffusion b-values used by all ADC pipelines.
    TEs_t2:
        TE list used by the single-shot and multishot T2 pipelines.
    TE1_values_triple:
        TE1 list used by the triple-SE T2 pipeline. Each TR yields 3 echoes.
    TE_adc:
        Fixed TE / TE1 used by the ADC pipelines (Stejskal-Tanner: identical
        T2 weighting across the b-value series).
    TR:
        Repetition time [ms].
    b_directions:
        Number of diffusion-gradient directions for the ADC pipelines.
    small_delta, big_DELTA:
        Diffusion timing for the ADC pipelines (``None`` → auto-computed).
    small_delta_t2, big_DELTA_t2:
        Diffusion timing for the T2 pipelines (these use a non-zero default
        because the upstream sequences require it even at b=0).
    ETL:
        Echo-train length for the multishot pipelines (1 = conventional SE
        per shot).
    blip_down:
        EPI blip direction (affects geometric distortion direction).
    use_gpu:
        ``None`` → auto-detect via :func:`torch.cuda.is_available`.
    pipelines:
        Iterable of pipeline names to run. ``None`` → all of
        :data:`PIPELINES`.
    """
    if pipelines is None:
        pipelines = PIPELINES
    pipelines = tuple(pipelines)
    for name in pipelines:
        if name not in PIPELINES:
            raise ValueError(f"Unknown pipeline: {name!r}. Allowed: {PIPELINES}")

    paths.ensure_dirs()
    b_values = np.asarray(b_values, dtype=int)

    # Extract the phantom slice once and share it across all pipelines.
    # When called standalone (no preloaded phantom) load from disk now so that
    # (a) each pipeline does not reload independently and (b) reference maps
    # can be saved before any simulation runs.
    if preloaded_phantom is not None:
        phantom_slice = extract_phantom_slice(preloaded_phantom, slice_idx)
    else:
        phantom_slice = load_phantom_for_sim(paths, res, slice_idx)
        phantom_obj = phantom_slice[0]
        reference_D = phantom_map_to_2d(phantom_obj.D)
        reference_T2 = phantom_map_to_2d(phantom_obj.T2)
        save_magnitude_nifti(reference_D,  os.path.join(paths.volumes_dir, f"{paths.phantom_name}-D_ref.nii.gz"),  res)
        save_magnitude_nifti(reference_T2, os.path.join(paths.volumes_dir, f"{paths.phantom_name}-T2_ref.nii.gz"), res)
        print(f"[ref] saved D_ref and T2_ref  shape={reference_D.shape}")

    shared_kwargs = dict(
        slice_idx=slice_idx,
        system_type=system_type,
        fov=fov,
        res=res,
        TR=TR,
        use_gpu=use_gpu,
        save_slice_npy=save_slice_npy,
        phantom_slice=phantom_slice,
    )

    results: dict = {}
    timings: dict[str, float] = {}

    if "adc_sse" in pipelines:
        t_start = time.perf_counter()
        results["adc_sse"] = run_adc_sse(
            paths,
            b_values=b_values,
            TE=TE_adc,
            b_directions=b_directions,
            blip_down=blip_down,
            small_delta=small_delta,
            big_DELTA=big_DELTA,
            **shared_kwargs,
        )
        timings["adc_sse"] = time.perf_counter() - t_start

    if "t2_sse" in pipelines:
        t_start = time.perf_counter()
        results["t2_sse"] = run_t2_sse(
            paths,
            TEs=TEs_t2,
            b_value=0,
            blip_down=blip_down,
            small_delta=small_delta_t2,
            big_DELTA=big_DELTA_t2,
            **shared_kwargs,
        )
        timings["t2_sse"] = time.perf_counter() - t_start

    if "adc_multishot" in pipelines:
        t_start = time.perf_counter()
        results["adc_multishot"] = run_adc_multishot(
            paths,
            b_values=b_values,
            TE_values=(TE_adc,),
            TE_for_DTI=TE_adc,
            ETL=ETL,
            b_directions=b_directions,
            small_delta=small_delta,
            big_DELTA=big_DELTA,
            **shared_kwargs,
        )
        timings["adc_multishot"] = time.perf_counter() - t_start

    if "t2_multishot" in pipelines:
        t_start = time.perf_counter()
        results["t2_multishot"] = run_t2_multishot(
            paths,
            TEs=TEs_t2,
            ETL=ETL,
            **shared_kwargs,
        )
        timings["t2_multishot"] = time.perf_counter() - t_start

    if "adc_triple" in pipelines:
        t_start = time.perf_counter()
        results["adc_triple"] = run_adc_triple(
            paths,
            b_values=b_values,
            TE=TE_adc,
            b_directions=b_directions,
            blip_down=blip_down,
            small_delta=small_delta,
            big_DELTA=big_DELTA,
            alternating_blip_polarity=alternating_blip_polarity,
            **shared_kwargs,
        )
        timings["adc_triple"] = time.perf_counter() - t_start

    if "t2_triple" in pipelines:
        t_start = time.perf_counter()
        results["t2_triple"] = run_t2_triple(
            paths,
            TE1_values=TE1_values_triple,
            b_value=0,
            blip_down=blip_down,
            small_delta=small_delta_t2,
            big_DELTA=big_DELTA_t2,
            alternating_blip_polarity=alternating_blip_polarity,
            **shared_kwargs,
        )
        timings["t2_triple"] = time.perf_counter() - t_start

    # Post-hoc MAE: register each estimated map to the phantom reference, then
    # compute MAE per tissue. The reference is read from the per-pipeline
    # phantom snapshot; tissue masks come from the shared phantom_slice tuple.
    slice_tissue_masks = phantom_slice[2]
    for pipeline_name, pipeline_result in results.items():
        if pipeline_name == "_timings" or "phantom" not in pipeline_result:
            continue
        result_phantom = pipeline_result["phantom"]
        if pipeline_name.startswith("adc"):
            est_map = pipeline_result["adc_nlls"]
            ref_map = phantom_map_to_2d(result_phantom.D)
        else:
            est_map = pipeline_result["t2_nlls"]
            ref_map = phantom_map_to_2d(result_phantom.T2)
        mae = mae_after_registration(est_map, ref_map, slice_tissue_masks, register=True)
        pipeline_result["reference_map"] = ref_map
        pipeline_result["mae_per_tissue"]          = mae["per_tissue"]
        pipeline_result["mae_total"]               = mae["total"]
        pipeline_result["mae_n_voxels_per_tissue"] = mae["n_voxels_per_tissue"]
        pipeline_result["mae_n_voxels_total"]      = mae["n_voxels_total"]

    results["_timings"] = timings
    return results


if __name__ == "__main__":
    import os

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    paths = PathConfig(
        seq_lib_path  = os.path.join(project_root, "pulseq_diffusion_mese"),
        phantoms_dir  = os.path.join(project_root, "brainweb_phantoms"),
        phantom_name  = "brainweb-subj04",
        sequences_dir = os.path.join(project_root, "simulation", "simulated", "seq"),
        volumes_dir   = os.path.join(project_root, "simulation", "simulated", "brainmaps"),
        masks_dir     = os.path.join(project_root, "simulation", "simulated", "masks"),
        diff_img_dir  = os.path.join(project_root, "simulation", "simulated", "diff_img"),
        t2_img_dir    = os.path.join(project_root, "simulation", "simulated", "t2_img"),
    )

    results = run_all_qmri_simulations(
        paths,
        TE1_values_triple=range(65, 66, 5),   # (3 per TR)
        slice_idx=None,                    # None → centre slice
        b_values=range(0, 1000, 1000),      # fewer b-values to speed up testing
        b_directions=1,                    # fewer directions to speed up testing
        pipelines=("t2_triple", "adc_triple"),   # remove kwarg to run all 6 pipelines
    )

    for pipeline_name, pipeline_result in results.items():
        if pipeline_name.startswith("_"):
            continue
        per_tissue_summary = ", ".join(
            f"{tissue}={mae:.4f}" for tissue, mae in pipeline_result["mae_per_tissue"].items()
        )
        print(f"{pipeline_name:<14}  MAE={pipeline_result['mae_total']:.4f}  [{per_tissue_summary}]")
