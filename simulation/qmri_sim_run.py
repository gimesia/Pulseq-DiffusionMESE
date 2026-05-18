"""Encompassing runner that executes all six qMRI simulation pipelines.

Typical usage
-------------
    from qmri_sim_run import run_all_qmri_simulations, PathConfig

    paths = PathConfig(
        seq_lib_path = r"C:\\...\\Pulseq-DiffusionMESE\\pulseq_diffusion_mese",
        phantoms_dir = r"C:\\...\\Pulseq-DiffusionMESE\\brainweb_phantoms",
        phantom_name = "brainweb_phantom_0",
        sequences_dir = r".\\simulated\\seq",
        volumes_dir   = r".\\simulated\\brainmaps",
        masks_dir     = r".\\simulated\\masks",
        diff_img_dir  = r".\\simulated\\diff_img",
        t2_img_dir    = r".\\simulated\\t2_img",
    )
    results = run_all_qmri_simulations(paths, slice_idx=None)
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from qmri_sim_lib import PathConfig
from qmri_sim_adc_sse import run_adc_sse
from qmri_sim_t2_sse import run_t2_sse
from qmri_sim_adc_multishot import run_adc_multishot
from qmri_sim_t2_multishot import run_t2_multishot
from qmri_sim_adc_triple import run_adc_triple
from qmri_sim_t2_triple import run_t2_triple


PIPELINES = ("adc_sse", "t2_sse", "adc_multishot", "t2_multishot", "adc_triple", "t2_triple")


# Default TE list shared by the T2 SSE and T2 multishot pipelines.
DEFAULT_T2_TES = (
    65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 123, 125, 128, 130,
    133, 135, 138, 140, 143, 145, 148, 150, 153, 158, 163, 168, 173, 178, 181,
    183, 186, 188, 191, 193, 196, 198, 201, 203, 206, 208, 211, 216, 221, 226,
    231, 236, 241, 246, 251, 256, 261, 266,
)

# Default TE1 list for the triple SE T2 pipeline (each TR yields 3 echoes).
DEFAULT_T2_TRIPLE_TE1 = tuple(range(65, 155, 5))


def run_all_qmri_simulations(
    paths: PathConfig,
    *,
    slice_idx: Optional[int] = None,
    system_type=None,
    fov: float = 224e-3,
    res: float = 2.33333333,
    b_values: Sequence[int] = tuple(range(0, 2001, 100)),
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
    use_gpu: Optional[bool] = None,
    pipelines: Optional[Iterable[str]] = None,
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

    common = dict(
        slice_idx=slice_idx,
        system_type=system_type,
        fov=fov,
        res=res,
        TR=TR,
        use_gpu=use_gpu,
    )

    results: dict = {}

    if "adc_sse" in pipelines:
        results["adc_sse"] = run_adc_sse(
            paths,
            b_values=b_values,
            TE=TE_adc,
            b_directions=b_directions,
            blip_down=blip_down,
            small_delta=small_delta,
            big_DELTA=big_DELTA,
            **common,
        )

    if "t2_sse" in pipelines:
        results["t2_sse"] = run_t2_sse(
            paths,
            TEs=TEs_t2,
            b_value=0,
            blip_down=blip_down,
            small_delta=small_delta_t2,
            big_DELTA=big_DELTA_t2,
            **common,
        )

    if "adc_multishot" in pipelines:
        results["adc_multishot"] = run_adc_multishot(
            paths,
            b_values=b_values,
            TE_values=(TE_adc,),
            TE_for_DTI=TE_adc,
            ETL=ETL,
            b_directions=b_directions,
            small_delta=small_delta,
            big_DELTA=big_DELTA,
            **common,
        )

    if "t2_multishot" in pipelines:
        results["t2_multishot"] = run_t2_multishot(
            paths,
            TEs=TEs_t2,
            ETL=ETL,
            **common,
        )

    if "adc_triple" in pipelines:
        results["adc_triple"] = run_adc_triple(
            paths,
            b_values=b_values,
            TE=TE_adc,
            b_directions=b_directions,
            blip_down=blip_down,
            small_delta=small_delta,
            big_DELTA=big_DELTA,
            **common,
        )

    if "t2_triple" in pipelines:
        results["t2_triple"] = run_t2_triple(
            paths,
            TE1_values=TE1_values_triple,
            b_value=0,
            blip_down=blip_down,
            small_delta=small_delta_t2,
            big_DELTA=big_DELTA_t2,
            **common,
        )

    return results
