"""
EPIDiffusionTripleSEPulseqSeq — Diffusion-weighted triple spin-echo EPI sequence.

Implements a Stejskal-Tanner diffusion preparation (one pair of bipolar gradients
flanking the first 180° pulse) followed by three successive spin echoes, each read
out with a separate :class:`EPIReadout` instance.  The three echoes share a single
excitation and produce images at TE1, TE2, and TE3 within a single TR — enabling
simultaneous multi-contrast (e.g. T2* mapping alongside DWI) or signal averaging.

Inherits directly from :class:`PulseqSeq` (flat hierarchy — no intermediate classes).
Uses :class:`EPIReadout` (standalone, no inheritance) for all three readout trains.

Key features:
    - Automatic TE feasibility checking with partial-Fourier reduction fallback
    - Hardware amplitude clamping with achieved-b-value reporting when b exceeds limits
    - Optional YXY RF-phase cycling across the three 180° pulses
    - Optional navigator / calibration readout (3-line spin echo before the DWI loop)
    - Configurable spoiler strategy: varied area / varied axis / uniform
    - Blip-down/up polarity per echo (supports alternating for B0-distortion correction)
    - Labeled and unlabeled (simulation) output modes

Author      : Aron Gimesi <aron.gimesi@tecnico.ulisboa.pt>
Affiliation : Instituto Superior Técnico | MSCA-DN IQ-BRAIN
Date        : 2026
Context     : ESMRMB 2026 — Pulseq DiffusionMESE showcase

Funding acknowledgement (mandatory):
    IQ-BRAIN is funded by the European Union (MSCA Doctoral Network,
    December 2024-November 2028, Grant Agreement No. 101169519).
"""

# %%
from pathlib import Path
import sys

import logging
import os
import numpy as np
import pypulseq as pp


from PulseqSeq import *
from EPIReadout import EPIReadout


class EPIDiffusionTripleSEPulseqSeq(PulseqSeq):
    """Diffusion-weighted triple spin-echo EPI sequence.

    Flat hierarchy: inherits only from :class:`PulseqSeq`.

    The sequence timeline for each diffusion direction within one TR is::

        RF90 → gz90_reph → [delay] → Gdiff1 → [delay] → RF180_1 + spoilers
             → Gdiff2 → [delay] → EPI1-prephase → EPI1 (echo at TE1)
             → [delay] → spoiler2 → RF180_2 → spoiler2
             → EPI2-prephase → [delay] → EPI2 (echo at TE2)
             → [delay] → spoiler3 → RF180_3 → spoiler3
             → EPI3-prephase → [delay] → EPI3 (echo at TE3)
             → [end-spoilers] → delayTR

    Key timing attributes (all in seconds):
        TE (float)  : First echo time (RF90 centre → EPI1 k-space centre).
        TE2 (float) : Second echo time (RF90 centre → EPI2 k-space centre), rounded up
                      to the nearest ms.
        TE3 (float) : Third echo time (RF90 centre → EPI3 k-space centre).
        delayTR (float)              : Padding delay at the end of each TR.
        delay_before_diff1 (float)   : Delay between gz90_reph and first diffusion gradient.
        delayTE1_inner (float)       : Delay between first diffusion gradient and RF180_1.
        delayTE2 (float)             : Delay between second diffusion gradient and EPI1 prephaser.
        delay_before_rf180_2 (float) : Delay between EPI1 end and RF180_2.
        delay_before_epi2 (float)    : Delay between EPI2 prephaser and EPI2 start.
        delay_before_rf180_3 (float) : Delay between EPI2 end and RF180_3.
        delay_before_epi3 (float)    : Delay between EPI3 prephaser and EPI3 start.
    """

    def __init__(
        self,
        name: str,
        fov: float,  # in meters
        Nx: int,
        Ny: int,
        slice_thickness: float,
        TR: int,  # in milliseconds
        TE: int,  # in milliseconds
        b_value: int = 0,
        b_directions: int = 0,
        b_0_frequency: int = 0,
        small_delta: float = None,  # in seconds
        big_DELTA: float = None,  # in seconds
        N_slices: int = 1,
        system_type=SystemLimitType.EXTRASAFE,
        rf90_duration: float = 0.003,  # in seconds
        rf180_duration: float = 0,  # in seconds
        resolution: float = None,
        flip_angle: int = 90,
        apodization: float = 0.5,
        time_bw_product: float = 4,
        dwell_time: float = None,  # in seconds
        end_spoilers: bool = False,
        rf180_spoiler: bool = False,
        spoiler_amplitude: float = 1,
        spoiler_duration: float = 1e-3,  # in seconds
        prephaser_duration: float = None,  # in seconds
        rephasers: bool = True,
        simultan_rephasers: bool = True,
        blip_down: bool = True,
        alternating_blip_polarity: bool = False,
        ramp_sampling: str = "ramp_sampled",  # 'none', 'optimized', 'ramp_sampled'
        save_dir: str = DEFAULT_SAVE_DIR,
        logger: logging.Logger = None,
        partial_fourier_factor: float = 1.0,
        calibration_readout: bool = False,
        acceleration_factor: int = 1,
        max_EPI_duration: float = None,  # in seconds
        oversampling_factor: int = 2,
        v141_compat: bool = False,
        adc_dead_time_correction: bool = True,
        fit_epi: bool = True,
        save_name=None,
        labeled: bool = True,
        uniform_spoiler_areas: bool = False,
        uniform_spoiler_directions: bool = False,
        phase_cycling: bool = False,
    ):
        """Construct and immediately build a triple spin-echo diffusion EPI sequence.

        All parameters are resolved, timing is validated, gradient events are created,
        and :meth:`build_seq` is called so the object is ready for export on return.

        Args:
            name: Sequence identifier (no underscores — reserved as filename delimiters).
            fov: Field of view in metres.
            Nx: Readout matrix size.
            Ny: Phase-encoding matrix size.
            slice_thickness: Slice thickness in metres.
            TR: Repetition time in **milliseconds**.
            TE: First echo time in **milliseconds** (RF90 centre → EPI1 k-space centre).
            b_value: Target diffusion weighting in s/mm².
            b_directions: Number of diffusion directions (must match a supported scheme in
                :func:`get_diffusion_directions`).
            b_0_frequency: Insert a b=0 volume every N directions (0 = no b0 interleaving).
            small_delta: Duration of each diffusion gradient lobe in seconds.
                ``None`` = auto-compute from available TE/2 window.
            big_DELTA: Separation between the onset of the two diffusion gradient lobes in
                seconds.  ``None`` = auto-compute.  Must be provided together with
                ``small_delta`` if either is given.
            N_slices: Number of slices.
            system_type: Hardware limits preset (:class:`SystemLimitType`).
            rf90_duration: Duration of the 90° sinc excitation pulse in seconds.
            rf180_duration: Duration of each 180° refocusing pulse in seconds.
                0 = reuse ``rf90_duration``.
            resolution: Isotropic in-plane resolution in mm (overrides Nx/Ny if given).
            flip_angle: Excitation flip angle in degrees.
            apodization: Sinc window apodization factor for RF pulses.
            time_bw_product: Time-bandwidth product for RF pulses.
            dwell_time: ADC dwell time in seconds (``None`` = auto).
            end_spoilers: Append gradient spoilers at end of each TR.
            rf180_spoiler: Play spoiler gradients immediately before and after each RF180.
            spoiler_amplitude: Peak spoiler amplitude as a fraction of ``system.max_grad``.
            spoiler_duration: Spoiler gradient duration in seconds.
            prephaser_duration: Fixed EPI prephaser duration in seconds (``None`` = auto).
            rephasers: Append EPI rephaser gradients after each readout train.
            simultan_rephasers: Play x and y rephasers simultaneously (``True``) or
                sequentially (``False``).
            blip_down: Global blip polarity — ``True`` = first step is in −ky direction
                (standard); ``False`` inverts all three readouts.
            alternating_blip_polarity: If ``True``, the three readouts use polarities
                [down, up, down] (or their inversion when ``blip_down=False``) to enable
                B0-distortion correction between odd and even echoes.
            ramp_sampling: ADC strategy (``'none'`` or ``'ramp_sampled'``).
            save_dir: Output directory for ``.seq`` files.
            logger: External logger; a default one is created if ``None``.
            partial_fourier_factor: Fraction of ky-space to acquire for EPI1 (0.5–1.0).
                Reduced automatically if the readout does not fit in TE/2.
            calibration_readout: Prepend a 3-line navigator spin echo before the DWI loop.
            acceleration_factor: In-plane EPI acceleration factor R.
            max_EPI_duration: Hard cap on the EPI readout duration in seconds.
            oversampling_factor: Unused reserved field (kept for API compatibility).
            v141_compat: Encode Pulseq v1.4.1-compatible gradient shapes.
            adc_dead_time_correction: Apply ADC dead-time delay in timing calculations.
            fit_epi: Automatically reduce ``partial_fourier_factor`` until the EPI readout
                fits within TE.  ``False`` raises an error immediately on failure.
            save_name: Override the auto-generated filename (full name including extension).
            labeled: Attach ICE reconstruction labels (LIN, REV, NAV, ECO, REP) to all
                ADC blocks.  Set ``False`` for simulation (MR-zero / Bloch).
            uniform_spoiler_areas: Use the maximum spoiler area for all three spoilers
                instead of linearly spacing them.
            uniform_spoiler_directions: Apply all three spoilers on the same axis (z)
                instead of distributing across x, y, z.
            phase_cycling: Apply YXY RF-phase cycling to the three 180° pulses
                (phases: Y=π/2, X=0, Y=π/2).
        """
        self.save_name = save_name

        # --- EPI parameters (from EPIDiffusionDoubleSEPulseqSeq._init_epi_params) ---
        self.partial_fourier_factor = partial_fourier_factor
        self.calibration_readout = calibration_readout
        self.epi_acceleration_factor = acceleration_factor
        self.oversampling_factor = oversampling_factor
        self.prephaser_duration = prephaser_duration
        self.rephasers = rephasers
        self.simultan_rephasers = simultan_rephasers

        # Base polarity pattern: alternate [down, up, down] for B0 correction, or all-down otherwise
        self.blip_down = (
            [True, False, True] if alternating_blip_polarity else [True, True, True]
        )
        # XOR with (not blip_down): when blip_down=False the entire pattern is inverted
        self.blip_down = np.logical_xor(self.blip_down, not blip_down).tolist()

        self.ramp_sampling = ramp_sampling
        self.max_EPI_duration = max_EPI_duration
        self.fit_epi = fit_epi
        self.adc_dead_time_correction = adc_dead_time_correction
        self.labeled = labeled
        self.uniform_spoiler_areas = uniform_spoiler_areas
        self.uniform_spoiler_directions = uniform_spoiler_directions
        self.phase_cycling = phase_cycling

        # --- PulseqSeq base init ---
        self._init_logging(
            logger or logging.getLogger("EPIDiffusionTripleSEPulseqSeq"),
            name,
            system_type,
            save_dir,
        )
        self._init_system(system_type)
        self._init_imaging_params(
            fov,
            Nx,
            Ny,
            slice_thickness,
            TR,
            N_slices,
            resolution,
            flip_angle,
            apodization,
            time_bw_product,
            v141_compat,
        )
        self._init_readout_timing(dwell_time)
        self._init_rf90(rf90_duration)
        self._init_spoilers(end_spoilers, spoiler_amplitude, spoiler_duration)

        # --- Diffusion parameters (from DiffusionSEPulseqSeq._init_diffusion_params) ---
        self._init_diffusion_params(
            b_value, b_directions, b_0_frequency, small_delta, big_DELTA, rf180_spoiler
        )

        # --- Spin-echo RF180 (from SEPulseqSeq._init_se) ---
        self._init_se(TE, rf180_duration)

        # --- EPI fit loop + timing ---
        self._run_epi_fit_loop()
        self._calc_epi_tr_delay()
        self.diffusion_gradient_amplitudes = [
            self.diffusion_gradient_amplitude
        ] * self.b_directions
        self.build_seq()

    # =========================================================================
    # Init helpers (flattened from intermediate classes)
    # =========================================================================
    def _init_spoilers(self, end_spoilers, spoiler_amplitude, spoiler_duration):
        """Build three spoiler gradient sets with linearly spaced dephasing areas.

        Overrides :meth:`PulseqSeq._init_spoilers` to create three distinct spoiler
        strengths (spoiler1 < spoiler2 < spoiler3) placed around each of the three
        180° pulses.  The area range spans from the minimum required for 4π dephasing
        (2 voxels) up to the maximum achievable at ``spoiler_amplitude × max_grad``.

        The spoiler axis and area uniformity are controlled by ``uniform_spoiler_areas``
        and ``uniform_spoiler_directions`` (set before this method is called).

        Args:
            end_spoilers: If ``True``, also append spoilers at the very end of each TR.
            spoiler_amplitude: Peak spoiler amplitude as a fraction of ``system.max_grad``.
            spoiler_duration: Spoiler duration in seconds (ceiled to gradient raster).
        """
        self.end_spoilers = end_spoilers

        # ── 1. Minimum area for 4π dephasing (2 voxels) ──────────────────────────────
        # 2π → 1/res, so 4π → 2/res
        min_area = 2 / (self.resolution * 1e-3)  # in m
        min_spoiler_dummy = pp.make_trapezoid(
            channel="z",
            area=min_area,
            duration=spoiler_duration,
            system=self.system,
        )

        self.logger.info(
            f"Min spoiler:"
            f"\n\tAmplitude (4π)={min_spoiler_dummy.amplitude/self.system.gamma:.2f} mT/m,"
            f"\n\tRise time={min_spoiler_dummy.rise_time*1e6:.1f} µs,"
            f"\n\tDuration={pp.calc_duration(min_spoiler_dummy)*1e6:.1f} µs,"
            f"\n\tTriangular area={min_spoiler_dummy.area:.1f} Hz/m·s"
        )

        # ── 2. Trapezoid at max_grad in minimum time ─────────────────────────────
        amplitude = (
            spoiler_amplitude * self.system.max_grad
        )  # of max_grad to allow for some tolerance while still achieving strong spoiling
        rise_time = (
            amplitude / self.system.max_slew
        )  # minimum rise time for this amplitude
        max_spoiler_dummy = pp.make_trapezoid(
            channel="z",
            amplitude=amplitude,
            duration=spoiler_duration,
            system=self.system,
        )
        self.logger.info(
            f"Max spoiler:"
            f"\n\tAmplitude ({spoiler_amplitude} max)={max_spoiler_dummy.amplitude/self.system.gamma*1e3:.2f} mT/m,"
            f"\n\tRise time={max_spoiler_dummy.rise_time*1e6:.1f} µs,"
            f"\n\tDuration={pp.calc_duration(max_spoiler_dummy)*1e6:.1f} µs,"
            f"\n\tTriangular area={max_spoiler_dummy.area:.1f} Hz/m·s"
        )

        self.spoiler_areas = np.linspace(
            min_spoiler_dummy.area, max_spoiler_dummy.area, num=3
        )

        duration = align2rastertime_ceil(spoiler_duration, self.system.grad_raster_time)

        self.spoiler_z = pp.make_trapezoid(
            channel="z",
            area=self.spoiler_areas[0],
            duration=duration,
            system=self.system,
        )
        self.spoiler_y = pp.make_trapezoid(
            channel="y",
            area=self.spoiler_areas[0],
            duration=duration,
            system=self.system,
        )
        self.spoiler_x = pp.make_trapezoid(
            channel="x",
            area=self.spoiler_areas[0],
            duration=duration,
            system=self.system,
        )

        self.spoiler_z2 = pp.make_trapezoid(
            channel="z",
            area=self.spoiler_areas[1],
            duration=duration,
            system=self.system,
        )
        self.spoiler_y2 = pp.make_trapezoid(
            channel="y",
            area=self.spoiler_areas[1],
            duration=duration,
            system=self.system,
        )
        self.spoiler_x2 = pp.make_trapezoid(
            channel="x",
            area=self.spoiler_areas[1],
            duration=duration,
            system=self.system,
        )

        self.spoiler_z3 = pp.make_trapezoid(
            channel="z",
            area=self.spoiler_areas[2],
            duration=duration,
            system=self.system,
        )
        self.spoiler_y3 = pp.make_trapezoid(
            channel="y",
            area=self.spoiler_areas[2],
            duration=duration,
            system=self.system,
        )
        self.spoiler_x3 = pp.make_trapezoid(
            channel="x",
            area=self.spoiler_areas[2],
            duration=duration,
            system=self.system,
        )

        if self.uniform_spoiler_areas:
            if self.uniform_spoiler_directions:
                self.spoiler1 = self.spoiler_z3
                self.spoiler2 = self.spoiler_z3
                self.spoiler3 = self.spoiler_z3
            else:
                self.spoiler1 = self.spoiler_z3
                self.spoiler2 = self.spoiler_y3
                self.spoiler3 = self.spoiler_x3

        else:
            if self.uniform_spoiler_directions:
                self.spoiler1 = self.spoiler_z
                self.spoiler2 = self.spoiler_z2
                self.spoiler3 = self.spoiler_z3
            else:
                self.spoiler1 = self.spoiler_z
                self.spoiler2 = self.spoiler_y2
                self.spoiler3 = self.spoiler_x3

    def _init_diffusion_params(
        self,
        b_value,
        b_directions,
        b_0_frequency,
        small_delta,
        big_DELTA,
        rf180_spoiler,
    ):
        """Store diffusion-weighting parameters and retrieve the gradient direction table.

        ``small_delta`` and ``big_DELTA`` are stored as ``None`` when auto-computation is
        requested; ``_try_epi_fit`` will populate them during the fit loop.  Both must
        either be ``None`` together or specified together — a partial specification raises
        :class:`AssertionError`.

        Args:
            b_value: Target b-value in s/mm².
            b_directions: Number of diffusion-encoding directions.
            b_0_frequency: Insert a b=0 volume every N directions (0 = no interleaving).
            small_delta: Duration of each diffusion gradient lobe in seconds, or ``None``.
            big_DELTA: Onset-to-onset separation of the two gradient lobes in seconds,
                or ``None``.
            rf180_spoiler: Whether spoiler gradients will be played around each RF180.
        """
        self.rf180_spoiler = rf180_spoiler
        self.b_value: int = b_value
        self.b_0_frequency: int = b_0_frequency
        self.b_dirs = b_directions
        self.b_directions: np.ndarray = get_diffusion_directions(
            b_directions, b_0_frequency
        )
        self.small_delta: float = small_delta
        self.big_DELTA: float = big_DELTA
        self._user_small_delta = small_delta
        self._user_big_DELTA = big_DELTA

        if self.big_DELTA or self.small_delta:
            assert (
                self.big_DELTA and self.small_delta
            ), "Both big_DELTA and small_delta must be set together."

    def _init_se(self, TE, rf180_duration):
        """Create the 180° refocusing pulse and placeholder SE readout objects.

        The placeholder ``gx``, ``adc``, and ``gx_pre`` are sized for a conventional
        (non-EPI) spin-echo readout.  They are overwritten by :meth:`_try_epi_fit`
        with the actual EPI gradient events once the fit loop succeeds.  They exist
        here only to allow the object to be constructed without error if ``_try_epi_fit``
        needs to inspect them during the fit.

        Also creates ``rf180_1``, ``rf180_2``, ``rf180_3`` — either three copies of the
        same pulse (``phase_cycling=False``) or three YXY phase-cycled variants.

        Args:
            TE: First echo time in **milliseconds** (stored as seconds).
            rf180_duration: Duration of the 180° pulse in seconds; 0 = reuse ``rf_duration``.
        """
        self.TE = TE * 1e-3
        self.rf180_duration = rf180_duration if rf180_duration > 0 else self.rf_duration

        self.rf180, self.gz180, _ = pp.make_sinc_pulse(
            flip_angle=deg2rad(180),
            duration=self.rf180_duration,
            delay=self.system.rf_dead_time if self.system.rf_dead_time else 0,
            slice_thickness=self.slice_thickness,
            apodization=self.apodization,
            time_bw_product=self.time_bw_product,
            system=self.system,
            use="refocusing",
            return_gz=True,
        )
        if self.phase_cycling:
            self.rf180_1, self.rf180_2, self.rf180_3 = self._make_rf180_phases()
        else:
            self.rf180_1 = self.rf180_2 = self.rf180_3 = self.rf180

        # Phase encoding gradient amplitudes (used by base DiffusionSE, kept for compatibility)
        self.phase_encoding_gradients = (
            np.arange(self.Ny) - self.Ny / 2
        ) * self.delta_k

        # SE readout objects — these get overwritten by _try_epi_fit but are needed
        # for the intermediate _init_se path to complete without error.
        self.flat_time_raster = align2rastertime_ceil(
            self.readout_time, 2 * self.system.grad_raster_time
        )
        self.gx_raster_difference = self.flat_time_raster - self.readout_time
        self.gx_amplitude = self.k_width / self.flat_time_raster
        self.gx = pp.make_trapezoid(
            channel="x",
            system=self.system,
            amplitude=self.gx_amplitude,
            flat_time=self.flat_time_raster,
        )
        adc_delay = self.gx.rise_time + self.gx_raster_difference / 2
        self.adc = pp.make_adc(
            system=self.system,
            num_samples=self.Nx,
            dwell=self.dwell_time,
            delay=adc_delay,
        )
        adc_prephase_area = self.gx.amplitude * self.flat_time_raster / 2
        ramp_area = self.gx.amplitude * self.gx.rise_time / 2
        self.gx_pre = pp.make_trapezoid(
            channel="x",
            system=self.system,
            area=-(ramp_area + adc_prephase_area),
        )
        self.gy_pre_dummy = pp.make_trapezoid(
            channel="y",
            system=self.system,
            area=self.phase_encoding_gradients[0],
        )

    def _make_rf180_phases(self) -> tuple:
        """Generate three phase-cycled 180° refocusing pulses using a YXY scheme.

        YXY phase cycling (phases π/2, 0, π/2) suppresses stimulated-echo artefacts
        and reduces sensitivity to B1⁺ inhomogeneity by cycling the refocusing axis.
        The slice-selection gradient ``gz180`` is phase-independent and shared across
        all three pulses.

        Returns:
            Tuple ``(rf180_Y, rf180_X, rf180_Y)`` — one pulse object per echo.
        """
        phases = [np.pi / 2, 0.0, np.pi / 2]  # Y, X, Y
        rfs = []
        for phi in phases:
            rf, gz, _ = pp.make_sinc_pulse(
                flip_angle=deg2rad(180),
                duration=self.rf180_duration,
                delay=self.system.rf_dead_time if self.system.rf_dead_time else 0,
                slice_thickness=self.slice_thickness,
                apodization=self.apodization,
                time_bw_product=self.time_bw_product,
                phase_offset=phi,  # Parameter for phase cycling
                system=self.system,
                use="refocusing",
                return_gz=True,
            )
            rfs.append(rf)
        return rfs[0], rfs[1], rfs[2]  # rf180_Y, rf180_X, rf180_Y

    # =========================================================================
    # EPI fit loop (from EPIDiffusionDoubleSEPulseqSeq)
    # =========================================================================

    def _run_epi_fit_loop(self):
        """Iteratively reduce partial Fourier factor until EPI readout fits within TE.

        Calls :meth:`_try_epi_fit` in a loop.  On each failure the partial-Fourier
        factor is decremented by 0.05 (5 percentage points).  If ``fit_epi=False``
        a failure on the first attempt raises :class:`ValueError` immediately instead
        of retrying.  Raises :class:`ValueError` if the factor drops below 0.5.

        On success all timing attributes and :class:`EPIReadout` objects are stored
        as instance attributes by :meth:`_try_epi_fit`.
        """
        epi_fit = False
        fit_step = 0.05
        while not epi_fit:
            self.small_delta = self._user_small_delta
            self.big_DELTA = self._user_big_DELTA
            epi_fit, _ = self._try_epi_fit()

            if not self.fit_epi and not epi_fit:
                raise ValueError(
                    f"EPI readout does not fit in TE={self.TE*1e3:.2f} ms with current parameters. "
                    f"Set fit_epi=True to automatically reduce partial Fourier factor and retry.\n"
                    f"{_}"
                )
            if not epi_fit:
                self.logger.warning(
                    f"EPI readout does not fit in TE={self.TE*1e3:.2f} ms with pff={self.partial_fourier_factor:.2f}, "
                    f"reducing pff to {self.partial_fourier_factor - fit_step:.2f} and retrying..."
                )
                self.partial_fourier_factor = np.round(
                    self.partial_fourier_factor - fit_step, 2
                )
                if self.partial_fourier_factor < 0.5:
                    raise ValueError(
                        "Partial Fourier factor reduced below 0.5, cannot fit EPI readout in TE."
                    )

    def _try_epi_fit(self):
        """Attempt to fit the EPI readout and diffusion gradients within TE constraints.

        Builds a trial :class:`EPIReadout` with the current partial-Fourier factor, then
        solves for the sequence of delays that satisfies the spin-echo condition:

        .. code-block:: text

            TE/2  =  [RF90 → RF180_1 centre]
                  =  [RF180_1 centre → EPI1 k-space centre]

        Steps:
        1. Compute RF centre offsets (``rf90_center``, ``rf180_center``).
        2. Derive raw ``delayTE1`` and ``delayTE2`` from the TE/2 constraint.
        3. Auto-compute or validate ``small_delta`` and ``big_DELTA``.
        4. Compute required diffusion gradient amplitude; clamp to ``max_grad`` if needed
           (preserving the user-specified ``small_delta`` and reporting the achieved b-value).
        5. Enforce the exact ``big_DELTA`` with an inner delay ``delayTE1_inner``.
        6. Apply a sub-raster TE tick correction to achieve exact TE.
        7. Store all EPI objects, gradient events, and delays as instance attributes.

        Returns:
            Tuple ``(success: bool, error_message: str | None)``.
        """
        epi = EPIReadout(
            system=self.system,
            Nx=self.Nx,
            Ny=self.Ny,
            fov=self.fov,
            dwell_time=self.dwell_time,
            partial_fourier_factor=self.partial_fourier_factor,
            blip_down=self.blip_down[0],
            prephaser_duration=self.prephaser_duration,
            acceleration_factor=self.epi_acceleration_factor,
            ramp_sampling=self.ramp_sampling,
            rephasers=self.rephasers,
            simultan_rephasers=self.simultan_rephasers,
            adc_dead_time_correction=self.adc_dead_time_correction,
        )

        gx_pre = epi.gx_prephaser
        gy_pre = epi.gy_prephaser
        self.prephaser_duration = epi.prephaser_duration
        self.time_until_echo = epi.time_until_echo
        self.epi_duration = epi.duration

        # --- TE timing: derive delays satisfying TE/2 = [RF90→RF180] = [RF180→EPI echo] ---
        rf90_center = pp.calc_rf_center(self.rf90)[0]
        rf180_center = pp.calc_rf_center(self.rf180)[0]

        # Time from RF90 k-space centre to end of the RF90+gz90 block
        rf90_center_with_delay = rf90_center + self.rf90.delay
        time_after_90 = pp.calc_duration(self.rf90, self.gz90) - rf90_center_with_delay

        # Time from start of RF180 block to RF180 k-space centre
        rf180_center_with_delay = rf180_center + self.rf180.delay
        # Time from RF180 k-space centre to end of the RF180+gz180 block
        time_after_180 = (
            pp.calc_duration(self.rf180, self.gz180) - rf180_center_with_delay
        )

        # delayTE1: total free time between gz90_reph and start of RF180 block
        # = TE/2 − (time_after_90 + gz90_reph + rf180_center_with_delay + optional spoiler)
        delayTE1_raw = self.TE / 2 - (
            time_after_90 + pp.calc_duration(self.gz90_reph) + rf180_center_with_delay
        )
        if self.rf180_spoiler:
            delayTE1_raw -= pp.calc_duration(self.spoiler_z)

        # delayTE2: free time after RF180 block before EPI k-space centre
        # = TE/2 − (time_after_180 + prephaser + time_to_echo + optional spoiler)
        delayTE2_raw = self.TE / 2 - (
            time_after_180 + self.prephaser_duration + self.time_until_echo
        )
        if self.rf180_spoiler:
            delayTE2_raw -= pp.calc_duration(self.spoiler_z)

        if delayTE2_raw < 0:
            self.logger.warning(
                f"TE too short for EPI readout! delayTE2_raw={delayTE2_raw*1e3:.3f} ms"
            )
            return (False, f"DelayTE2 ({delayTE2_raw*1e3:.3f} ms) is negative")

        # Raster-align delayTE1; compensate delayTE2 for the rounding residual so the
        # sum delayTE1 + delayTE2 stays on-raster without changing the total TE
        delayTE1 = align2rastertime_nearest(delayTE1_raw, self.system.grad_raster_time)
        delayTE1_error = delayTE1 - delayTE1_raw

        delayTE2_compensated = delayTE2_raw - delayTE1_error
        delayTE2 = align2rastertime_nearest(
            delayTE2_compensated, self.system.grad_raster_time
        )

        if delayTE1 < 0:
            self.logger.warning(f"delayTE1 ({delayTE1*1e3:.3f} ms) is negative!")
            return (False, f"DelayTE1 ({delayTE1*1e3:.3f} ms) is negative!")

        # --- Diffusion gradient sizing ---
        # Total time occupied by the RF180 block (pulse + optional surrounding spoilers)
        time_rf180_block = pp.calc_duration(self.rf180, self.gz180)
        if self.rf180_spoiler:
            time_rf180_block += 2 * pp.calc_duration(self.spoiler_z)

        if self.small_delta is None or self.big_DELTA is None:
            # Auto mode: fill as much of the available TE/2 window as possible
            available_window = min(delayTE1, delayTE2)
            max_ramp_time = align2rastertime_ceil(
                self.system.max_grad / self.system.max_slew,
                self.system.grad_raster_time,
            )
            # Leave room for the gradient ramps on both sides of the flat top
            small_delta = available_window - 2 * max_ramp_time
            # big_DELTA: onset-to-onset separation = first gradient window + RF180 block
            big_DELTA = delayTE1 + time_rf180_block
        else:
            small_delta = self.small_delta
            big_DELTA = self.big_DELTA

        if small_delta <= 0:
            self.logger.warning(
                f"small_delta ({small_delta*1e3:.3f} ms) is non-positive!"
            )
            return (False, f"small_delta ({small_delta*1e3:.3f} ms) is non-positive!")

        diffusion_gradient_amplitude = calc_diffusion_gradient_amplitude(
            self.b_value, small_delta, big_DELTA
        )

        self.diffusion_gradient_amplitude = diffusion_gradient_amplitude
        self.small_delta = small_delta

        # Amplitude clamp
        self._amplitude_clamped = False
        if diffusion_gradient_amplitude > self.system.max_grad:
            if self._user_small_delta is None:
                self.logger.warning(
                    f"Required diffusion amplitude ({diffusion_gradient_amplitude:.0f} Hz/m) "
                    f"exceeds max_grad ({self.system.max_grad:.0f} Hz/m) for auto small_delta="
                    f"{small_delta*1e3:.2f} ms. Returning False for fit loop to retry."
                )
                return (
                    False,
                    f"Amplitude {diffusion_gradient_amplitude:.0f} Hz/m exceeds max_grad for auto small_delta",
                )

            trap_params = calc_area_preserving_trapezoid(
                diffusion_gradient_amplitude,
                small_delta,
                self.system.max_grad,
                self.system.max_slew,
                self.system.grad_raster_time,
            )
            if trap_params is None:
                msg = (
                    f"max_grad ramps exceed user-specified small_delta={small_delta*1e3:.2f} ms "
                    f"-- cannot fit gradient even as a pure triangle at max_grad"
                )
                self.logger.error(msg)
                return (False, msg)

            actual_b = (
                calc_bval(
                    self.system.max_grad / self.system.gamma,
                    trap_params["delta_eff"],
                    big_DELTA,
                    trap_params["rise_time"],
                )
                * 1e-6
            )
            self.logger.warning(
                f"Diffusion amplitude clamped: {diffusion_gradient_amplitude:.0f} -> "
                f"{self.system.max_grad:.0f} Hz/m (user small_delta={small_delta*1e3:.2f} ms preserved). "
                f"Actual b-value ~ {actual_b:.1f} s/mm^2 (requested {self.b_value} s/mm^2)."
            )

            diffusion_gradient_amplitude = self.system.max_grad
            self._amplitude_clamped = True
            try:
                g_diffusion_dummy = pp.make_trapezoid(
                    "z",
                    system=self.system,
                    amplitude=trap_params["amplitude"],
                    rise_time=trap_params["rise_time"],
                    flat_time=trap_params["flat_time"],
                )
            except ValueError as e:
                self.logger.error(
                    f"Could not create area-preserving diffusion gradient: {e}"
                )
                return (
                    False,
                    f"Could not create area-preserving diffusion gradient: {e}",
                )

            self.diffusion_gradient_flat_time = g_diffusion_dummy.flat_time
            self.diffusion_gradient_rise_time = g_diffusion_dummy.rise_time
        else:
            max_rise_time = align2rastertime_ceil(
                diffusion_gradient_amplitude / self.system.max_slew,
                self.system.grad_raster_time,
            )
            min_flat_time = small_delta - 2 * max_rise_time

            if min_flat_time < 0:
                self.logger.error(
                    f"minimum flat time ({min_flat_time*1e3:.3f} ms) is negative!"
                )
                return (False, f"small_delta too short for worst-case ramp times")

            try:
                g_diffusion_dummy = pp.make_trapezoid(
                    "z",
                    system=self.system,
                    amplitude=diffusion_gradient_amplitude,
                    rise_time=max_rise_time,
                    flat_time=min_flat_time,
                )
                self.diffusion_gradient_flat_time = g_diffusion_dummy.flat_time
                self.diffusion_gradient_rise_time = g_diffusion_dummy.rise_time
            except ValueError as e:
                self.logger.error(
                    f"Could not create diffusion gradient with amplitude {diffusion_gradient_amplitude} Hz/m: {e}"
                )
                return (False, f"Could not create diffusion gradient: {e}")

        diffusion_gradient_duration = pp.calc_duration(g_diffusion_dummy)

        # --- Enforce exact big_DELTA via inner delay between Gdiff1 and RF180_1 ---
        # delayTE1_inner fills the gap: big_DELTA = Gdiff1_duration + delayTE1_inner + RF180_block
        delayTE1_inner = big_DELTA - diffusion_gradient_duration - time_rf180_block
        delayTE1_inner = align2rastertime_nearest(
            delayTE1_inner, self.system.grad_raster_time
        )

        if delayTE1_inner < 0:
            self.logger.error(
                f"delayTE1_inner ({delayTE1_inner*1e3:.3f} ms) is negative! big_DELTA may be too small."
            )
            return (False, f"delayTE1_inner ({delayTE1_inner*1e3:.3f} ms) is negative")

        delay_before_diff1 = delayTE1 - diffusion_gradient_duration - delayTE1_inner
        delay_before_diff1 = align2rastertime_nearest(
            delay_before_diff1, self.system.grad_raster_time
        )

        if delay_before_diff1 < 0:
            self.logger.error(
                f"delay_before_diff1 ({delay_before_diff1*1e3:.3f} ms) is negative! big_DELTA may be too large for TE."
            )
            return (
                False,
                f"delay_before_diff1 ({delay_before_diff1*1e3:.3f} ms) is negative",
            )

        delayTE2_adjusted = delayTE2 - diffusion_gradient_duration
        delayTE2_adjusted = align2rastertime_nearest(
            delayTE2_adjusted, self.system.grad_raster_time
        )

        if delayTE2_adjusted < 0:
            self.logger.error(
                f"delayTE2_adjusted ({delayTE2_adjusted*1e3:.3f} ms) is negative!"
            )
            return (
                False,
                f"delayTE2_adjusted ({delayTE2_adjusted*1e3:.3f} ms) is negative",
            )

        # --- Sub-raster TE correction: absorb any remaining timing error into delayTE2 ---
        # Accumulate the actual TE1 from all blocks, then compute how many raster ticks
        # separate it from the target TE.  The correction is applied to delayTE2_adjusted.
        actual_TE1 = (
            time_after_90
            + pp.calc_duration(self.gz90_reph)
            + delay_before_diff1
            + diffusion_gradient_duration
            + delayTE1_inner
            + (pp.calc_duration(self.spoiler_z) if self.rf180_spoiler else 0)
            + rf180_center_with_delay
        )
        actual_TE2 = (
            time_after_180
            + (pp.calc_duration(self.spoiler_z) if self.rf180_spoiler else 0)
            + diffusion_gradient_duration
            + delayTE2_adjusted
            + self.prephaser_duration
            + self.time_until_echo
        )
        te_correction_ticks = int(
            np.floor(
                (self.TE - (actual_TE1 + actual_TE2)) / self.system.grad_raster_time
                + 0.5
            )
        )
        if te_correction_ticks != 0:
            self.logger.info(
                f"TE tick correction: {te_correction_ticks} ticks ({te_correction_ticks * self.system.grad_raster_time * 1e6:.1f} us)"
            )
        delayTE2_adjusted += te_correction_ticks * self.system.grad_raster_time

        if delayTE2_adjusted < 0:
            self.logger.error(
                f"delayTE2_adjusted after TE correction ({delayTE2_adjusted*1e3:.3f} ms) is negative!"
            )
            return (
                False,
                f"delayTE2_adjusted after TE correction ({delayTE2_adjusted*1e3:.3f} ms) is negative",
            )

        # Verify actual big_DELTA
        actual_big_DELTA = (
            diffusion_gradient_duration + delayTE1_inner + time_rf180_block
        )
        self.logger.info(
            f"Requested big_DELTA: {big_DELTA*1e3:.2f} ms, Actual big_DELTA: {actual_big_DELTA*1e3:.2f} ms"
        )
        self.logger.info(
            f"delay_before_diff1: {delay_before_diff1*1e3:.2f} ms, delayTE1_inner: {delayTE1_inner*1e3:.2f} ms"
        )
        self.logger.info(f"delayTE2_adjusted: {delayTE2_adjusted*1e3:.2f} ms")
        self.logger.info(f"Calculated small_delta: {small_delta*1e3:.2f} ms")
        self.logger.info(f"Calculated big_DELTA: {big_DELTA*1e3:.2f} ms")
        self.logger.info(
            f"Diffusion amplitude: {diffusion_gradient_amplitude:.2f} Hz/m ({diffusion_gradient_amplitude / self.system.gamma * 1e3:.4f} mT/m)"
        )
        self.logger.info(f"Diffusion amplitude for b={self.b_value} s/mm^2")

        # Store all calculated values
        self.epi = epi
        self.gx_pre = gx_pre
        self.gy_pre = gy_pre
        self.gx = epi.gx
        self.gy = epi.gy
        self.gx_pre = epi.gx_prephaser
        self.gy_pre = epi.gy_prephaser
        self.adc = epi.adc
        self.rf90_center = rf90_center
        self.rf180_center = rf180_center
        self.rf90_center_with_delay = rf90_center_with_delay
        self.time_after_90 = time_after_90
        self.rf180_center_with_delay = rf180_center_with_delay
        self.time_after_180 = time_after_180

        # Diffusion timing
        self.small_delta = small_delta
        self.big_DELTA = big_DELTA
        self.diffusion_gradient_amplitude = diffusion_gradient_amplitude
        self.g_diffusion_dummy = g_diffusion_dummy
        self.diffusion_gradient_duration = diffusion_gradient_duration
        self.diffusion_gradient_total_duration = diffusion_gradient_duration

        # Delay timing
        self.delay_before_diff1 = delay_before_diff1
        self.delayTE1_inner = delayTE1_inner
        self.delayTE1 = delayTE1
        self.delayTE2 = delayTE2_adjusted
        self.time_rf180_block = time_rf180_block

        epi._log_parameters()

        return (True, None)

    # =========================================================================
    # Second echo (from EPIDiffusionDoubleSEPulseqSeq._create_epi2)
    # =========================================================================

    def _create_epi2(self):
        """Create the second EPI readout (full k-space) and solve its timing.

        The second spin echo is refocused by RF180_2, which must sit exactly halfway
        between the TE1 echo centre and the TE2 echo centre.  The half-time
        ``T_half`` is the larger of the pre-RF180_2 path and the post-RF180_2 path,
        so that both delays are non-negative.

        After computing the raw TE2, the value is rounded **up** to the nearest
        millisecond (``target_TE2 = ceil(raw_TE2 / 1 ms)``); the extra raster ticks
        are split evenly between ``delay_before_rf180_2`` (before) and
        ``delay_before_epi2`` (after) to preserve the spin-echo symmetry condition.

        Sets attributes: ``epi2``, ``delay_before_rf180_2``, ``delay_before_epi2``,
        ``T_half``, ``TE2``, ``epi2_total_time``.
        """
        self.epi2 = EPIReadout(
            system=self.system,
            Nx=self.Nx,
            Ny=self.Ny,
            fov=self.fov,
            dwell_time=self.dwell_time,
            partial_fourier_factor=1.0,
            blip_down=self.blip_down[1],
            prephaser_duration=self.prephaser_duration,
            acceleration_factor=self.epi_acceleration_factor,
            ramp_sampling=self.ramp_sampling,
            rephasers=self.rephasers,
            simultan_rephasers=self.simultan_rephasers,
            adc_dead_time_correction=self.adc_dead_time_correction,
        )

        spoiler_dur = pp.calc_duration(self.spoiler_z)

        # Time from EPI1 echo centre to end of EPI1 readout
        time_after_epi1_echo = self.epi_duration - self.time_until_echo

        # Minimum half-interval on each side of RF180_2
        before_rf180_2 = (
            time_after_epi1_echo + spoiler_dur + self.rf180_center_with_delay
        )
        after_rf180_2 = (
            self.time_after_180
            + spoiler_dur
            + self.epi2.prephaser_duration
            + self.epi2.time_until_echo
        )

        # T_half_min: the larger of the two sides, raster-aligned — guarantees both delays ≥ 0
        T_half_min = align2rastertime_ceil(
            max(before_rf180_2, after_rf180_2),
            self.system.grad_raster_time,
        )

        raw_delay_before_rf180_2 = align2rastertime_nearest(
            T_half_min - before_rf180_2, self.system.grad_raster_time
        )
        raw_delay_before_epi2 = align2rastertime_nearest(
            T_half_min - after_rf180_2, self.system.grad_raster_time
        )
        raw_TE2 = (
            self.TE
            + time_after_epi1_echo
            + raw_delay_before_rf180_2
            + spoiler_dur
            + self.rf180_center_with_delay
            + self.time_after_180
            + spoiler_dur
            + self.epi2.prephaser_duration
            + raw_delay_before_epi2
            + self.epi2.time_until_echo
        )

        # Round TE2 up to the nearest ms for clean protocol display; split the extra
        # ticks symmetrically to preserve spin-echo symmetry.
        # Use integer tick arithmetic to avoid np.ceil misfire on values like
        # 142.00000000000003 ms (FP representation of exact 142 ms) → would
        # otherwise jump target_TE2 to 143 ms.
        raster = self.system.grad_raster_time
        ms_ticks = round(1e-3 / raster)  # ticks per ms; 100 for 10 µs raster
        raw_ticks = round(raw_TE2 / raster)  # integer ticks, eliminates FP noise
        
        extra_ticks = int(round(raw_ticks / ms_ticks)) * ms_ticks - raw_ticks
        # print(f"raw_TE2 = {raw_TE2*1e3:.4f} ms, extra_ticks = {extra_ticks}, "
        #                 f"target TE2 = {(raw_TE2 + extra_ticks*raster)*1e3:.4f} ms")

        # Odd tick counts: one more tick goes to the pre-RF180_2 side
        extra_before = (extra_ticks // 2) * raster
        extra_after = (extra_ticks - extra_ticks // 2) * raster

        self.delay_before_rf180_2 = raw_delay_before_rf180_2 + extra_before
        self.delay_before_epi2 = raw_delay_before_epi2 + extra_after

        T_half = T_half_min + extra_before
        self.T_half = T_half

        actual_TE2 = (
            self.TE
            + time_after_epi1_echo
            + self.delay_before_rf180_2
            + spoiler_dur
            + self.rf180_center_with_delay
            + self.time_after_180
            + spoiler_dur
            + self.epi2.prephaser_duration
            + self.delay_before_epi2
            + self.epi2.time_until_echo
        )
        self.TE2 = align2rastertime_nearest(actual_TE2, raster)

        self.epi2_total_time = (
            self.delay_before_rf180_2
            + spoiler_dur
            + pp.calc_duration(self.rf180, self.gz180)
            + spoiler_dur
            + self.epi2.prephaser_duration
            + self.delay_before_epi2
            + self.epi2.duration
        )

        self.logger.info(f"--- Second Echo (EPI2) Timing ---")
        self.logger.info(
            f"T_half: {T_half*1e3:.2f} ms (limiting constraint: {'before' if before_rf180_2 >= after_rf180_2 else 'after'} RF180_2)"
        )
        self.logger.info(f"TE1: {self.TE*1e3:.2f} ms, TE2: {self.TE2*1e3:.2f} ms")
        self.logger.info(
            f"delay_before_rf180_2: {self.delay_before_rf180_2*1e3:.2f} ms"
        )
        self.logger.info(f"delay_before_epi2: {self.delay_before_epi2*1e3:.2f} ms")
        self.logger.info(f"epi2_total_time: {self.epi2_total_time*1e3:.2f} ms")
        self.epi2._log_parameters()

    # =========================================================================
    # Third echo (from EPIDiffusionTripleSEPulseqSeq._create_epi3)
    # =========================================================================

    def _create_epi3(self):
        """Create the third EPI readout (full k-space) and solve its timing.

        Mirrors :meth:`_create_epi2` but references TE2 as the preceding echo and
        EPI2 as the preceding readout.  ``T_half_3`` and ``TE3`` are computed by the
        same minimum half-interval + ms-rounding procedure.

        Sets attributes: ``epi3``, ``delay_before_rf180_3``, ``delay_before_epi3``,
        ``T_half_3``, ``TE3``, ``epi3_total_time``.
        """
        self.epi3 = EPIReadout(
            system=self.system,
            Nx=self.Nx,
            Ny=self.Ny,
            fov=self.fov,
            dwell_time=self.dwell_time,
            partial_fourier_factor=1.0,
            blip_down=self.blip_down[2],
            prephaser_duration=self.prephaser_duration,
            acceleration_factor=self.epi_acceleration_factor,
            ramp_sampling=self.ramp_sampling,
            rephasers=self.rephasers,
            simultan_rephasers=self.simultan_rephasers,
            adc_dead_time_correction=self.adc_dead_time_correction,
        )

        spoiler_dur = pp.calc_duration(self.spoiler_z)

        time_after_epi2_echo = self.epi2.duration - self.epi2.time_until_echo

        before_rf180_3 = (
            time_after_epi2_echo + spoiler_dur + self.rf180_center_with_delay
        )
        after_rf180_3 = (
            self.time_after_180
            + spoiler_dur
            + self.epi3.prephaser_duration
            + self.epi3.time_until_echo
        )

        T_half_3_min = align2rastertime_ceil(
            max(before_rf180_3, after_rf180_3, self.T_half),
            self.system.grad_raster_time,
        )

        raw_delay_before_rf180_3 = align2rastertime_nearest(
            T_half_3_min - before_rf180_3, self.system.grad_raster_time
        )
        raw_delay_before_epi3 = align2rastertime_nearest(
            T_half_3_min - after_rf180_3, self.system.grad_raster_time
        )

        raw_TE3 = (
            self.TE2
            + time_after_epi2_echo
            + raw_delay_before_rf180_3
            + spoiler_dur
            + self.rf180_center_with_delay
            + self.time_after_180
            + spoiler_dur
            + self.epi3.prephaser_duration
            + raw_delay_before_epi3
            + self.epi3.time_until_echo
        )

        # Same integer-tick ceiling as _create_epi2(): avoids np.ceil misfire on
        # FP values like 177.00000000000003 (exact 177 ms) → TE3 += 1 ms ghost.
        raster = self.system.grad_raster_time
        ms_ticks = round(1e-3 / raster)  # ticks per ms; 100 for 10 µs raster
        raw_ticks = round(raw_TE3 / raster)  # integer ticks, eliminates FP noise

        extra_ticks = int(round(raw_ticks / ms_ticks)) * ms_ticks - raw_ticks
        # print(f"raw_TE3 = {raw_TE3*1e3:.4f} ms, extra_ticks = {extra_ticks}, "
        #                 f"target TE3 = {(raw_TE3 + extra_ticks*raster)*1e3:.4f} ms")

        extra_before = (extra_ticks // 2) * raster
        extra_after = (extra_ticks - extra_ticks // 2) * raster

        self.delay_before_rf180_3 = raw_delay_before_rf180_3 + extra_before
        self.delay_before_epi3 = raw_delay_before_epi3 + extra_after

        self.T_half_3 = T_half_3_min + extra_before

        actual_TE3 = (
            self.TE2
            + time_after_epi2_echo
            + self.delay_before_rf180_3
            + spoiler_dur
            + self.rf180_center_with_delay
            + self.time_after_180
            + spoiler_dur
            + self.epi3.prephaser_duration
            + self.delay_before_epi3
            + self.epi3.time_until_echo
        )
        self.TE3 = align2rastertime_nearest(actual_TE3, raster)

        self.epi3_total_time = (
            self.delay_before_rf180_3
            + spoiler_dur
            + pp.calc_duration(self.rf180, self.gz180)
            + spoiler_dur
            + self.epi3.prephaser_duration
            + self.delay_before_epi3
            + self.epi3.duration
        )

        self.logger.info(f"--- Third Echo (EPI3) Timing ---")
        self.logger.info(f"T_half_3_min: {T_half_3_min*1e3:.2f} ms")
        self.logger.info(
            f"T_half_3: {self.T_half_3*1e3:.2f} ms (limiting constraint: {'before' if before_rf180_3 >= after_rf180_3 else 'after'} RF180_3)"
        )
        self.logger.info(f"TE2: {self.TE2*1e3:.2f} ms, TE3: {self.TE3*1e3:.2f} ms")
        self.logger.info(
            f"delay_before_rf180_3: {self.delay_before_rf180_3*1e3:.2f} ms"
        )
        self.logger.info(f"delay_before_epi3: {self.delay_before_epi3*1e3:.2f} ms")
        self.logger.info(f"epi3_total_time: {self.epi3_total_time*1e3:.2f} ms")
        self.epi3._log_parameters()

    # =========================================================================
    # TR delay (triple SE override)
    # =========================================================================

    def _calc_epi_tr_delay(self):
        """Orchestrate echo 2 & 3 creation, then compute the TR padding delay.

        Calls :meth:`_create_epi2` and :meth:`_create_epi3` to finalise the second
        and third echo objects, then sums all blocks in the TR to derive ``delayTR``
        as the remainder of the repetition time.  Logs a warning if ``delayTR < 0``
        (TR is too short).
        """
        self._create_epi2()
        self._create_epi3()

        time_used = (
            pp.calc_duration(self.rf90, self.gz90)
            + pp.calc_duration(self.gz90_reph)
            + self.delay_before_diff1
            + self.diffusion_gradient_duration
            + self.delayTE1_inner
            + pp.calc_duration(self.rf180, self.gz180)
            + self.diffusion_gradient_duration
            + self.prephaser_duration
            + self.delayTE2
            + self.epi_duration
            + self.epi2_total_time
            + self.epi3_total_time
        )

        if self.rf180_spoiler:
            time_used += 2 * pp.calc_duration(self.spoiler_z)

        if self.end_spoilers:
            time_used += pp.calc_duration(
                self.spoiler_x, self.spoiler_y, self.spoiler_z
            )

        if self.ramp_sampling == "none":
            time_used -= self.epi.blip_duration / 2

        delayTR_exact = self.TR - time_used
        self.delayTR = align2rastertime_nearest(
            delayTR_exact, self.system.grad_raster_time
        )

        if self.delayTR < 0:
            self.logger.warning(f"TR too short! delayTR={self.delayTR*1e3:.2f} ms")

        self.logger.info(
            f"TE1: {self.TE*1e3:.2f} ms, TE2: {self.TE2*1e3:.2f} ms, TE3: {self.TE3*1e3:.2f} ms"
        )
        self.logger.info(
            f"delayTR_exact: {delayTR_exact*1e3:.2f} ms -> delayTR: {self.delayTR*1e3:.2f} ms"
        )
        self.logger.info(
            f"Time used in TR (without delayTR): {time_used*1e3:.2f} ms of {self.TR*1e3:.2f} ms"
        )

    # =========================================================================
    # Navigator (from EPIDiffusionSEPulseqSeqV2)
    # =========================================================================

    def _add_navigator_acquisition(self, seq: pp.Sequence):
        """Add a 3-line spin-echo navigator before the diffusion loop for EPI phase calibration.

        The navigator consists of a full RF90 → RF180 spin-echo with only three
        readout lines (forward, reversed, forward) centred on ky=0.  It is used
        offline to estimate the N/2-ghost phase correction coefficients for each
        slice and repetition.

        In labelled mode the three lines carry LIN=0/1/2 and REV=0/1/0 so that
        the ICE reconstruction can identify them as navigator data.  In unlabelled
        mode (simulation) the labels are omitted.

        Args:
            seq: pypulseq :class:`pp.Sequence` object to append blocks to.
        """
        delay_to_180 = (
            self.TE / 2
            - self.time_after_90
            - pp.calc_duration(self.gz90_reph)
            - self.rf180_center_with_delay
        )
        delay_to_180 = align2rastertime_nearest(
            delay_to_180, self.system.grad_raster_time
        )
        delay_to_echo = (
            self.TE / 2
            - self.time_after_180
            - self.prephaser_duration
            - pp.calc_duration(self.epi.gx, self.epi.adc)
            - self.gx.delay
            - self.epi.gx_.rise_time
            - (self.epi.gx_.flat_time / 2)
        )
        delay_to_echo = align2rastertime_nearest(
            delay_to_echo, self.system.grad_raster_time
        )
        self.delay_to_echo_nav = delay_to_echo

        delayTR = (
            self.TR
            - self.TE
            - ((self.epi.gx_.flat_time / 2) + self.epi.gx_.fall_time)
            - pp.calc_duration(self.epi.gx, self.epi.adc)
            - self.rf180_center_with_delay
        )
        delayTR = align2rastertime_ceil(delayTR, self.system.grad_raster_time)

        if self.labeled:
            # Excitation
            seq.add_block(self.rf90, self.gz90)
            seq.add_block(self.gz90_reph)
            seq.add_block(pp.make_delay(delay_to_180))
            # Refocusing
            seq.add_block(self.rf180, self.gz180)
            seq.add_block(pp.make_delay(delay_to_echo))
            seq.add_block(self.gx_pre)

            # Navigator EPI readout (3 lines, no ramp sampling)
            seq.add_block(
                pp.make_label(label="LIN", type="SET", value=0),
                pp.make_label(label="REV", type="SET", value=0),
                self.epi.gx,
                self.epi.adc,
            )

            seq.add_block(
                pp.make_label(label="LIN", type="SET", value=1),
                pp.make_label(label="REV", type="SET", value=1),
                self.epi.gx_,
                self.epi.adc,
            )

            seq.add_block(
                pp.make_label(label="LIN", type="SET", value=2),
                pp.make_label(label="REV", type="SET", value=0),
                self.epi.gx,
                self.epi.adc,
            )
            seq.add_block(pp.make_delay(delayTR))

        else:
            seq.add_block(self.rf90, self.gz90)
            seq.add_block(self.gz90_reph)
            seq.add_block(pp.make_delay(delay_to_180))
            seq.add_block(self.rf180, self.gz180)
            seq.add_block(pp.make_delay(delay_to_echo))
            seq.add_block(self.gx_pre)
            seq.add_block(self.epi.gx, self.epi.adc)
            seq.add_block(self.epi.gx_, self.epi.adc)
            seq.add_block(self.epi.gx, self.epi.adc)
            seq.add_block(pp.make_delay(delayTR))

    # =========================================================================
    # Write sequence to file
    # =========================================================================`

    def write(self, filename=None, overwrite=True):
        """Write the sequence to a Pulseq ``.seq`` file with embedded metadata.

        Embeds FOV, echo times, TR, slice thickness, diffusion directions, b-value,
        and Stejskal-Tanner timing in the file header so that the scanner and
        offline tools can parse them without a separate sidecar.

        Args:
            filename: Destination filename.  ``None`` uses :meth:`get_save_filename`.
                Relative filenames are resolved against ``self.save_dir``.
            overwrite: If ``True`` (default), silently overwrite an existing file.
                If ``False``, skip writing and log a warning.
        """
        if filename is None:
            filename = self.get_save_filename(full_path=True)
        else:
            filename = os.path.join(self.save_dir, filename)

        # Set metadata definitions in the sequence object for export
        self.seq.set_definition("FOV", [self.fov, self.fov, self.slice_thickness])
        self.seq.set_definition("Name", "epi_diffusion_se_" + self.name)
        self.seq.set_definition("TE", [self.TE, self.TE2, self.TE3])
        self.seq.set_definition("TR", self.TR)
        self.seq.set_definition("SliceThickness", self.slice_thickness)
        self.seq.set_definition("NNavigatorLines", 3)
        self.seq.set_definition("DiffusionDirections", self.b_directions.tolist())
        self.seq.set_definition("bValue", int(self.b_value))
        self.seq.set_definition("b0Frequency", self.b_0_frequency)
        self.seq.set_definition("SmallDelta", self.small_delta)
        self.seq.set_definition("BigDelta", self.big_DELTA)
        self.seq.set_definition("AdcNumSamples", self.adc.num_samples)
        self.seq.set_definition("AdcDwellTime", self.adc.dwell)
        self.seq.set_definition("AccelerationFactor", self.epi_acceleration_factor)
        self.seq.set_definition("PartialFourierFactor", self.partial_fourier_factor)
        self.seq.set_definition("Nx", self.Nx)
        self.seq.set_definition("Ny", self.Ny)
        self.logger.info(f"Writing sequence to file: {filename}")

        if os.path.exists(filename):
            if overwrite:
                self.logger.warning(
                    f"File {filename} already exists and overwrite=True. Overwriting."
                )
                self.seq.write(filename, v141_compat=self.v141_compat)
            else:
                self.logger.warning(
                    f"File {filename} already exists and overwrite=False. Skipping write."
                )
                return
        else:
            self.seq.write(filename, v141_compat=self.v141_compat)

    # =========================================================================
    # Sequence build (triple SE)
    # =========================================================================

    def build_seq(self, old_seq=None):
        """Assemble the complete triple spin-echo diffusion EPI sequence.

        Iterates over all diffusion directions and, for each direction:

        1. Scales the diffusion gradient amplitude along (gx, gy, gz) using the
           unit direction vector; each axis gets its own rise time calculated from
           its amplitude and ``max_slew``.
        2. Adds the RF90 excitation, the two diffusion gradient lobes flanking RF180_1,
           then the three EPI readout trains (each preceded by its prephaser and
           separated by RF180_2 and RF180_3 with spoilers).
        3. Appends end-spoilers (optional) and a TR padding delay.

        The ECO label (0/1/2) distinguishes the three echo images in the ICE
        reconstruction buffer.

        Args:
            old_seq: Existing :class:`pp.Sequence` to append blocks to (``None``
                creates a fresh sequence).  Useful for concatenating sequences.

        Returns:
            The assembled :class:`pp.Sequence` object (also stored as ``self.seq``).
        """
        if old_seq is None:
            seq = pp.Sequence(self.system)
        else:
            seq = old_seq

        if self.calibration_readout:
            if self.labeled:
                seq.add_block(
                    pp.make_label(label="NAV", type="SET", value=1),
                    pp.make_label(label="IMA", type="SET", value=0),
                    pp.make_label(label="REP", type="SET", value=0),
                    pp.make_label(label="ECO", type="SET", value=0),
                )  # Set calibration labels

            self._add_navigator_acquisition(seq)

            if self.labeled:
                seq.add_block(
                    pp.make_label(label="NAV", type="SET", value=0),
                    pp.make_label(label="IMA", type="SET", value=1),
                )  # Clear calibration labels

        for i, dir in enumerate(self.b_directions):
            if self.labeled:
                seq.add_block(
                    pp.make_label(label="REP", type="SET", value=i)
                )  # Set repetition label for diffusion direction

            self.logger.info(
                f"Generating sequence for b-direction: {dir}, b-value: {self.b_value}"
            )

            diffusion_gradients = self.diffusion_gradient_amplitude * dir
            self.logger.info(
                f"Diffusion gradient amplitudes (Hz/m): {diffusion_gradients} for b-value: {self.b_value} direction: {dir}"
            )

            # Per-axis rise time: minimum ramp to reach this axis's amplitude at max_slew.
            # Zero-amplitude axes use one grad-raster tick to avoid a zero-duration trapezoid.
            gx_rise_time = (
                align2rastertime_ceil(
                    abs(diffusion_gradients[0]) / self.system.max_slew,
                    self.system.grad_raster_time,
                )
                if abs(diffusion_gradients[0]) > 0
                else self.system.grad_raster_time
            )
            gy_rise_time = (
                align2rastertime_ceil(
                    abs(diffusion_gradients[1]) / self.system.max_slew,
                    self.system.grad_raster_time,
                )
                if abs(diffusion_gradients[1]) > 0
                else self.system.grad_raster_time
            )
            gz_rise_time = (
                align2rastertime_ceil(
                    abs(diffusion_gradients[2]) / self.system.max_slew,
                    self.system.grad_raster_time,
                )
                if abs(diffusion_gradients[2]) > 0
                else self.system.grad_raster_time
            )

            # All three axes must have the same total duration so they play simultaneously
            gx_flat_time = self.diffusion_gradient_total_duration - 2 * gx_rise_time
            gy_flat_time = self.diffusion_gradient_total_duration - 2 * gy_rise_time
            gz_flat_time = self.diffusion_gradient_total_duration - 2 * gz_rise_time

            self.logger.info(
                f"Individual rise times (ms): gx={gx_rise_time*1e3:.3f}, gy={gy_rise_time*1e3:.3f}, gz={gz_rise_time*1e3:.3f}"
            )
            self.logger.info(
                f"Individual flat times (ms): gx={gx_flat_time*1e3:.3f}, gy={gy_flat_time*1e3:.3f}, gz={gz_flat_time*1e3:.3f}"
            )

            gx_diff = pp.make_trapezoid(
                channel="x",
                system=self.system,
                amplitude=diffusion_gradients[0],
                rise_time=gx_rise_time,
                flat_time=gx_flat_time,
            )
            gy_diff = pp.make_trapezoid(
                channel="y",
                system=self.system,
                amplitude=diffusion_gradients[1],
                rise_time=gy_rise_time,
                flat_time=gy_flat_time,
            )
            gz_diff = pp.make_trapezoid(
                channel="z",
                system=self.system,
                amplitude=diffusion_gradients[2],
                rise_time=gz_rise_time,
                flat_time=gz_flat_time,
            )

            # ================================================================
            # 90 RF Pulse + Slice Select Gradient
            # ================================================================
            seq.add_block(self.rf90, self.gz90)
            seq.add_block(self.gz90_reph)

            # ================================================================
            # Delay before first diffusion gradient (to enforce big_DELTA)
            # ================================================================
            if self.delay_before_diff1 > 0:
                seq.add_block(pp.make_delay(self.delay_before_diff1))

            # ================================================================
            # First diffusion gradient
            # ================================================================
            seq.add_block(gx_diff, gy_diff, gz_diff)

            # ================================================================
            # Inner delay between diff_grad_1 and RF180
            # ================================================================
            if self.delayTE1_inner > 0:
                seq.add_block(pp.make_delay(self.delayTE1_inner))

            # ================================================================
            # First 180 RF Pulse (with optional spoilers)
            # ================================================================
            if self.rf180_spoiler:
                seq.add_block(self.spoiler1)

            seq.add_block(self.rf180_1, self.gz180)

            if self.rf180_spoiler:
                seq.add_block(self.spoiler1)

            # ================================================================
            # Second diffusion gradient
            # ================================================================
            seq.add_block(gx_diff, gy_diff, gz_diff)

            # ================================================================
            # Delay before EPI readout (to achieve TE)
            # ================================================================
            if self.delayTE2 > 0:
                seq.add_block(pp.make_delay(self.delayTE2))

            # ================================================================
            # EPI1 Prephasing gradients + Readout train
            # ================================================================
            seq.add_block(self.epi.gx_prephaser, self.epi.gy_prephaser)

            if self.labeled:
                seq.add_block(
                    pp.make_label(label="ECO", type="SET", value=0)
                )  # Set ECO label for first echo
                self.epi.add_to_sequence(seq)
            else:
                self.epi.add_to_sequence_unlabeled(seq)

            # ================================================================
            # Second Spin Echo: RF180_2 + EPI2
            # ================================================================
            if self.delay_before_rf180_2 > 0:
                seq.add_block(pp.make_delay(self.delay_before_rf180_2))

            seq.add_block(self.spoiler2)
            seq.add_block(self.rf180_2, self.gz180)
            seq.add_block(self.spoiler2)

            seq.add_block(self.epi2.gx_prephaser, self.epi2.gy_prephaser)

            if self.delay_before_epi2 > 0:
                seq.add_block(pp.make_delay(self.delay_before_epi2))

            if self.labeled:
                seq.add_block(
                    pp.make_label(label="ECO", type="SET", value=1)
                )  # Set ECO label for second echo
                self.epi2.add_to_sequence(seq)
            else:
                self.epi2.add_to_sequence_unlabeled(seq)

            # ================================================================
            # Third Spin Echo: RF180_3 + EPI3
            # ================================================================
            if self.delay_before_rf180_3 > 0:
                seq.add_block(pp.make_delay(self.delay_before_rf180_3))

            seq.add_block(self.spoiler3)
            seq.add_block(self.rf180_3, self.gz180)
            seq.add_block(self.spoiler3)

            seq.add_block(self.epi3.gx_prephaser, self.epi3.gy_prephaser)

            if self.delay_before_epi3 > 0:
                seq.add_block(pp.make_delay(self.delay_before_epi3))

            if self.labeled:
                seq.add_block(
                    pp.make_label(label="ECO", type="SET", value=2)
                )  # Set ECO label for third echo
                self.epi3.add_to_sequence(seq)
            else:
                self.epi3.add_to_sequence_unlabeled(seq)

            # ================================================================
            # End spoilers (optional)
            # ================================================================
            if self.end_spoilers:
                seq.add_block(self.spoiler_x, self.spoiler_y, self.spoiler_z)

            # ================================================================
            # TR delay
            # ================================================================
            seq.add_block(pp.make_delay(self.delayTR))

        self.seq = seq
        return seq

    # =========================================================================
    # Overridden PulseqSeq methods
    # =========================================================================

    def init_message(self):
        """Log a one-line banner announcing this sequence type at construction time."""
        self.logger.info(
            f"Initializing EPI Diffusion Triple SE Pulseq Sequence: {self.name}"
        )

    def metadata(self):
        """Flattened metadata from PulseqSeq + SE + Diffusion."""
        meta = {
            "name": self.name,
            "fov": self.fov,
            "Nx": self.Nx,
            "Ny": self.Ny,
            "slice_thickness": self.slice_thickness,
            "rf_duration": self.rf_duration,
            "flip_angle": self.flip_angle,
            "TE": self.TE,
            "b_value": self.b_value,
            "b_directions": self.b_directions.tolist(),
            "b_0_frequency": self.b_0_frequency,
            "small_delta": self.small_delta,
            "big_DELTA": self.big_DELTA,
            "diffusion_gradient_ramp_duration_s": self.diffusion_gradient_rise_time,
            "diffusion_gradient_amplitude_mTm": self.diffusion_gradient_amplitude,
        }
        return meta

    def get_save_filename(self, full_path=False) -> str:
        """Flattened filename from PulseqSeq + SE + Diffusion + EPI."""
        if self.save_name is not None:
            return os.path.join(self.save_dir, self.save_name)

        # Build base filename (PulseqSeq)
        filename = f"{self.name}{'_v14' if self.v141_compat else '_v15'}_{self.system_type.value}_fov{self.fov*1000:.0f}mm_{self.Ny}x{self.Nx}x{self.N_slices}_TR{self.TR*1000:.0f}ms"
        # SE suffix
        filename += (
            f"_TEs[{self.TE*1000:.0f}-{self.TE2*1000:.0f}-{self.TE3*1000:.0f}]ms"
        )
        # Diffusion suffix
        filename += f"_b{self.b_value}_dirs{self.b_dirs}{f'_b0s{self.b_0_frequency}' if self.b_0_frequency else ''}_delta{self.small_delta*1e3:.2f}ms_DELTA{self.big_DELTA*1e3:.2f}ms"
        # EPI suffix
        if self.ramp_sampling == "none":
            ramping_str = ""
        elif self.ramp_sampling == "optimized":
            ramping_str = "_opt"
        elif self.ramp_sampling == "ramp_sampled":
            ramping_str = "_rs"
        filename += f"_pff{self.partial_fourier_factor*100:.0f}_acc{self.epi_acceleration_factor}{ramping_str}"

        # Blip direction suffix
        blip_dirs_str = ["d" if blip else "u" for blip in self.blip_down]
        filename += f"_blips-{''.join(blip_dirs_str)}"

        filename += ".seq"

        if full_path:
            return os.path.join(self.save_dir, filename)
        return filename

    def validate_sequence_properties(
        self, expected_values: dict = None, tolerance: float = None
    ) -> tuple[bool, list[str]]:
        """Flattened validation from PulseqSeq + DiffusionSE."""
        # Run base PulseqSeq validation
        all_passed, failed_tests = super().validate_sequence_properties(
            expected_values, tolerance
        )

        self.logger.info("Validating Diffusion SE sequence properties...")

        # Validate achieved b-value
        if self.b_value > 0:
            b_calc = calc_bval(
                self.diffusion_gradient_amplitude / 1000,
                self.small_delta,
                self.big_DELTA,
                self.diffusion_gradient_rise_time,
            )
            b_tol_abs = max(10.0, 0.03 * self.b_value)
            b_matches = np.isclose(b_calc, self.b_value, atol=b_tol_abs, rtol=0.0)

            if not b_matches:
                msg = (
                    f"b-value validation: calculated={b_calc:.2f} s/mm^2, "
                    f"requested={self.b_value:.2f} s/mm^2, "
                    f"difference={abs(b_calc - self.b_value):.2f} s/mm^2 (tolerance: +/-{b_tol_abs:.2f} s/mm^2)"
                )
                self.logger.error(msg)
                failed_tests.append(msg)
                all_passed = False
            else:
                self.logger.info(
                    f"b-value validation: requested={self.b_value:.2f}, calculated={b_calc:.2f} s/mm^2"
                )

        # Validate diffusion timing
        if self.small_delta > 0 and self.big_DELTA > 0:
            if self.big_DELTA <= self.small_delta:
                msg = f"Diffusion timing: big_DELTA ({self.big_DELTA*1e3:.2f} ms) must be > small_delta ({self.small_delta*1e3:.2f} ms)"
                self.logger.error(msg)
                failed_tests.append(msg)
                all_passed = False
            else:
                self.logger.info(
                    f"Diffusion timing: small_delta={self.small_delta*1e3:.2f} ms, big_DELTA={self.big_DELTA*1e3:.2f} ms"
                )

        # Validate delay timing
        delay_checks = [
            ("delay_before_diff1", self.delay_before_diff1),
            ("delayTE1_inner", self.delayTE1_inner),
            ("delayTE2", self.delayTE2),
        ]
        for delay_name, delay_val in delay_checks:
            if delay_val < 0:
                msg = f"{delay_name} is negative: {delay_val*1e3:.2f} ms"
                self.logger.error(msg)
                failed_tests.append(msg)
                all_passed = False
            else:
                self.logger.info(f"{delay_name}: {delay_val*1e3:.2f} ms")

        self.logger.info("=" * 60)
        if all_passed:
            self.logger.info("All validation checks PASSED")
        else:
            self.logger.warning(f"Validation: {len(failed_tests)} check(s) FAILED")
        self.logger.info("=" * 60)

        return all_passed, failed_tests

    def validate_echo_timing(self, tolerance_us: float = 1.0) -> tuple[bool, list[str]]:
        """
        Validate that each spin echo center occurs at the expected TE.

        Re-derives all three echo times from stored timing parameters and
        compares them to self.TE, self.TE2, self.TE3.  Also checks the
        spin-echo symmetry condition: each RF180 must be exactly equidistant
        between the two surrounding echoes (or between RF90 and EPI1 for the
        first refocusing pulse).

        Parameters
        ----------
        tolerance_us : float
            Acceptable timing deviation in microseconds (default 1 µs ≈ 1 grad
            raster tick at 10 µs raster — tighten for debugging).

        Returns
        -------
        passed : bool
        errors : list[str]
            One entry per failed check with computed vs. expected values.
        """
        tol = tolerance_us * 1e-6
        errors: list[str] = []
        passed = True

        spoiler_dur = pp.calc_duration(self.spoiler1) if self.rf180_spoiler else 0.0

        # ── Echo 1: RF90 centre → EPI1 k-space centre ────────────────────
        # first half  = RF90 centre → RF180_1 centre
        te1_half_before = (
            self.time_after_90
            + pp.calc_duration(self.gz90_reph)
            + self.delay_before_diff1
            + self.diffusion_gradient_duration
            + self.delayTE1_inner
            + spoiler_dur
            + self.rf180_center_with_delay
        )
        # second half = RF180_1 centre → EPI1 echo centre
        te1_half_after = (
            self.time_after_180
            + spoiler_dur
            + self.diffusion_gradient_duration
            + self.delayTE2
            + self.prephaser_duration
            + self.time_until_echo
        )
        echo1_time = te1_half_before + te1_half_after

        # ── Echo 2: RF90 centre → EPI2 k-space centre ────────────────────
        time_after_epi1 = self.epi_duration - self.time_until_echo

        # first half  = EPI1 echo → RF180_2 centre
        te2_half_before = (
            time_after_epi1
            + self.delay_before_rf180_2
            + spoiler_dur
            + self.rf180_center_with_delay
        )
        # second half = RF180_2 centre → EPI2 echo centre
        te2_half_after = (
            self.time_after_180
            + spoiler_dur
            + self.epi2.prephaser_duration
            + self.delay_before_epi2
            + self.epi2.time_until_echo
        )
        echo2_time = echo1_time + te2_half_before + te2_half_after

        # ── Echo 3: RF90 centre → EPI3 k-space centre ────────────────────
        time_after_epi2 = self.epi2.duration - self.epi2.time_until_echo

        # first half  = EPI2 echo → RF180_3 centre
        te3_half_before = (
            time_after_epi2
            + self.delay_before_rf180_3
            + spoiler_dur
            + self.rf180_center_with_delay
        )
        # second half = RF180_3 centre → EPI3 echo centre
        te3_half_after = (
            self.time_after_180
            + spoiler_dur
            + self.epi3.prephaser_duration
            + self.delay_before_epi3
            + self.epi3.time_until_echo
        )
        echo3_time = echo2_time + te3_half_before + te3_half_after

        self.logger.info("=" * 60)
        self.logger.info("Echo Timing Validation")
        self.logger.info("=" * 60)

        # ── TE value checks ───────────────────────────────────────────────
        for label, computed, expected in [
            ("TE1", echo1_time, self.TE),
            ("TE2", echo2_time, self.TE2),
            ("TE3", echo3_time, self.TE3),
        ]:
            err_us = (computed - expected) * 1e6
            ok = abs(err_us) <= tolerance_us
            msg = (
                f"{label}: expected={expected*1e3:.4f} ms, "
                f"computed={computed*1e3:.4f} ms, "
                f"error={err_us:+.2f} µs — {'OK' if ok else 'FAIL'}"
            )
            self.logger.info(msg)
            if not ok:
                errors.append(msg)
                passed = False

        # ── RF180 symmetry checks ─────────────────────────────────────────
        # Spin echo forms only when the refocusing pulse sits exactly halfway
        # between the two surrounding echo centers.
        for label, half_before, half_after in [
            ("RF180_1", te1_half_before, te1_half_after),
            ("RF180_2", te2_half_before, te2_half_after),
            ("RF180_3", te3_half_before, te3_half_after),
        ]:
            err_us = (half_before - half_after) * 1e6
            ok = abs(err_us) <= tolerance_us
            msg = (
                f"{label} symmetry: {half_before*1e3:.4f} ms before, "
                f"{half_after*1e3:.4f} ms after, "
                f"asymmetry={err_us:+.2f} µs — {'OK' if ok else 'FAIL'}"
            )
            self.logger.info(msg)
            if not ok:
                errors.append(msg)
                passed = False

        self.logger.info("=" * 60)
        if passed:
            self.logger.info("Echo timing validation PASSED")
        else:
            self.logger.warning(
                f"Echo timing validation: {len(errors)} check(s) FAILED"
            )
        self.logger.info("=" * 60)

        return passed, errors

    def print_spoiler_info(self):
        """Print amplitude, duration, area, and axis for each of the three spoiler gradients.

        Only prints useful output when ``rf180_spoiler=True``; otherwise logs a notice
        that no RF180 spoiler was requested.
        """
        if self.rf180_spoiler:
            print(
                f"Spoiler 1:" + f"\n\t  Amplitude: {self.spoiler1.amplitude} Hz/m"
                f"\n\t  Duration: {pp.calc_duration(self.spoiler1)} s"
                f"\n\t  Area: {self.spoiler1.area} Hz*s/m"
                + f"\n\t  Axis: {self.spoiler1.channel}"
            )
            print(
                f"Spoiler 2:" + f"\n\t  Amplitude: {self.spoiler2.amplitude} Hz/m"
                f"\n\t  Duration: {pp.calc_duration(self.spoiler2)} s"
                f"\n\t  Area: {self.spoiler2.area} Hz*s/m"
                + f"\n\t  Axis: {self.spoiler2.channel}"
            )
            print(
                f"Spoiler 3:" + f"\n\t  Amplitude: {self.spoiler3.amplitude} Hz/m"
                f"\n\t  Duration: {pp.calc_duration(self.spoiler3)} s"
                f"\n\t  Area: {self.spoiler3.area} Hz*s/m"
                + f"\n\t  Axis: {self.spoiler3.channel}"
            )
        else:
            self.logger.info("No RF180 spoiler applied.")

    def plot(
        self,
        TRs=0,
        show_blocks: bool = False,
        save: bool = False,
        time_range=None,
        time_disp: str = "s",
        grad_disp: str = "kHz/m",
        plot_now: bool = True,
    ):
        """
        Plot the sequence diagram.

        Args:
            TRs: Number of TRs to display (0 for all).
            show_blocks: Show block structure (default=False).
            save: Save the plot (default=False).
            time_range: Custom time range tuple (optional).
            time_disp: Time display unit (default='s').
            grad_disp: Gradient display unit (default='kHz/m').
            plot_now: Display plot immediately (default=True).
        """
        if self.TR:
            time_range = (
                (0, self.TR * TRs if TRs > 0 else np.inf)
                if time_range is None
                else time_range
            )

        self.seq.plot(
            time_range=time_range,
            time_disp=time_disp,
            grad_disp=grad_disp,
            show_blocks=show_blocks,
            plot_now=False,
        )

        # Add vertical lines for TEs
        fig = plt.gcf()

        # Get TE times for all directions
        te_times = []
        te_labels = []

        te_times_tr = [self.TE, self.TE2, self.TE3]
        te_labels_tr = ["TE1", "TE2", "TE3"]
        for n in range(self.b_directions.shape[0]):
            te_times.extend([t + n * self.TR for t in te_times_tr])

        for idx, te_time in enumerate(te_times):
            for ax in fig.axes:
                idx_mod = idx % len(te_labels_tr)
                ax.axvline(
                    x=te_time,
                    color="red",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.7,
                    label=te_labels_tr[idx_mod],
                )

        if save:
            plt.savefig(
                os.path.join(self.save_dir, f"{self.name}_sequence_diagram.png"),
                dpi=500,
                bbox_inches="tight",
            )
        plt.show()


# %%
if __name__ == "__main__":
    acceleration_factor = 1
    pff = 0.75
    res = 2.33333333333333333333333333333
    dwell = 5 * 0.000001

    results = {'blipup': [], 'blipdown': []}
    for te in range(75, 110, 5):
        for bp in [True, False]:
            mesepi = EPIDiffusionTripleSEPulseqSeq(
                name=f"DiffMESE",
                fov=224e-3,
                Nx=96,
                Ny=96,
                resolution=res,
                slice_thickness=res * 1e-3,
                partial_fourier_factor=0.75,
                TR=5000,
                TE=te,
                rf90_duration=0.003,
                rf180_duration=0.003,
                dwell_time=dwell,
                prephaser_duration=0.0005,
                rephasers=True,
                simultan_rephasers=False,
                system_type=SystemLimitType.EXTRASAFE,
                rf180_spoiler=True,
                ramp_sampling="ramp_sampled",
                spoiler_amplitude=0.95,
                b_0_frequency=3,
                b_directions=3,
                b_value=500,
                small_delta=0.018,
                big_DELTA=0.035,
                acceleration_factor=1,
                v141_compat=True,
                fit_epi=False,
                calibration_readout=True,
                adc_dead_time_correction=True,
                uniform_spoiler_areas=False,
                uniform_spoiler_directions=False,
                phase_cycling=False,
                labeled=True,
                blip_down=bp,
                alternating_blip_polarity=True,
            )
            # mesepi.print_spoiler_info()
            # mesepi.plot(time_range=(4.995, 5.225),)
            # mesepi.plot_kspace_traj()
            # mesepi.write()
            # mesepi.report()
            # plot_gradient_and_slew(mesepi.seq)
            # mesepi.validate_echo_timing()
            if bp:
                results['blipdown'].append((mesepi.TE, mesepi.TE2, mesepi.TE3))
            else:
                results['blipup'].append((mesepi.TE, mesepi.TE2, mesepi.TE3))
    print(results)
    # %%
