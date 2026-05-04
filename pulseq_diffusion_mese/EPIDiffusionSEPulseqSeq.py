"""
EPIDiffusionSEPulseqSeq — Diffusion-weighted single spin-echo EPI sequence.

Implements a Stejskal–Tanner diffusion preparation (one pair of bipolar gradients
flanking a single 180° refocusing pulse) followed by one EPI readout.  This is
the simplest diffusion-EPI topology and the reference against which the triple-SE
variant (:class:`EPIDiffusionTripleSEPulseqSeq`) is compared.

Inherits directly from :class:`PulseqSeq` (flat hierarchy — no intermediate classes).
Uses :class:`EPIReadout` (standalone, no inheritance) for the readout train.

Sequence timeline per diffusion direction within one TR::

    RF90 → gz90_reph → [delay] → Gdiff1 → [delayTE1_inner] → [spoiler] → RF180
         → [spoiler] → Gdiff2 → [delayTE2] → EPI-prephase → EPI (echo at TE)
         → [end-spoilers] → delayTR

Key features:
    - Automatic TE feasibility checking with partial-Fourier reduction fallback
    - Per-axis diffusion gradient design (amplitude scaled per direction vector)
    - Hardware amplitude clamping with achieved-b-value reporting when b exceeds limits
    - Optional navigator / calibration readout (3-line spin echo before the DWI loop)
    - Configurable blip polarity (blip-down or blip-up)
    - Labeled and unlabeled (simulation) output modes

Author      : Aron Gimesi <aron.gimesi@tecnico.ulisboa.pt>
Affiliation : Instituto Superior Técnico | MSCA-DN IQ-BRAIN
Date        : 2026
Context     : ESMRMB 2026 — Pulseq DiffusionMESE showcase

Funding acknowledgement (mandatory):
    IQ-BRAIN is funded by the European Union (MSCA Doctoral Network,
    December 2024–November 2028, Grant Agreement No. 101169519).
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


class EPIDiffusionSEPulseqSeq(PulseqSeq):
    """
    Single spin-echo EPI diffusion sequence.

    Flat hierarchy: inherits only from PulseqSeq.
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
        rephasers: bool = False,
        simultan_rephasers: bool = True,
        blip_down: bool = True,
        ramp_sampling: str = "none",  # 'none', 'optimized', 'ramp_sampled'
        eddy_currents: bool = False,
        eddy_currents_induced_delay: float = 0.0000015,
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
    ):
        """
        Initialise and build the single spin-echo diffusion EPI sequence.

        Calls all ``_init_*`` helpers in dependency order, then runs the EPI fit
        loop (reducing partial_fourier_factor until the readout fits in TE) and
        calls ``build_seq``.  If ``small_delta``/``big_DELTA`` are both given they
        are treated as fixed; if either is ``None`` both are auto-derived from the
        TE window.

        Args:
            name: Sequence identifier used in filenames and log messages.
            fov: Field of view in metres (isotropic in-plane).
            Nx: Readout matrix size.
            Ny: Phase-encode matrix size (before partial-Fourier reduction).
            slice_thickness: Slice thickness in metres.
            TR: Repetition time in milliseconds.
            TE: Echo time in milliseconds.
            b_value: Diffusion weighting in s/mm².
            b_directions: Number of DWI directions (electrostatic scheme).
            b_0_frequency: Number of interleaved b=0 acquisitions.
            small_delta: Diffusion gradient duration in seconds (``None`` = auto).
            big_DELTA: Diffusion gradient separation in seconds (``None`` = auto).
            N_slices: Number of slices per TR.
            system_type: Scanner hardware limits (``SystemLimitType`` enum).
            rf90_duration: Duration of the 90° sinc pulse in seconds.
            rf180_duration: Duration of the 180° sinc pulse (0 = same as rf90).
            resolution: Isotropic in-plane resolution in mm; overrides Nx/Ny.
            flip_angle: Excitation flip angle in degrees.
            apodization: Sinc apodization factor (0–1).
            time_bw_product: Time–bandwidth product for sinc pulses.
            dwell_time: ADC dwell time in seconds (``None`` = auto-selected).
            end_spoilers: Apply spoiler gradients after the EPI readout.
            rf180_spoiler: Apply crusher gradients flanking the 180° pulse.
            spoiler_amplitude: Spoiler gradient amplitude as a fraction of max_grad.
            spoiler_duration: Spoiler gradient duration in seconds.
            prephaser_duration: Fixed EPI prephaser duration in seconds (``None`` = auto).
            rephasers: Append readout rephaser gradients after the EPI train.
            simultan_rephasers: Play x/y rephasers simultaneously.
            blip_down: Use blip-down (negative ky) phase-encode direction.
            ramp_sampling: ADC sampling strategy (``'none'``, ``'optimized'``, ``'ramp_sampled'``).
            eddy_currents: Reserved for future eddy-current compensation (unused).
            eddy_currents_induced_delay: Eddy-current-induced delay in seconds (unused).
            save_dir: Output directory for ``.seq`` files.
            logger: External logger; defaults to a new ``EPIDiffusionSEPulseqSeqV4`` logger.
            partial_fourier_factor: Fraction of k-space lines acquired (0.5–1.0).
            calibration_readout: Prepend a 3-line navigator for N/2 ghost correction.
            acceleration_factor: EPI acceleration (GRAPPA-style line skipping).
            max_EPI_duration: Hard cap on EPI readout duration in seconds (``None`` = unconstrained).
            oversampling_factor: Readout oversampling multiplier (currently stored but unused).
            v141_compat: Write Pulseq v1.4.1-compatible ``.seq`` files instead of v1.5.
            adc_dead_time_correction: Shift ADC window to account for hardware dead time.
            fit_epi: If ``True``, automatically reduce ``partial_fourier_factor`` until
                the EPI readout fits in TE; if ``False``, raise on failure.
            save_name: Override the auto-generated filename (without directory).
            labeled: Emit Pulseq ``LABEL`` blocks (required for online reconstruction).
        """
        self.save_name = save_name

        # --- EPI parameters (from EPIDiffusionSEPulseqSeqV2._init_epi_params) ---
        self.partial_fourier_factor = partial_fourier_factor
        self.calibration_readout = calibration_readout
        self.epi_acceleration_factor = acceleration_factor
        self.oversampling_factor = oversampling_factor
        self.prephaser_duration = prephaser_duration
        self.rephasers = rephasers
        self.simultan_rephasers = simultan_rephasers
        self.blip_down = blip_down
        self.ramp_sampling = ramp_sampling
        self.max_EPI_duration = max_EPI_duration
        self.fit_epi = fit_epi
        self.adc_dead_time_correction = adc_dead_time_correction
        self.labeled = labeled

        # --- PulseqSeq base init ---
        self._init_logging(logger or logging.getLogger("EPIDiffusionSEPulseqSeqV4"), name, system_type, save_dir)
        self._init_system(system_type)
        self._init_imaging_params(fov, Nx, Ny, slice_thickness, TR, N_slices, resolution, flip_angle, apodization, time_bw_product, v141_compat)
        self._init_readout_timing(dwell_time)
        self._init_rf90(rf90_duration)
        self._init_spoilers(end_spoilers, spoiler_amplitude, spoiler_duration)

        # --- Diffusion parameters (from DiffusionSEPulseqSeq._init_diffusion_params) ---
        self._init_diffusion_params(b_value, b_directions, b_0_frequency, small_delta, big_DELTA, rf180_spoiler)

        # --- Spin-echo RF180 (from SEPulseqSeq._init_se) ---
        self._init_se(TE, rf180_duration)

        # --- EPI fit loop + timing ---
        self._run_epi_fit_loop()
        self._calc_epi_tr_delay()
        self.diffusion_gradient_amplitudes = [self.diffusion_gradient_amplitude] * self.b_directions
        self.build_seq()

    # =========================================================================
    # Init helpers (flattened from intermediate classes)
    # =========================================================================

    def _init_diffusion_params(self, b_value, b_directions, b_0_frequency, small_delta, big_DELTA, rf180_spoiler):
        """
        Store diffusion parameters and generate the direction table.

        Converts the integer ``b_directions`` count into a unit-vector array via
        ``get_diffusion_directions``.  Caches the user-supplied ``small_delta`` /
        ``big_DELTA`` separately so the fit loop can reset them on each iteration
        without losing the original intent.
        """
        self.rf180_spoiler = rf180_spoiler
        self.b_value: int = b_value
        self.b_0_frequency: int = b_0_frequency
        self.b_dirs = b_directions
        self.b_directions: np.ndarray = get_diffusion_directions(b_directions, b_0_frequency)
        self.small_delta: float = small_delta
        self.big_DELTA: float = big_DELTA
        self._user_small_delta = small_delta
        self._user_big_DELTA = big_DELTA

        if self.big_DELTA or self.small_delta:
            assert self.big_DELTA and self.small_delta, "Both big_DELTA and small_delta must be set together."

    def _init_se(self, TE, rf180_duration):
        """
        Create the 180° refocusing pulse and placeholder readout objects.

        The readout objects (``gx``, ``adc``, ``gx_pre``, ``gy_pre_dummy``) created
        here are temporary scaffolds so that subsequent code paths can reference them
        before ``_try_epi_fit`` overwrites them with the real EPI versions.  Using
        ``rf180_duration=0`` mirrors the 90° pulse duration, which is the standard
        paired-pulse convention.
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

        # Phase encoding gradient amplitudes
        self.phase_encoding_gradients = (np.arange(self.Ny) - self.Ny / 2) * self.delta_k

        # SE readout objects — these get overwritten by _try_epi_fit but are needed
        # for the intermediate _init_se path to complete without error.
        self.flat_time_raster = align2rastertime_ceil(self.readout_time, 2 * self.system.grad_raster_time)
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

    # =========================================================================
    # EPI fit loop
    # =========================================================================

    def _run_epi_fit_loop(self):
        """
        Iteratively reduce ``partial_fourier_factor`` until the EPI readout fits in TE.

        Each iteration resets ``small_delta``/``big_DELTA`` to the user-supplied values
        (or ``None`` for auto) so that ``_try_epi_fit`` sees a clean state.  Steps down
        in 5 % increments; raises if PF factor would fall below 0.5 (half k-space).
        When ``fit_epi=False`` any failure is immediately fatal.
        """
        epi_fit = False
        fit_step = 0.05  # 5 % PF reduction per retry — coarse enough to converge quickly
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
                self.partial_fourier_factor = np.round(self.partial_fourier_factor - fit_step, 2)
                if self.partial_fourier_factor < 0.5:
                    raise ValueError("Partial Fourier factor reduced below 0.5, cannot fit EPI readout in TE.")

    def _try_epi_fit(self):
        """
        Attempt to fit the EPI readout within TE timing constraints.

        Returns (success, error_message).
        """
        epi = EPIReadout(
            system=self.system,
            Nx=self.Nx,
            Ny=self.Ny,
            fov=self.fov,
            dwell_time=self.dwell_time,
            partial_fourier_factor=self.partial_fourier_factor,
            blip_down=self.blip_down,
            prephaser_duration=self.prephaser_duration,
            acceleration_factor=self.epi_acceleration_factor,
            ramp_sampling=self.ramp_sampling,
            rephasers=self.rephasers,
            simultan_rephasers=self.simultan_rephasers,
            max_duration=self.max_EPI_duration,
            adc_dead_time_correction=self.adc_dead_time_correction,
        )

        gx_pre = epi.gx_prephaser
        gy_pre = epi.gy_prephaser
        self.prephaser_duration = epi.prephaser_duration
        self.time_until_echo = epi.time_until_echo
        self.epi_duration = epi.duration

        # ======================================================================
        # TE Calculations (before diffusion gradient adjustment)
        # ======================================================================
        rf90_center = pp.calc_rf_center(self.rf90)[0]
        rf180_center = pp.calc_rf_center(self.rf180)[0]

        rf90_center_with_delay = rf90_center + self.rf90.delay
        time_after_90 = pp.calc_duration(self.rf90, self.gz90) - rf90_center_with_delay

        rf180_center_with_delay = rf180_center + self.rf180.delay
        time_after_180 = pp.calc_duration(self.rf180, self.gz180) - rf180_center_with_delay

        # Calculate delayTE1 (time available between gz90_reph and rf180 center)
        delayTE1_raw = self.TE / 2 - (time_after_90 + pp.calc_duration(self.gz90_reph) + rf180_center_with_delay)
        if self.rf180_spoiler:
            delayTE1_raw -= pp.calc_duration(self.spoiler_z)

        # Calculate delayTE2 (time available between rf180 and echo)
        delayTE2_raw = self.TE / 2 - (time_after_180 + self.prephaser_duration + self.time_until_echo)
        if self.rf180_spoiler:
            delayTE2_raw -= pp.calc_duration(self.spoiler_z)

        # Check basic TE feasibility before diffusion
        if delayTE2_raw < 0:
            self.logger.warning(f"TE too short for EPI readout! delayTE2_raw={delayTE2_raw*1e3:.3f} ms")
            return (False, f"DelayTE2 ({delayTE2_raw*1e3:.3f} ms) is negative")

        delayTE1 = align2rastertime_nearest(delayTE1_raw, self.system.grad_raster_time)
        delayTE1_error = delayTE1 - delayTE1_raw

        # delayTE1 rounding shifts total TE; absorb the error into delayTE2 before
        # rounding it, so the two half-echoes sum to exactly the requested TE.
        delayTE2_compensated = delayTE2_raw - delayTE1_error
        delayTE2 = align2rastertime_nearest(delayTE2_compensated, self.system.grad_raster_time)

        if delayTE1 < 0:
            self.logger.warning(f"delayTE1 ({delayTE1*1e3:.3f} ms) is negative!")
            return (False, f"DelayTE1 ({delayTE1*1e3:.3f} ms) is negative!")

        # ======================================================================
        # Diffusion Gradient Calculations
        # ======================================================================
        time_rf180_block = pp.calc_duration(self.rf180, self.gz180)
        if self.rf180_spoiler:
            time_rf180_block += 2 * pp.calc_duration(self.spoiler_z)

        if self.small_delta is None or self.big_DELTA is None:
            available_window = min(delayTE1, delayTE2)
            max_ramp_time = align2rastertime_ceil(self.system.max_grad / self.system.max_slew, self.system.grad_raster_time)
            small_delta = available_window - 2 * max_ramp_time
            big_DELTA = delayTE1 + time_rf180_block
        else:
            small_delta = self.small_delta
            big_DELTA = self.big_DELTA

        if small_delta <= 0:
            self.logger.warning(f"small_delta ({small_delta*1e3:.3f} ms) is non-positive!")
            return (False, f"small_delta ({small_delta*1e3:.3f} ms) is non-positive!")

        diffusion_gradient_amplitude = calc_diffusion_gradient_amplitude(self.b_value, small_delta, big_DELTA)

        # Store for individual gradient calculations - will calculate individual rise times per axis
        self.diffusion_gradient_amplitude = diffusion_gradient_amplitude
        self.small_delta = small_delta

        # ======================================================================
        # Amplitude clamp: if G_req > max_grad, try area-preserving fallback
        # ======================================================================
        self._amplitude_clamped = False
        if diffusion_gradient_amplitude > self.system.max_grad:
            if self._user_small_delta is None:
                # Auto-calculated small_delta: amplitude overshoot means the TE window
                # is too tight. Return False so the fit loop can reduce pff and retry
                # with a longer auto-calculated small_delta.
                self.logger.warning(
                    f"Required diffusion amplitude ({diffusion_gradient_amplitude:.0f} Hz/m) "
                    f"exceeds max_grad ({self.system.max_grad:.0f} Hz/m) for auto small_delta="
                    f"{small_delta*1e3:.2f} ms. Returning False for fit loop to retry."
                )
                return (False, f"Amplitude {diffusion_gradient_amplitude:.0f} Hz/m exceeds max_grad for auto small_delta")

            # User-specified small_delta: honour the fixed duration, clamp amplitude
            # to max_grad, and accept the reduced b-value.
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

            # Clamped solution accepted — log actual achieved b-value
            actual_b = (
                calc_bval(
                    self.system.max_grad / self.system.gamma,  # Hz/m -> T/m
                    trap_params["delta_eff"],
                    big_DELTA,
                    trap_params["rise_time"],
                )
                * 1e-6
            )  # s/m^2 -> s/mm^2
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
                self.logger.error(f"Could not create area-preserving diffusion gradient: {e}")
                return (False, f"Could not create area-preserving diffusion gradient: {e}")

            self.diffusion_gradient_flat_time = g_diffusion_dummy.flat_time
            self.diffusion_gradient_rise_time = g_diffusion_dummy.rise_time
        else:
            # Normal path: amplitude within limits
            # Calculate maximum rise time for validation (worst case scenario)
            max_rise_time = align2rastertime_ceil(diffusion_gradient_amplitude / self.system.max_slew, self.system.grad_raster_time)
            min_flat_time = small_delta - 2 * max_rise_time

            if min_flat_time < 0:
                self.logger.error(f"minimum flat time ({min_flat_time*1e3:.3f} ms) is negative!")
                return (False, f"small_delta too short for worst-case ramp times")

            try:
                # Create dummy gradient for duration calculation using maximum amplitude
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
                self.logger.error(f"Could not create diffusion gradient with amplitude {diffusion_gradient_amplitude} Hz/m: {e}")
                return (False, f"Could not create diffusion gradient: {e}")

        diffusion_gradient_duration = pp.calc_duration(g_diffusion_dummy)

        # ======================================================================
        # Calculate delays to enforce big_DELTA timing
        # ======================================================================
        delayTE1_inner = big_DELTA - diffusion_gradient_duration - time_rf180_block
        delayTE1_inner = align2rastertime_nearest(delayTE1_inner, self.system.grad_raster_time)

        if delayTE1_inner < 0:
            self.logger.error(f"delayTE1_inner ({delayTE1_inner*1e3:.3f} ms) is negative! big_DELTA may be too small.")
            return (False, f"delayTE1_inner ({delayTE1_inner*1e3:.3f} ms) is negative")

        delay_before_diff1 = delayTE1 - diffusion_gradient_duration - delayTE1_inner
        delay_before_diff1 = align2rastertime_nearest(delay_before_diff1, self.system.grad_raster_time)

        if delay_before_diff1 < 0:
            self.logger.error(f"delay_before_diff1 ({delay_before_diff1*1e3:.3f} ms) is negative! big_DELTA may be too large for TE.")
            return (False, f"delay_before_diff1 ({delay_before_diff1*1e3:.3f} ms) is negative")

        delayTE2_adjusted = delayTE2 - diffusion_gradient_duration
        delayTE2_adjusted = align2rastertime_nearest(delayTE2_adjusted, self.system.grad_raster_time)

        if delayTE2_adjusted < 0:
            self.logger.error(f"delayTE2_adjusted ({delayTE2_adjusted*1e3:.3f} ms) is negative!")
            return (False, f"delayTE2_adjusted ({delayTE2_adjusted*1e3:.3f} ms) is negative")

        # ======================================================================
        # Tick-based final TE correction (no re-rounding needed)
        # ======================================================================
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
        # Round residual TE error to the nearest raster tick and absorb into delayTE2.
        # This compensates accumulated float arithmetic drift without re-rounding other delays.
        te_correction_ticks = int(np.floor((self.TE - (actual_TE1 + actual_TE2)) / self.system.grad_raster_time + 0.5))
        if te_correction_ticks != 0:
            self.logger.info(f"TE tick correction: {te_correction_ticks} ticks ({te_correction_ticks * self.system.grad_raster_time * 1e6:.1f} us)")
        delayTE2_adjusted += te_correction_ticks * self.system.grad_raster_time

        if delayTE2_adjusted < 0:
            self.logger.error(f"delayTE2_adjusted after TE correction ({delayTE2_adjusted*1e3:.3f} ms) is negative!")
            return (False, f"delayTE2_adjusted after TE correction ({delayTE2_adjusted*1e3:.3f} ms) is negative")

        # ======================================================================
        # Verify actual big_DELTA
        # ======================================================================
        actual_big_DELTA = diffusion_gradient_duration + delayTE1_inner + time_rf180_block
        self.logger.info(f"Requested big_DELTA: {big_DELTA*1e3:.2f} ms, Actual big_DELTA: {actual_big_DELTA*1e3:.2f} ms")
        self.logger.info(f"delay_before_diff1: {delay_before_diff1*1e3:.2f} ms, delayTE1_inner: {delayTE1_inner*1e3:.2f} ms")
        self.logger.info(f"delayTE2_adjusted: {delayTE2_adjusted*1e3:.2f} ms")
        self.logger.info(f"Calculated small_delta: {small_delta*1e3:.2f} ms")
        self.logger.info(f"Calculated big_DELTA: {big_DELTA*1e3:.2f} ms")
        self.logger.info(
            f"Diffusion amplitude: {diffusion_gradient_amplitude:.2f} Hz/m ({diffusion_gradient_amplitude / self.system.gamma * 1e3:.4f} mT/m)"
        )
        self.logger.info(f"Diffusion amplitude for b={self.b_value} s/mm^2")

        # ======================================================================
        # Store all calculated values
        # ======================================================================
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
        # Actual per-axis trapezoid duration (equals small_delta normally;
        # larger than small_delta when amplitude was clamped to max_grad)
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
    # TR delay calculation (single SE EPI)
    # =========================================================================

    def _calc_epi_tr_delay(self):
        """
        Compute the idle delay needed at the end of each TR block.

        Sums the duration of every event in the sequence timeline and subtracts from
        TR.  The result is raster-aligned to the nearest grad raster tick.  A negative
        ``delayTR`` means the sequence exceeds TR and ``build_seq`` will add an
        invalid (negative) delay block — caught by ``validate_sequence_properties``.
        """
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
        )

        # Add time for crushers around 180 pulse if they are included
        if self.rf180_spoiler:
            time_used += 2 * pp.calc_duration(self.spoiler_z)

        # Add time for end spoilers if they are included
        if self.end_spoilers:
            time_used += pp.calc_duration(self.spoiler_x, self.spoiler_y, self.spoiler_z)

        delayTR_exact = self.TR - time_used
        self.delayTR = align2rastertime_nearest(delayTR_exact, self.system.grad_raster_time)
        self.logger.info(f"delayTR_exact: {delayTR_exact*1e3:.2f} ms -> delayTR: {self.delayTR*1e3:.2f} ms")
        self.logger.info(f"Time used in TR (without delayTR): {time_used*1e3:.2f} ms of {self.TR*1e3:.2f} ms")

    # =========================================================================
    # Navigator
    # =========================================================================

    def _add_navigator_acquisition(self, seq: pp.Sequence):
        """
        Add a 3-line navigator acquisition without phase encodings for EPI phase calibration.

        Uses simplified timing (no diffusion gradients). Acquires 3 readout lines with
        alternating gx polarity but no gy blips or gy prephaser, providing the odd/even
        line phase difference needed for N/2 ghost correction.

        The center of the second readout line (line 1, negative polarity) is timed to
        coincide with TE from the 90 RF center. Timing is calculated from first principles.
        """
        # Calculate delay to center echo at TE
        # Time from RF90 center to RF180 center should be TE/2
        delay_to_180 = self.TE / 2 - self.time_after_90 - pp.calc_duration(self.gz90_reph) - self.rf180_center_with_delay
        delay_to_180 = align2rastertime_nearest(delay_to_180, self.system.grad_raster_time)
        # Calculate delay from RF180 center to echo center
        delay_to_echo = (
            self.TE / 2
            - self.time_after_180
            - self.prephaser_duration
            - pp.calc_duration(self.epi.gx, self.epi.adc)
            - self.gx.delay
            - self.epi.gx_.rise_time
            - (self.epi.gx_.flat_time / 2)  # Center of readout should be at echo
        )
        delay_to_echo = align2rastertime_nearest(delay_to_echo, self.system.grad_raster_time)
        self.delay_to_echo_nav = delay_to_echo  # Store for logging

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
                # pp.make_label(label="NAV", type="SET", value=1),
                pp.make_label(label="LIN", type="SET", value=0),
                pp.make_label(label="REV", type="SET", value=0),
                self.epi.gx,
                self.epi.adc,
            )

            seq.add_block(
                # pp.make_label(label="NAV", type="SET", value=1),
                pp.make_label(label="LIN", type="SET", value=1),
                pp.make_label(label="REV", type="SET", value=1),
                self.epi.gx_,
                self.epi.adc,
            )

            seq.add_block(
                # pp.make_label(label="NAV", type="SET", value=1),
                pp.make_label(label="LIN", type="SET", value=2),
                pp.make_label(label="REV", type="SET", value=0),
                self.epi.gx,
                self.epi.adc,
            )
            seq.add_block(pp.make_delay(delayTR))

        else:
            # Create navigator sequence without labels
            seq.add_block(self.rf90, self.gz90)
            seq.add_block(self.gz90_reph)
            seq.add_block(pp.make_delay(delay_to_180))
            seq.add_block(self.rf180, self.gz180)
            seq.add_block(pp.make_delay(delay_to_echo))
            seq.add_block(self.gx_pre)  # Add prephaser
            seq.add_block(self.epi.gx, self.epi.adc)
            seq.add_block(self.epi.gx_, self.epi.adc)
            seq.add_block(self.epi.gx, self.epi.adc)
            seq.add_block(pp.make_delay(delayTR))  # Add delay to complete TR



    # =========================================================================
    # Sequence build (single SE)
    # =========================================================================

    def build_seq(self, old_seq=None):
        """
        Assemble the full Pulseq sequence and store it in ``self.seq``.

        Iterates over every diffusion direction, computes per-axis trapezoid parameters
        (amplitude × unit-vector, with individual rise times so weaker axes ramp faster),
        and appends blocks in the order: RF90 → diff_grad_1 → RF180 → diff_grad_2 →
        EPI prephase → EPI readout → [end spoilers] → TR delay.

        An optional navigator (3-line spin echo without phase encoding) is prepended
        when ``calibration_readout=True``.

        Args:
            old_seq: If provided, blocks are appended to this existing sequence object
                instead of creating a new one (useful for multi-contrast concatenation).

        Returns:
            The assembled ``pp.Sequence`` object (also stored as ``self.seq``).
        """
        if old_seq is None:
            seq = pp.Sequence(self.system)
        else:
            seq = old_seq

        # Add navigator at the start (first acquisition for calibration)
        if self.calibration_readout:
            self._add_navigator_acquisition(seq)

        for i, dir in enumerate(self.b_directions):
            if self.labeled:
                seq.add_block(pp.make_label(label="REP", type="SET", value=i))

            self.logger.info(f"Generating sequence for b-direction: {dir}, b-value: {self.b_value}")

            # Generate diffusion gradients for this direction with individual rise times
            diffusion_gradients = self.diffusion_gradient_amplitude * dir
            self.logger.info(f"Diffusion gradient amplitudes (Hz/m): {diffusion_gradients} for b-value: {self.b_value} direction: {dir}")

            # Per-axis rise times: weaker axes (small |amplitude|) ramp up faster than
            # the worst-case rise time used during fitting, keeping total duration fixed
            # while maximising slew efficiency on each channel independently.
            gx_rise_time = (
                align2rastertime_ceil(abs(diffusion_gradients[0]) / self.system.max_slew, self.system.grad_raster_time)
                if abs(diffusion_gradients[0]) > 0
                else self.system.grad_raster_time
            )

            gy_rise_time = (
                align2rastertime_ceil(abs(diffusion_gradients[1]) / self.system.max_slew, self.system.grad_raster_time)
                if abs(diffusion_gradients[1]) > 0
                else self.system.grad_raster_time
            )

            gz_rise_time = (
                align2rastertime_ceil(abs(diffusion_gradients[2]) / self.system.max_slew, self.system.grad_raster_time)
                if abs(diffusion_gradients[2]) > 0
                else self.system.grad_raster_time
            )

            # Calculate individual flat times based on individual rise times.
            # Use diffusion_gradient_total_duration (== small_delta normally, but
            # larger when amplitude was clamped to max_grad via area-preserving fallback).
            gx_flat_time = self.diffusion_gradient_total_duration - 2 * gx_rise_time
            gy_flat_time = self.diffusion_gradient_total_duration - 2 * gy_rise_time
            gz_flat_time = self.diffusion_gradient_total_duration - 2 * gz_rise_time

            self.logger.info(f"Individual rise times (ms): gx={gx_rise_time*1e3:.3f}, gy={gy_rise_time*1e3:.3f}, gz={gz_rise_time*1e3:.3f}")
            self.logger.info(f"Individual flat times (ms): gx={gx_flat_time*1e3:.3f}, gy={gy_flat_time*1e3:.3f}, gz={gz_flat_time*1e3:.3f}")

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
            # 180 RF Pulse (with optional spoilers)
            # ================================================================
            if self.rf180_spoiler:
                seq.add_block(self.spoiler_z)

            seq.add_block(self.rf180, self.gz180)

            if self.rf180_spoiler:
                seq.add_block(self.spoiler_z)

            # ================================================================
            # Second diffusion gradient (starts right after RF180/spoiler)
            # ================================================================
            seq.add_block(gx_diff, gy_diff, gz_diff)

            # ================================================================
            # Delay before EPI readout (to achieve TE)
            # ================================================================
            if self.delayTE2 > 0:
                seq.add_block(pp.make_delay(self.delayTE2))

            # ================================================================
            # EPI Prephasing gradients
            # ================================================================
            seq.add_block(self.gx_pre, self.gy_pre)

            # ================================================================
            # EPI Readout train
            # ================================================================
            if self.labeled:
                self.epi.add_to_sequence(seq)
            else:
                self.epi.add_to_sequence_unlabeled(seq)

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
        """Log initialization message for the sequence."""
        self.logger.info(f"Initializing EPI Diffusion Spin Echo Pulseq Sequence: {self.name}")

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
        filename += f"_TE{self.TE*1000:.0f}ms"
        # Diffusion suffix
        filename += f"_b{self.b_value}_dirs{self.b_dirs}{f'_b0s{self.b_0_frequency}' if self.b_0_frequency else ''}_delta{self.small_delta*1e3:.2f}ms_DELTA{self.big_DELTA*1e3:.2f}ms"
        # EPI suffix
        filename += f"_pff{self.partial_fourier_factor*100:.0f}_acc{self.epi_acceleration_factor}{'_rs' if self.ramp_sampling == 'ramps_sampling' else f''}"
        filename += ".seq"

        if full_path:
            return os.path.join(self.save_dir, filename)
        return filename
    
    def write(self, filename=None, overwrite=True):
        """Write sequence to file, with logging."""
        if filename is None:
            filename = self.get_save_filename(full_path=True)
        else:
            filename = os.path.join(self.save_dir, filename)

        # Set metadata definitions in the sequence object for export
        self.seq.set_definition("FOV", [self.fov, self.fov, self.slice_thickness])
        self.seq.set_definition("Name", "epi_diffusion_se_" + self.name)
        self.seq.set_definition("TE", self.TE)
        self.seq.set_definition("TR", self.TR)
        self.seq.set_definition("SliceThickness", self.slice_thickness)
        self.seq.set_definition("NNavigatorLines", 3)
        self.seq.set_definition("DiffusionDirections", self.b_directions.tolist())  
        self.seq.set_definition("bValue", self.b_value)
        self.seq.set_definition("b0Frequency", self.b_0_frequency)
        self.seq.set_definition("SmallDelta", self.small_delta)
        self.seq.set_definition("BigDelta", self.big_DELTA)
        self.seq.set_definition("AdcNumSamples", self.adc.num_samples)
        self.seq.set_definition("AdcDwellTime", self.adc.dwell)
        self.seq.set_definition("AccelerationFactor", self.epi_acceleration_factor)
        self.seq.set_definition("PartialFourierFactor", self.partial_fourier_factor)
        
        self.logger.info(f"Writing sequence to file: {filename}")
        
        if os.path.exists(filename):
            if overwrite:
                self.logger.warning(f"File {filename} already exists and overwrite=True. Overwriting.")
                self.seq.write(filename, v141_compat=self.v141_compat)
            else:
                self.logger.warning(f"File {filename} already exists and overwrite=False. Skipping write.")
                return
        else:
            self.seq.write(filename, v141_compat=self.v141_compat)

    def validate_sequence_properties(self, expected_values: dict = None, tolerance: float = None) -> tuple[bool, list[str]]:
        """Flattened validation from PulseqSeq + DiffusionSE."""
        # Run base PulseqSeq validation
        all_passed, failed_tests = super().validate_sequence_properties(expected_values, tolerance)

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
                self.logger.info(f"b-value validation: requested={self.b_value:.2f}, calculated={b_calc:.2f} s/mm^2")

        # Validate diffusion timing
        if self.small_delta > 0 and self.big_DELTA > 0:
            if self.big_DELTA <= self.small_delta:
                msg = f"Diffusion timing: big_DELTA ({self.big_DELTA*1e3:.2f} ms) must be > small_delta ({self.small_delta*1e3:.2f} ms)"
                self.logger.error(msg)
                failed_tests.append(msg)
                all_passed = False
            else:
                self.logger.info(f"Diffusion timing: small_delta={self.small_delta*1e3:.2f} ms, big_DELTA={self.big_DELTA*1e3:.2f} ms")

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


# %%
if __name__ == "__main__":
    acceleration_factor = 1
    pff = 0.75
    res = 2.33333333333333
    dwell = 5 * 0.000001

    epi = EPIDiffusionSEPulseqSeq(
        name=f"DiffSE",
        fov=224e-3,  # in meters
        Nx=96,
        Ny=96,
        resolution=res,  # in mm
        slice_thickness=res * 1e-3,  # in meters
        partial_fourier_factor=0.75,
        TR=5000,  # in milliseconds
        TE=75,  # in milliseconds
        rf90_duration=0.003,
        rf180_duration=0.003,
        dwell_time=dwell,
        prephaser_duration=0.0005,
        rephasers=True,
        simultan_rephasers=False,
        system_type=SystemLimitType.EXTREME,
        rf180_spoiler=True,
        ramp_sampling="ramp_sampled",
        spoiler_amplitude=0.9,
        b_0_frequency=3,
        b_directions=3,
        b_value=500,  # in s/mm^2
        small_delta=0.018,
        big_DELTA=0.035,
        acceleration_factor=1,
        v141_compat=True,
        fit_epi=False,
        labeled=True,
        blip_down=False,
        calibration_readout=True,
    )

    # epi.plot()
    epi.plot_kspace_traj()
    epi.validate_sequence_properties()
    # epi.report()
    epi.write()

# %%
