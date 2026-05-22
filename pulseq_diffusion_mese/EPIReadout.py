# %%
"""
EPIReadout - Standalone EPI readout block with ramp sampling and pypulseq label support.

Encapsulates the design of EPI readout gradients, ADC events, prephasing/rephasing
gradients, and k-space trajectory for echo-planar imaging.  Supports partial Fourier,
GRAPPA/SENSE acceleration, and ramp-sampled (oversampled) acquisition.

Labels attached to each ADC block:
    LIN : absolute k-space line index  (ky_indices[i] + Ny // 2)
    REV : reversed-readout flag        (1 for odd-numbered lines, 0 for even)
    NAV : navigator flag               (always 0 for imaging lines)

Author      : Aron Gimesi <aron.gimesi@tecnico.ulisboa.pt>
Affiliation : Instituto Superior Técnico | MSCA-DN IQ-BRAIN
Date        : 2026
Context     : ESMRMB 2026 - Pulseq DiffusionMESE showcase

Funding acknowledgement (mandatory):
    IQ-BRAIN is funded by the European Union (MSCA Doctoral Network,
    December 2024–November 2028, Grant Agreement No. 101169519).
"""

import numpy as np
import pypulseq as pp
from pypulseq import make_label
import logging
from typing import Tuple


def _align2rastertime_ceil(x, rt):
    """Round ``x`` up to the next integer multiple of raster time ``rt``."""
    return np.ceil(x / rt) * rt


def _align2rastertime_floor(x, rt):
    """Round ``x`` down to the previous integer multiple of raster time ``rt``."""
    return np.floor(x / rt) * rt


def _align2rastertime_nearest(x, rt):
    """Round ``x`` to the nearest integer multiple of raster time ``rt``."""
    return np.round(x / rt) * rt


class EPIReadout:
    """Standalone EPI readout that designs optimal gradients internally and labels every ADC block.

    Handles the full readout train: ramp-sampled readout gradient (gx), phase-encoding
    blips (gy), prephasing/rephasing gradients, and k-space trajectory bookkeeping.
    Designed for direct use inside a parent :class:`PulseqSeq` subclass — the parent
    constructs one :class:`EPIReadout` instance and calls :meth:`add_to_sequence` each TR.

    Attributes:
        fov (float): Field of view in metres.
        Nx (int): Readout matrix size (number of ADC samples per line, before ramp oversampling).
        Ny (int): Phase-encoding matrix size (full k-space height).
        Ny_eff (int): Number of lines actually acquired (accounting for partial Fourier / acceleration).
        Ny_pre (int): Number of lines acquired before the ky=0 echo line.
        Ny_post (int): Number of lines acquired from ky=0 onwards.
        ky_indices (np.ndarray): Ordered ky line indices (relative to DC, i.e. ky=0 at echo).
        echo_line_index (int): Position of the ky=0 line within ``ky_indices``.
        delta_k (float): k-space sampling interval = 1/FOV [m⁻¹].
        k_width (float): Total k-space extent covered per readout line = Nx × delta_k [m⁻¹].
        gx (pp.Trapezoid): Positive-polarity readout gradient.
        gx_ (pp.Trapezoid): Negative-polarity readout gradient (odd lines).
        gy (pp.Trapezoid): Phase-encoding blip gradient.
        gy_blip_rise (pp.Trapezoid): First half of the blip (overlaps trailing ramp of gx).
        gy_blip_fall (pp.Trapezoid): Second half of the blip (overlaps leading ramp of next gx).
        gy_composite (pp.Trapezoid): Merged blip for interior lines (fall half + rise half).
        gx_prephaser (pp.Trapezoid): x-axis prephaser (winds to kx start).
        gy_prephaser (pp.Trapezoid): y-axis prephaser (winds to ky start).
        adc (pp.ADC): ADC event (possibly ramp-sampled with more samples than Nx).
        duration (float): Total readout duration in seconds (prephaser excluded).
        time_until_echo (float): Time from start of readout train to ky=0 line centre.
        time_after_echo (float): Remaining readout time after the echo line.
        line_duration (float): Duration of one readout line (gx block).
    """

    def __init__(
        self,
        fov: float,
        Nx: int,
        Ny: int,
        dwell_time: float,
        system: pp.Opts,
        partial_fourier_factor: float = 1.0,
        blip_down: bool = True,
        acceleration_factor: int = 1,
        ramp_sampling: str = "ramp_sampled",  # 'none', 'optimized', 'ramp_sampled'
        prephaser_duration: float = None,
        rephasers: bool = False,
        simultan_rephasers: bool = True,
        max_duration: float = None,
        logger: logging.Logger = None,
        verbose: bool = False,
        adc_dead_time_correction: bool = True,
    ):
        """Design and store all gradient/ADC events for a complete EPI readout train.

        Args:
            fov: Field of view in metres.
            Nx: Number of readout samples per line (before ramp oversampling).
            Ny: Full phase-encoding matrix size.
            dwell_time: ADC dwell time in seconds.
            system: Scanner hardware limits (:class:`pp.Opts`).
            partial_fourier_factor: Fraction of ky-space to acquire (0.5–1.0); values < 1
                shift the echo line off-centre, acquiring more lines before the echo.
            blip_down: If ``True`` the first phase-encoding blip steps in the negative ky
                direction (standard convention); ``False`` inverts the blip polarity.
            acceleration_factor: In-plane acceleration factor R (1 = fully sampled).
                Every R-th line is acquired; the blip area is scaled accordingly.
            ramp_sampling: ADC strategy — only ``'ramp_sampled'`` is currently supported.
                Samples are collected over the entire trapezoid including ramps, then
                regridded during reconstruction.
            prephaser_duration: Fixed duration for prephasing gradients in seconds.
                ``None`` lets pypulseq choose the shortest feasible trapezoid.
            rephasers: If ``True``, append gradient blocks after the readout train to
                return the magnetisation to k-space centre (useful for multi-echo sequences).
            simultan_rephasers: If ``True``, x and y rephasers are played simultaneously;
                otherwise they are played sequentially.
            max_duration: Hard upper limit on total readout duration in seconds.
                Lines are trimmed symmetrically until the budget is met.
            logger: External :class:`logging.Logger`; a default one is created if ``None``.
            verbose: Print detailed per-step gradient design information to the logger.
            adc_dead_time_correction: Account for the ADC dead-time delay in timing
                calculations (disable only for debugging).
        """
        if logger is None:
            logging.basicConfig(format="%(message)s")
            logger = logging.getLogger(__name__)

        assert acceleration_factor >= 1, "Acceleration factor must be >= 1"
        assert (
            0.5 <= partial_fourier_factor <= 1.0
        ), "Partial Fourier must be in [0.5, 1.0]"
        assert ramp_sampling in [
            "none",
            "optimized",
            "ramp_sampled",
        ], "ramp_sampling must be 'none', 'optimized', or 'ramp_sampled'"

        self.fov = fov
        self.Nx = Nx
        self.Ny = Ny
        self.dwell_time = dwell_time
        self.system = system
        self.partial_fourier_factor = partial_fourier_factor
        self.blip_down = blip_down
        self.acceleration_factor = acceleration_factor
        self.ramp_sampling = ramp_sampling
        self._user_prephaser_duration = prephaser_duration
        self.prephaser_duration = prephaser_duration
        self.rephasers = rephasers
        self.simultan_rephasers = simultan_rephasers
        self.logger = logger
        self.verbose = verbose
        self.adc_dead_time_correction = adc_dead_time_correction

        self.polarity = 1 if not blip_down else -1

        # Calculate k-space parameters
        self.delta_k = 1 / fov
        self.k_width = Nx * self.delta_k
        self.adc_dead_time = (
            self.system.adc_dead_time if self.adc_dead_time_correction else 0
        )

        # Design gradients based on ramp sampling mode
        if ramp_sampling == "ramp_sampled":
            self._design_ramp_sampled_gradients()
        else:
            raise ValueError(
                f"Invalid ramp_sampling option: {ramp_sampling}. Choose from 'none', 'optimized', or 'ramp_sampled'."
            )

        # Setup k-space trajectory (independent of gradient design)
        self._setup_kspace_trajectory()

        # Create gradient variants and blips
        self._create_gradient_variants()

        # Setup prephasing gradients
        self._setup_prephasing()

        # Calculate timing
        self._calculate_timing()

        # Log configuration
        if self.verbose:
            self._log_parameters()

        # Trim k-space lines to fit within max_duration if specified
        if max_duration is not None and self.duration > max_duration:
            self.fit_to_duration(max_duration)

        # Verify slew rates for ramp-sampled mode
        if ramp_sampling == "ramp_sampled" and verbose:
            self.verify_slew_rates()

    # =========================================================================
    # Gradient design methods
    # =========================================================================
    def _design_ramp_sampled_gradients(self):
        """Design ramp-sampled EPI readout and blip gradients following the pypulseq epi_rs.py pattern.

        The readout gradient (gx) is sized so that ADC sampling spans the full trapezoid
        including both ramps.  The amplitude is then rescaled to deliver exactly k_width
        of k-space area *excluding* the triangular dead zones at the ramp edges that
        overlap with the neighbouring phase-encoding blip.

        Sets the following attributes:
            gx, gx_ : positive / negative readout trapezoids
            gy       : phase-encoding blip trapezoid
            adc      : ADC event (ramp-sampled, num_samples > Nx)
            rs_dwell_time, rs_adc_samples : ramp-sampled ADC parameters
            rs_ramp_samples_rise/fall      : samples collected on each ramp
        """
        # Nominal readout window — used to size gx; actual value is updated below
        self.readout_time = self.Nx * self.dwell_time
        self.logger.info(
            f"[RS] readout_time = {self.readout_time*1e6:.2f} us  (Nx={self.Nx}, dwell={self.dwell_time*1e6:.3f} us)"
        )

        # Minimum blip half-duration at max_slew to encode one ky step (delta_k × R)
        half_blip = _align2rastertime_ceil(
            np.sqrt(self.delta_k * self.acceleration_factor / self.system.max_slew),
            self.system.grad_raster_time,
        )
        blip_duration = 2 * half_blip
        self.blip_duration = blip_duration
        self.min_blip_duration = blip_duration
        self.logger.info(
            f"[RS] blip: half_blip={half_blip*1e6:.2f} us  blip_duration={blip_duration*1e6:.2f} us"
        )

        self.gy = pp.make_trapezoid(
            channel="y",
            system=self.system,
            area=self.delta_k * self.acceleration_factor,
            duration=blip_duration,
        )
        self.blip_area = self.gy.area

        self.logger.info(
            f"[RS] gy: area={self.gy.area:.4f} m^-1  amp={self.gy.amplitude:.1f} T/m  rise={self.gy.rise_time*1e6:.2f} us  flat={self.gy.flat_time*1e6:.2f} us  fall={self.gy.fall_time*1e6:.2f} us"
        )

        # extra_area: k-space area swept during the blip half-duration that overlaps the
        # leading/trailing ramp of gx.  Adding it here keeps the total gx area large enough
        # so that after dead-zone correction the effective area equals exactly k_width.
        extra_area = (blip_duration / 2) ** 2 * self.system.max_slew
        # Align to 2×grad_raster so gx.flat_time/2 (the EPI echo offset) stays on the gradient grid
        double_raster = 2 * self.system.grad_raster_time
        gx_duration = _align2rastertime_ceil(
            self.readout_time + blip_duration, double_raster
        )
        _gx_min_duration = pp.calc_duration(
            pp.make_trapezoid(
                channel="x", system=self.system, area=self.k_width + extra_area
            )
        )
        if gx_duration < _gx_min_duration:
            self.logger.info(
                f"[RS] gx duration too short ({gx_duration*1e6:.2f} us < min {_gx_min_duration*1e6:.2f} us) -- using solver minimum"
            )
            gx_duration = _align2rastertime_ceil(_gx_min_duration, double_raster)
        self.logger.info(
            f"[RS] gx design: extra_area={extra_area:.4f} m^-1  total_area={self.k_width + extra_area:.4f} m^-1  duration={gx_duration*1e6:.2f} us"
        )
        gx = pp.make_trapezoid(
            channel="x",
            system=self.system,
            area=self.k_width + extra_area,
            duration=gx_duration,
        )
        self.logger.info(
            f"[RS] gx (pre-scale): amp={gx.amplitude:.2f} T/m  rise={gx.rise_time*1e6:.2f} us  flat={gx.flat_time*1e6:.2f} us  fall={gx.fall_time*1e6:.2f} us  area={gx.area:.4f} m^-1"
        )

        # Dead-zone correction: the blip overlaps the first/last (blip_duration/2) of each
        # ramp, making those ramp segments unusable for k-space encoding.  Compute the
        # triangular area lost on each ramp and rescale gx.amplitude so the net encoding
        # area equals k_width exactly.
        actual_area = gx.area
        dead_rise = gx.amplitude / gx.rise_time * (blip_duration / 2) ** 2 / 2
        dead_fall = gx.amplitude / gx.fall_time * (blip_duration / 2) ** 2 / 2
        actual_area -= dead_rise
        actual_area -= dead_fall
        self.logger.info(
            f"[RS] dead zones: rise={dead_rise:.4f} m^-1  fall={dead_fall:.4f} m^-1  effective_area={actual_area:.4f} m^-1  (target={self.k_width:.4f} m^-1)"
        )
        gx.amplitude = gx.amplitude / actual_area * self.k_width
        gx.area = gx.amplitude * (gx.flat_time + gx.rise_time / 2 + gx.fall_time / 2)
        gx.flat_area = gx.amplitude * gx.flat_time
        self.gx = gx
        self.logger.info(
            f"[RS] gx (post-scale): amp={gx.amplitude:.2f} T/m  area={gx.area:.4f} m^-1"
        )

        # Nyquist dwell time for the rescaled gradient amplitude, then floor to ADC raster
        adc_dwell_nyquist = self.delta_k / self.gx.amplitude
        rs_dwell = _align2rastertime_floor(
            adc_dwell_nyquist, self.system.adc_raster_time
        )
        self.logger.info(
            f"[RS] ADC dwell: nyquist={adc_dwell_nyquist*1e6:.4f} us  rs_dwell={rs_dwell*1e6:.4f} us"
        )

        # Round sample count to the nearest multiple of 4 (required by most scanner ADC hardware)
        adc_samples = int(np.floor(self.readout_time / rs_dwell / 4)) * 4
        self.logger.info(
            f"[RS] ADC samples: {adc_samples}  (Nx={self.Nx}, factor={adc_samples/self.Nx:.3f}x)"
        )

        # Centre the ADC window on the k-space echo (gx flat-time midpoint)
        time_to_center = rs_dwell * ((adc_samples - 1) / 2 + 0.5)
        adc_delay = _align2rastertime_nearest(
            self.gx.rise_time + self.gx.flat_time / 2 - time_to_center,
            self.system.adc_raster_time,
        )
        self.logger.info(
            f"[RS] ADC delay: time_to_center={time_to_center*1e6:.2f} us  adc_delay={adc_delay*1e6:.2f} us  (gx.rise={self.gx.rise_time*1e6:.2f} us)"
        )
        self.adc = pp.make_adc(
            num_samples=adc_samples,
            dwell=rs_dwell,
            delay=adc_delay,
        )

        # Update readout_time to reflect actual samples × dwell (may differ from Nx × dwell_time)
        self.readout_time = adc_samples * rs_dwell

        self.rs_dwell_time = rs_dwell
        self.rs_adc_samples = adc_samples

        self.gx_raster_difference = 0
        self.min_gx_ramp_time = self.gx.rise_time + self.gx.fall_time

        # Count how many ADC samples fall on each ramp — needed for ramp-correction during recon
        sample_times = adc_delay + np.arange(adc_samples) * rs_dwell
        self.rs_ramp_samples_rise = int(np.sum(sample_times < self.gx.rise_time))
        self.rs_ramp_samples_fall = int(
            np.sum(sample_times >= self.gx.rise_time + self.gx.flat_time)
        )
        self.logger.info(
            f"[RS] ramp samples: rise={self.rs_ramp_samples_rise}  flat={adc_samples - self.rs_ramp_samples_rise - self.rs_ramp_samples_fall}  fall={self.rs_ramp_samples_fall}"
        )

    # =========================================================================
    # K-space trajectory
    # =========================================================================

    def _setup_kspace_trajectory(self):
        """Calculate which k-space lines to acquire, accounting for partial Fourier and acceleration.

        ky indices are defined relative to the echo line (ky=0 at DC).  Partial Fourier
        shifts the acquired range asymmetrically: the post-echo half is always Ny/2 lines,
        while the pre-echo half is extended by the partial-Fourier factor so the total
        acquired lines equal partial_fourier_factor × Ny.

        ``blip_down=True`` means the readout starts at the most negative ky and steps
        upward (standard convention); ``blip_down=False`` starts at positive ky and steps
        downward (used for blip-up/down B0 distortion correction pairs).

        Sets attributes: ky_indices, echo_line_index, Ny_pre, Ny_post, Ny_eff, ky_start.
        """
        # Post-echo half: always symmetric (Ny/2 lines from DC upward)
        Ny_post_requested = int(np.round(self.Ny / 2))
        # Pre-echo half: extended by partial Fourier so total = round(PF × Ny)
        Ny_pre_requested = (
            int(np.round(self.partial_fourier_factor * self.Ny)) - Ny_post_requested
        )

        if self.blip_down:
            # Start at ky=0 stepping negative (pre-echo), then positive (post-echo)
            ky_negative = np.arange(0, -Ny_pre_requested - 1, -self.acceleration_factor)
            ky_positive = np.arange(
                self.acceleration_factor, Ny_post_requested, self.acceleration_factor
            )
        else:
            # Mirror of the blip_down=True coverage about ky=0: PF short side on +ky.
            ky_positive = np.arange(
                0, Ny_pre_requested + 1, self.acceleration_factor
            )
            ky_negative = np.arange(
                -self.acceleration_factor,
                -Ny_post_requested,
                -self.acceleration_factor,
            )

        # Sort ascending then reverse for blip-up so acquisition order matches traversal direction
        self.ky_indices = np.sort(np.unique(np.concatenate([ky_negative, ky_positive])))

        if not self.blip_down:
            self.ky_indices = self.ky_indices[::-1]

        # Locate the echo line (ky=0) within the ordered acquisition array
        self.echo_line_index = np.where(self.ky_indices == 0)[0][0]

        self.Ny_pre = self.echo_line_index
        self.Ny_post = len(self.ky_indices) - self.echo_line_index
        self.Ny_eff = len(self.ky_indices)
        self.ky_start = self.ky_indices[0]

    # =========================================================================
    # Gradient variants
    # =========================================================================

    def _create_gradient_variants(self):
        """Create all gradient variants needed to drive the EPI readout train.

        EPI alternates readout polarity every line (even lines read left→right, odd lines
        right→left).  The phase-encoding blip straddles the boundary between adjacent lines:
        the fall half of the blip overlaps the trailing ramp of the current gx, and the rise
        half overlaps the leading ramp of the next gx.  This method produces:

        * ``gx`` / ``gx_``  — positive/negative readout trapezoids (same shape, opposite sign)
        * ``gy_blip_rise`` / ``gy_blip_fall`` — split half-blips for edge lines
        * ``gy_composite`` — merged (fall + rise) blip for interior lines, played simultaneously
          with gx so no dead time is introduced between readout lines
        """
        if self.blip_down:
            half_blip_raster = _align2rastertime_ceil(
                self.blip_duration / 2, self.system.grad_raster_time
            )
            blip_duration_raster = half_blip_raster * 2
            self.gy = pp.make_trapezoid(
                channel="y",
                system=self.system,
                area=self.polarity * self.delta_k * self.acceleration_factor,
                duration=blip_duration_raster,
            )
            self.blip_duration = blip_duration_raster
        else:
            half_blip_raster = _align2rastertime_ceil(
                self.blip_duration / 2, self.system.grad_raster_time
            )
            if self.blip_duration != half_blip_raster * 2:
                blip_duration_raster = half_blip_raster * 2
                self.gy = pp.make_trapezoid(
                    channel="y",
                    system=self.system,
                    area=self.polarity * self.delta_k * self.acceleration_factor,
                    duration=blip_duration_raster,
                )
                self.blip_duration = blip_duration_raster

        if self.ramp_sampling == "optimized":
            half_blip = self.blip_duration / 2
            current_ramp_time = self.gx.rise_time + self.gx.fall_time

            if not np.isclose(self.blip_duration, current_ramp_time, rtol=1e-6):
                self.gx = pp.make_trapezoid(
                    channel="x",
                    system=self.system,
                    flat_area=self.k_width,
                    flat_time=self.gx.flat_time,
                    rise_time=half_blip,
                    fall_time=half_blip,
                    delay=self.gx.delay,
                )
                self.adc.delay = self.gx.delay + self.gx.rise_time

        # Split the blip at its midpoint so each half can be overlapped with the adjacent gx ramp
        split_time = _align2rastertime_ceil(
            self.blip_duration / 2, self.system.grad_raster_time
        )
        self.gy_blip_rise, self.gy_blip_fall = pp.split_gradient_at(self.gy, split_time)

        if self.ramp_sampling in ["none"]:
            # Non-ramp-sampled: gx starts after the blip fall half; blip rise starts after gx ends
            half_blip = _align2rastertime_ceil(
                self.blip_duration / 2, self.system.grad_raster_time
            )
            self.gx.delay = half_blip
            self.adc.delay = self.gx.delay + self.gx.rise_time
            self.gy_blip_fall.delay = 0
            self.gy_blip_rise.delay = pp.calc_duration(self.gx)
        else:
            # Ramp-sampled: blip rise overlaps the trailing ramp of gx (no dead time between lines)
            self.gy_blip_rise.delay = pp.calc_duration(self.gx) - split_time
            self.gy_blip_fall.delay = 0

        # Composite blip for interior lines: fall half of previous blip + rise half of next blip,
        # merged into a single gradient event played simultaneously with gx
        self.gy_composite = pp.add_gradients(
            [self.gy_blip_fall, self.gy_blip_rise], system=self.system
        )

        # Negative-polarity readout for odd lines (EPI zigzag)
        self.gx_ = pp.make_trapezoid(
            channel="x",
            system=self.system,
            amplitude=-self.gx.amplitude,
            rise_time=self.gx.rise_time,
            flat_time=self.gx.flat_time,
            fall_time=self.gx.fall_time,
            delay=self.gx.delay,
        )

        # Sanity check: gx and gy_composite must be the same duration so they can be played together
        if self.ramp_sampling not in ["none"]:
            if self.ramp_sampling == "ramp_sampled":
                assert (
                    abs(pp.calc_duration(self.gy_composite) - pp.calc_duration(self.gx))
                    <= self.system.grad_raster_time
                ), f"Timing mismatch: gx {pp.calc_duration(self.gx)} vs gy_composite {pp.calc_duration(self.gy_composite)}"
            else:
                assert np.isclose(
                    pp.calc_duration(self.gx),
                    pp.calc_duration(self.gy_composite),
                    rtol=1e-6,
                ), f"Timing mismatch: gx {pp.calc_duration(self.gx)} vs gy_composite {pp.calc_duration(self.gy_composite)}"

    # =========================================================================
    # Prephasing / rephasing
    # =========================================================================

    def _setup_prephasing(self):
        """Build prephasing (and optionally rephasing) gradients for the EPI readout.

        The x-axis prephaser winds the magnetisation to the kx starting position of the
        first readout line (negative half of k-space for blip_down).  Its area equals the
        k-space distance from DC to the centre of the ADC window of the first readout line.

        The y-axis prephaser steps to the starting ky line (``ky_start``) by accumulating
        ``Ny_pre`` blip steps in the direction opposite to the readout blips.

        The optional rephasers (used in multi-echo sequences) return magnetisation to
        k-space centre after the last readout line so that the next echo starts at DC.
        """
        # kx area accumulated from the start of gx up to the ADC centre of the first line
        ramp_area = self.gx.amplitude * self.gx.rise_time / 2
        flat_area_to_adc_center = self.gx.amplitude * (
            self.adc.delay + self.readout_time / 2 - self.gx.delay - self.gx.rise_time
        )
        gx_area_to_echo = ramp_area + flat_area_to_adc_center

        if self.prephaser_duration is None:
            gx_pre_temp = pp.make_trapezoid(
                channel="x",
                system=self.system,
                area=-gx_area_to_echo,
            )
            duration = pp.calc_duration(gx_pre_temp)
        else:
            duration = self.prephaser_duration

        # x-prephaser: negative area pre-winds to the kx start of the first readout line
        self.gx_prephaser = pp.make_trapezoid(
            channel="x",
            system=self.system,
            area=-gx_area_to_echo,
            duration=duration,
        )

        # y-prephaser: steps Ny_pre blip increments in the reverse blip direction to reach ky_start
        self.gy_prephaser = pp.make_trapezoid(
            channel="y",
            system=self.system,
            area=self.Ny_pre * -self.polarity * self.blip_area,
            duration=duration,
        )

        # Left-align both prephasers so they end at the same time
        self.gx_prephaser, self.gy_prephaser = pp.align(
            left=self.gx_prephaser,
            right=self.gy_prephaser,
        )

        self.prephaser_duration = pp.calc_duration(self.gx_prephaser, self.gy_prephaser)

        if self.rephasers:
            # The last line's polarity determines the sign of the x-rephaser
            last_line_index = self.Ny_eff - 1
            last_line_is_negative = last_line_index % 2 == 1

            # x-rephaser: returns to kx=0 from the end of the last readout line
            self.gx_rephaser = pp.make_trapezoid(
                channel="x",
                system=self.system,
                area=self.gx.area / 2 if last_line_is_negative else -self.gx.area / 2,
            )

            # y-rephaser: returns to ky=0 by reversing the (Ny_post - 1) blip steps after the echo
            self.gy_rephaser = pp.make_trapezoid(
                channel="y",
                system=self.system,
                area=-self.polarity * self.blip_area * (self.Ny_post - 1),
            )

            if self.simultan_rephasers:
                self.gx_rephaser, self.gy_rephaser = pp.align(
                    left=self.gx_rephaser,
                    right=self.gy_rephaser,
                )

            if self.simultan_rephasers:
                self.rephaser_duration = pp.calc_duration(
                    self.gx_rephaser, self.gy_rephaser
                )
            else:
                self.rephaser_duration = pp.calc_duration(
                    self.gx_rephaser
                ) + pp.calc_duration(self.gy_rephaser)

    # =========================================================================
    # Timing
    # =========================================================================

    def _calculate_timing(self):
        """Derive total readout duration, line duration, and echo timing offsets.

        In ramp-sampled mode each line occupies exactly ``pp.calc_duration(gx)`` because
        the blip is overlapped with the ramp.  In non-ramp-sampled mode the blip is played
        sequentially, so ``line_duration = gx_duration + blip_duration``; the last line has
        no trailing blip, hence the total is shortened by ``half_blip``.
        """
        self.gx_duration = self.gx.rise_time + self.gx.flat_time + self.gx.fall_time
        self.gx_flat_time = self.gx.flat_time

        if self.ramp_sampling == "none":
            # Blip is sequential: each line occupies gx + full blip duration
            line_duration = self.gx_duration + self.blip_duration
        else:
            # Blip overlaps gx ramp; line duration is just the gx block duration
            line_duration = pp.calc_duration(self.gx)

        self.line_duration = line_duration

        # Time from readout start to the ADC centre of the echo line (ky=0)
        self.time_until_echo = (
            self.Ny_pre * line_duration + self.adc.delay + self.readout_time / 2
        )

        if self.ramp_sampling == "none":
            # Last line has no trailing half-blip, so subtract half_blip from the total
            half_blip = self.blip_duration / 2
            self.duration = self.Ny_eff * line_duration - half_blip
        else:
            self.duration = self.Ny_eff * line_duration

        if self.rephasers:
            self.duration += self.rephaser_duration

        self.time_after_echo = self.duration - self.time_until_echo

    # =========================================================================
    # Duration trimming
    # =========================================================================

    def fit_to_duration(self, max_duration: float):
        """Trim acquired k-space lines symmetrically until the readout fits within ``max_duration``.

        Lines are removed alternately from the pre-echo side (if Ny_pre > Ny_post // 2)
        and the post-echo side to preserve image quality while shortening the echo train.
        After each removal, the prephaser, rephasers (if enabled), and timing are
        recomputed.

        Args:
            max_duration: Hard upper limit on total readout duration in seconds.

        Raises:
            ValueError: If even a single-line readout exceeds ``max_duration``.
        """
        if self.duration <= max_duration:
            return

        initial_Ny_eff = self.Ny_eff
        lines_removed = 0
        trim_post_next = False

        while self.duration > max_duration:
            if self.Ny_eff <= 1:
                raise ValueError(
                    f"Cannot fit EPI readout within max_duration={max_duration*1e3:.3f} ms: "
                    f"even a single line takes {self.duration*1e3:.3f} ms."
                )

            if self.Ny_pre >= self.Ny_post // 2:
                self.ky_indices = self.ky_indices[1:]
            elif self.Ny_post > self.Ny_pre:
                self.ky_indices = self.ky_indices[:-1]
            else:
                if trim_post_next:
                    self.ky_indices = self.ky_indices[:-1]
                    trim_post_next = False
                else:
                    self.ky_indices = self.ky_indices[1:]
                    trim_post_next = True

            self.echo_line_index = np.where(self.ky_indices == 0)[0][0]
            self.Ny_pre = self.echo_line_index
            self.Ny_post = len(self.ky_indices) - self.echo_line_index
            self.Ny_eff = len(self.ky_indices)
            self.ky_start = self.ky_indices[0]
            lines_removed += 1

            self._update_prephaser()

            if self.rephasers:
                self._update_rephasers()

            self._calculate_timing()

        if self.verbose:
            self.logger.info(
                f"fit_to_duration: removed {lines_removed} line(s) "
                f"(from {initial_Ny_eff} -> {self.Ny_eff}); "
                f"final duration = {self.duration*1e3:.3f} ms "
                f"(budget = {max_duration*1e3:.3f} ms)"
            )
            self._log_parameters()

    def _update_prephaser(self):
        """Update prephasing gradients after ky_start changes."""
        duration = (
            self._user_prephaser_duration
            if self._user_prephaser_duration is not None
            else pp.calc_duration(
                pp.make_trapezoid(
                    channel="x",
                    system=self.system,
                    area=-self.gx.area / 2,
                )
            )
        )

        self.gx_prephaser = pp.make_trapezoid(
            channel="x",
            system=self.system,
            area=-self.gx.area / 2,
            duration=duration,
        )

        self.gy_prephaser = pp.make_trapezoid(
            channel="y",
            system=self.system,
            area=self.ky_start * self.gy.area / self.acceleration_factor,
            duration=duration,
        )

        self.gx_prephaser, self.gy_prephaser = pp.align(
            left=self.gx_prephaser,
            right=self.gy_prephaser,
        )

        self.prephaser_duration = pp.calc_duration(self.gx_prephaser, self.gy_prephaser)

    def _update_rephasers(self):
        """Update rephasing gradients after Ny_post / last-line polarity changes."""
        last_line_index = self.Ny_eff - 1
        last_line_is_negative = last_line_index % 2 == 1

        self.gx_rephaser = pp.make_trapezoid(
            channel="x",
            system=self.system,
            area=self.gx.area / 2 if last_line_is_negative else -self.gx.area / 2,
        )

        self.gy_rephaser = pp.make_trapezoid(
            channel="y",
            system=self.system,
            area=-self.gy.area * (self.Ny_post - 1),
        )

        if self.simultan_rephasers:
            self.gx_rephaser, self.gy_rephaser = pp.align(
                left=self.gx_rephaser,
                right=self.gy_rephaser,
            )
            self.rephaser_duration = pp.calc_duration(
                self.gx_rephaser, self.gy_rephaser
            )
        else:
            self.rephaser_duration = pp.calc_duration(
                self.gx_rephaser
            ) + pp.calc_duration(self.gy_rephaser)

    # =========================================================================
    # Readout events
    # =========================================================================

    def get_readout_events(self, line_index: int) -> Tuple:
        """Return the gradient and ADC events for a single k-space line.

        EPI alternates readout polarity every line: even lines use ``gx`` (positive),
        odd lines use ``gx_`` (negative).  The phase-encoding blip used depends on
        position within the echo train:
        - First line  → only the rise half of the blip (``gy_blip_rise``)
        - Last line   → only the fall half of the blip  (``gy_blip_fall``)
        - All others  → the composite blip (``gy_composite``, fall + rise merged)

        For reversed (odd) lines the ADC delay is shifted by ``gx_raster_difference``
        to compensate for any sub-raster-time timing offset between the positive and
        negative readout gradients.

        Args:
            line_index: Zero-based index into the acquired line list (0 to Ny_eff − 1).

        Returns:
            Tuple (gx_event, gy_event, adc_event) ready to be passed to ``seq.add_block()``.
        """
        assert (
            0 <= line_index < self.Ny_eff
        ), f"line_index must be in [0, {self.Ny_eff})"

        # Alternate readout polarity: even→positive, odd→negative
        if line_index % 2 == 0:
            gx = self.gx
        else:
            gx = self.gx_

        # Select the appropriate blip variant based on position in the echo train
        if line_index == 0:
            gy = self.gy_blip_rise  # first line: only the leading half-blip
        elif line_index == self.Ny_eff - 1:
            gy = self.gy_blip_fall  # last line: only the trailing half-blip
        else:
            gy = self.gy_composite  # interior lines: merged fall + rise half-blips

        # Odd lines read in the opposite direction; create a new ADC with the raster offset
        if line_index % 2 == 1:
            adc = pp.make_adc(
                num_samples=self.adc.num_samples,
                system=self.system,
                dwell=self.adc.dwell,
                delay=self.adc.delay + self.gx_raster_difference,
            )
        else:
            adc = self.adc

        return gx, gy, adc

    # =========================================================================
    # Sequence building
    # =========================================================================

    def add_to_sequence(self, seq: pp.Sequence, rep: int = 0, slc: int = 0):
        """Add the full labelled EPI readout train to a pypulseq Sequence object.

        Each ADC block receives three labels used by the scanner's ICE reconstruction:

        * **LIN** — absolute k-space line index (``ky + Ny // 2``), mapping the
          DC-centred relative index to the zero-based scanner line counter.
        * **REV** — reversed-readout flag (1 for odd lines, 0 for even), tells ICE
          to flip the sample order before inserting into the k-space buffer.
        * **NAV** — navigator flag, always 0 here (navigator echoes are separate blocks).

        Args:
            seq: pypulseq :class:`pp.Sequence` object to append blocks to.
            rep: Repetition / diffusion-direction index (not used for labelling here;
                 set by the parent sequence before calling this method).
            slc: Slice index (same note as ``rep``).
        """
        for i, ky in enumerate(self.ky_indices):
            gx, gy, adc = self.get_readout_events(i)

            # Convert relative ky (−Ny/2 … +Ny/2−1) to absolute scanner line index (0 … Ny−1)
            absolute_lin = int(ky + self.Ny // 2)
            is_reversed = int(i % 2 == 1)

            labels = [
                pp.make_label(label="LIN", type="SET", value=absolute_lin),
                pp.make_label(label="REV", type="SET", value=is_reversed),
                pp.make_label(
                    label="NAV", type="SET", value=0
                ),  # 0 = imaging data, not navigator
            ]

            seq.add_block(*labels, gx, gy, adc)

        if self.rephasers:
            if self.simultan_rephasers:
                seq.add_block(self.gx_rephaser, self.gy_rephaser)
            else:
                seq.add_block(self.gx_rephaser)
                seq.add_block(self.gy_rephaser)

    def add_to_sequence_unlabeled(self, seq: pp.Sequence):
        """Add the EPI readout train to a sequence without attaching ICE labels.

        Use this variant for simulation or when the downstream reconstruction does not
        rely on pypulseq labels (e.g. MR-zero / Bloch simulations).

        Args:
            seq: pypulseq :class:`pp.Sequence` object to append blocks to.
        """
        for i in range(self.Ny_eff):
            gx, gy, adc = self.get_readout_events(i)
            seq.add_block(gx, gy, adc)

        if self.rephasers:
            if self.simultan_rephasers:
                seq.add_block(self.gx_rephaser, self.gy_rephaser)
            else:
                seq.add_block(self.gx_rephaser)
                seq.add_block(self.gy_rephaser)

    # =========================================================================
    # Diagnostics
    # =========================================================================

    def _log_parameters(self):
        """Log readout configuration."""
        if self.verbose and self.logger is not None:
            self.logger.info(f"EPI Readout Configuration")
            self.logger.info(f"{'='*60}")
            self.logger.info(f"FOV: {self.fov*1e3:.1f} mm")
            self.logger.info(f"Matrix: {self.Nx} x {self.Ny}")
            self.logger.info(f"Dwell time: {self.dwell_time*1e6:.1f} us")
            self.logger.info(f"Ramp sampling: {self.ramp_sampling}")
            if self.ramp_sampling in ["partial", "full"]:
                self.logger.info(f"  RS dwell time: {self.rs_dwell_time*1e6:.1f} us")
                self.logger.info(
                    f"  RS ADC samples: {self.rs_adc_samples} (vs {self.Nx} Nx)"
                )
            self.logger.info(f"Acceleration: R={self.acceleration_factor}")
            self.logger.info(f"Partial Fourier: {self.partial_fourier_factor}")
            self.logger.info(f"\nAcquisition:")
            self.logger.info(f"  Lines acquired: {self.Ny_eff} (of {self.Ny} total)")
            self.logger.info(f"  Echo at line: {self.echo_line_index + 1} (ky=0)")
            self.logger.info(f"\nTiming:")
            self.logger.info(f"  Line duration: {self.line_duration*1e3:.3f} ms")
            self.logger.info(f"  Time to echo: {self.time_until_echo*1e3:.2f} ms")
            self.logger.info(f"  Time after echo: {self.time_after_echo*1e3:.2f} ms")
            self.logger.info(f"  Total readout: {self.duration*1e3:.2f} ms")
            self.logger.info(f"{'='*60}\n")

    def verify_slew_rates(self):
        """Assert that the peak slew rate of every readout line stays within system limits.

        Reconstructs the gx waveform on the gradient raster and checks that no
        finite-difference step exceeds ``system.max_slew × 1.01`` (1% tolerance for
        floating-point rounding).  Raises ``AssertionError`` on the first violation.

        Called automatically by ``__init__`` when ``ramp_sampling='ramp_sampled'``
        and ``verbose=True``.
        """
        dt = self.system.grad_raster_time

        self.logger.info(f"RS ramp samples rise: {self.rs_ramp_samples_rise}")
        self.logger.info(f"RS ramp samples fall: {self.rs_ramp_samples_fall}")

        for i in range(self.Ny_eff):
            gx, gy, adc = self.get_readout_events(i)

            duration = pp.calc_duration(gx)
            n_points = int(round(duration / dt))
            t = np.arange(n_points) * dt
            t_rel = t - gx.delay

            rise_end = gx.rise_time
            flat_end = gx.rise_time + gx.flat_time
            fall_end = gx.rise_time + gx.flat_time + gx.fall_time

            waveform = np.where(
                t_rel < 0,
                0.0,
                np.where(
                    t_rel < rise_end,
                    gx.amplitude * t_rel / gx.rise_time,
                    np.where(
                        t_rel < flat_end,
                        gx.amplitude,
                        np.where(
                            t_rel < fall_end,
                            gx.amplitude * (fall_end - t_rel) / gx.fall_time,
                            0.0,
                        ),
                    ),
                ),
            )

            slew = np.diff(waveform) / dt
            max_slew = np.max(np.abs(slew))
            assert max_slew <= self.system.max_slew * 1.01, (
                f"Line {i}: max slew {max_slew:.1f} T/m/s exceeds "
                f"limit {self.system.max_slew:.1f} T/m/s"
            )


# %%
