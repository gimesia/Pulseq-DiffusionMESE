"""
SEMultishotPulseqSeq — Fast Spin Echo (FSE/RARE) Cartesian sequence.

Extends the single-echo SE design with an echo train of length ETL: each TR
acquires ETL phase-encoded echoes via ETL successive 180° refocusing pulses,
covering full k-space in Ny/ETL shots (linear, contiguous ordering).

When ETL=1 the sequence reduces to a conventional single-echo SE identical in
timing to SEPulseqSeq (IQ-BRAIN DC3 reference).

Spin-echo symmetry between consecutive echoes is maintained by an
``inter_echo_delay`` computed so that the time from echo_n centre to
RF180_{n+1} centre equals the time from RF180_{n+1} centre to echo_{n+1}
centre (= half-ESP).

Author      : Aron Gimesi <aron.gimesi@tecnico.ulisboa.pt>
Affiliation : Instituto Superior Técnico | MSCA-DN IQ-BRAIN
Date        : 2026
Context     : ESMRMB 2026 — Pulseq DiffusionMESE showcase

Funding acknowledgement (mandatory):
    IQ-BRAIN is funded by the European Union (MSCA Doctoral Network,
    December 2024-November 2028, Grant Agreement No. 101169519).
"""

# %%
import logging
import os
import numpy as np
import pypulseq as pp

from PulseqSeq import *


class SEMultishotPulseqSeq(PulseqSeq):
    """Fast Spin Echo (FSE/RARE) sequence with configurable echo train length.

    Sequence timeline for one shot (ETL echoes)::

        RF90 → gz90_reph → delayTE1 →
          [n = 0 … ETL-1]:
            RF180_n → gx_pre → gy_{shot·ETL+n} → delayTE2 → gx+ADC →
            gy_rewind_{shot·ETL+n} → [inter_echo_delay  if n < ETL-1]
          [end_spoilers] → delayTR

    K-space ordering: linear/contiguous — shot *s* acquires lines
    ``[s·ETL, …, (s+1)·ETL − 1]`` from :attr:`phase_encoding_gradients`.

    Key timing attributes (all in seconds):
        TE (float)              : First echo time (RF90 centre → echo_0 centre).
        ESP (float)             : Echo spacing (= 2 × half-ESP).
        delayTE1 (float)        : Delay between gz90_reph end and RF180_0 start.
        delayTE2 (float)        : Delay between gx_pre/gy end and readout window.
        inter_echo_delay (float): Symmetry gap between gy_rewind end and next RF180.
        delayTR (float)         : Padding at the end of each TR.
    """

    def __init__(
        self,
        name: str,
        fov: float,
        Nx: int,
        Ny: int,
        slice_thickness: float,
        TR: int,
        TE: int,
        ETL: int = 1,
        N_slices: int = 1,
        system_type=SystemLimitType.SAFE,
        rf90_duration: float = 0.003,
        rf180_duration: float = 0,
        resolution: float = None,
        flip_angle: int = 90,
        apodization: float = 0.5,
        time_bw_product: float = 4,
        dwell_time: float = None,
        end_spoilers: bool = False,
        spoiler_amplitude: float = 1,
        spoiler_duration: float = 1e-3,
        save_dir: str = DEFAULT_SAVE_DIR,
        logger: logging.Logger = None,
        v141_compat: bool = False,
    ):
        """Initialise an FSE multishot sequence.

        Args:
            name: Sequence identifier (no underscores — reserved for filenames).
            fov: Field of view in metres.
            Nx: Readout matrix size.
            Ny: Phase-encoding matrix size. Must be divisible by ETL.
            slice_thickness: Slice thickness in metres.
            TR: Repetition time in **milliseconds**.
            TE: First echo time in **milliseconds** (RF90 centre → echo_0 centre).
            ETL: Echo train length — number of refocused echoes per TR. Default=1
                (conventional SE). Ny must be divisible by ETL.
            N_slices: Number of slices.
            system_type: Hardware limits preset.
            rf90_duration: Duration of the 90° sinc excitation pulse in seconds.
            rf180_duration: Duration of each 180° refocusing pulse in seconds.
                0 (default) uses rf90_duration.
            resolution: Isotropic in-plane resolution in mm (overrides Nx/Ny).
            flip_angle: Excitation flip angle in degrees.
            apodization: Sinc apodization factor.
            time_bw_product: Time-bandwidth product of sinc pulses.
            dwell_time: ADC dwell time in seconds (None = auto-compute minimum).
            end_spoilers: If True, add spoiler gradients at the end of each TR.
            spoiler_amplitude: Spoiler amplitude as fraction of system max_grad.
            spoiler_duration: Spoiler duration in seconds.
            save_dir: Output directory for .seq files.
            logger: Logger instance (None = create default).
            v141_compat: Encode Pulseq v1.4.1-compatible gradient shapes.
        """
        assert ETL >= 1, "ETL must be >= 1."

        self._init_logging(logger or logging.getLogger("SEMultishotPulseqSeq"), name, system_type, save_dir)
        self._init_system(system_type)
        self._init_imaging_params(fov, Nx, Ny, slice_thickness, TR, N_slices, resolution, flip_angle, apodization, time_bw_product, v141_compat)
        self._init_readout_timing(dwell_time)
        self._init_rf90(rf90_duration)
        self._init_spoilers(end_spoilers, spoiler_amplitude, spoiler_duration)

        assert self.Ny % ETL == 0, (
            f"Ny={self.Ny} must be divisible by ETL={ETL}."
        )
        self.ETL = ETL

        self._init_se(TE, rf180_duration)
        self._calc_se_te_delays()
        self._compute_inter_echo_delay()
        self._calc_se_tr_delay()

    # -------------------------------------------------------------------------
    # Protected initialisation methods
    # -------------------------------------------------------------------------

    def _init_se(self, TE, rf180_duration):
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

        self.phase_encoding_gradients = (np.arange(self.Ny) - self.Ny / 2) * self.delta_k

        # Ensure flat_time is raster-aligned AND that flat_time_raster/2 is grt-aligned.
        self.flat_time_raster = align2rastertime_ceil(self.readout_time, 2 * self.system.grad_raster_time)
        self.gx_raster_difference = self.flat_time_raster - self.readout_time

        if self.gx_raster_difference > 0:
            self.logger.info(f"Raster alignment correction: {self.gx_raster_difference*1e6:.3f} μs added to flat_time")

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

        assert self.gx.flat_time >= self.adc.duration, (
            f"Readout gradient flat time ({self.gx.flat_time:.6f}s) is shorter than "
            f"ADC duration ({self.adc.duration:.6f}s)!"
        )

        self.logger.info(f"ADC readout duration: {self.adc.duration*1e3:.4f} ms")
        self.logger.info(f"Gx flat time (raster-aligned): {self.gx.flat_time*1e3:.4f} ms")
        self.logger.info(f"Raster difference: {self.gx_raster_difference*1e6:.3f} μs")
        self.logger.info(f"ADC delay: {self.adc.delay*1e3:.6f} ms")
        self.logger.info(f"GX rise time: {self.gx.rise_time*1e3:.6f} ms")

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

    def _calc_se_te_delays(self):
        rf90_center = pp.calc_rf_center(self.rf90)[0]
        rf180_center = pp.calc_rf_center(self.rf180)[0]

        rf90_center_with_delay = rf90_center + self.rf90.delay
        rf180_center_with_delay = rf180_center + self.rf180.delay

        time_after_90 = pp.calc_duration(self.rf90, self.gz90) - rf90_center_with_delay
        time_after_180 = pp.calc_duration(self.rf180, self.gz180) - rf180_center_with_delay

        delayTE1_raw = self.TE / 2 - time_after_90 - pp.calc_duration(self.gz90_reph) - rf180_center_with_delay
        self.delayTE1 = align2rastertime_ceil(delayTE1_raw, self.system.grad_raster_time)
        delayTE1_error = self.delayTE1 - delayTE1_raw

        delayTE2_raw = (
            self.TE / 2
            - time_after_180
            - pp.calc_duration(self.gx_pre)
            - pp.calc_duration(self.gy_pre_dummy)
            - self.adc.delay
            - self.readout_time / 2
        )
        delayTE2_compensated = delayTE2_raw - delayTE1_error
        self.delayTE2 = align2rastertime_nearest(delayTE2_compensated, self.system.grad_raster_time)

        # Tick-based final TE correction
        actual_TE1 = time_after_90 + pp.calc_duration(self.gz90_reph) + self.delayTE1 + rf180_center_with_delay
        actual_TE2 = (
            time_after_180
            + pp.calc_duration(self.gx_pre)
            + pp.calc_duration(self.gy_pre_dummy)
            + self.delayTE2
            + self.adc.delay
            + self.readout_time / 2
        )
        te_correction_ticks = int(np.floor((self.TE - (actual_TE1 + actual_TE2)) / self.system.grad_raster_time + 0.5))
        if te_correction_ticks != 0:
            self.logger.info(f"TE tick correction: {te_correction_ticks} ticks ({te_correction_ticks * self.system.grad_raster_time * 1e6:.1f} μs)")
        self.delayTE2 += te_correction_ticks * self.system.grad_raster_time

        assert self.delayTE1 > 0, f"delayTE1 is negative: {self.delayTE1:.6f}s! Increase TE."
        assert self.delayTE2 > 0, f"delayTE2 is negative: {self.delayTE2:.6f}s! Increase TE."

        self.logger.info(
            f"delayTE1_raw: {delayTE1_raw*1e3:.4f} ms -> delayTE1: {self.delayTE1*1e3:.4f} ms "
            f"(diff: {(self.delayTE1-delayTE1_raw)*1e3:.4f} ms)"
        )
        self.logger.info(
            f"delayTE2_raw: {delayTE2_raw*1e3:.4f} ms -> delayTE2: {self.delayTE2*1e3:.4f} ms "
            f"(diff: {(self.delayTE2-delayTE2_raw)*1e3:.4f} ms)"
        )

    def _compute_inter_echo_delay(self):
        """Compute the symmetry gap between gy_rewind and the next RF180.

        After each non-final echo, a kx-rewind gradient (equal to gx_pre, since it
        has the identical area magnitude) restores kx to zero so the next echo's
        gx_pre pre-phases correctly.  The echo train therefore has, between echo n
        and echo n+1::

            gx+ADC → gx_pre[kx-rewind] → gy_rewind → inter_echo_delay → RF180_{n+1}

        Derivation: time(echo_n centre → RF180_{n+1} centre) must equal half-ESP.
        With gx_pre (kx-rewind) and gy_rewind on both sides of the equation, they
        cancel, and the formula simplifies to:

            inter_echo_delay_raw = delayTE2 − rf180_centre_wd
        """
        rf180_center_with_delay = pp.calc_rf_center(self.rf180)[0] + self.rf180.delay

        inter_echo_delay_raw = self.delayTE2 - rf180_center_with_delay
        self.inter_echo_delay = align2rastertime_ceil(inter_echo_delay_raw, self.system.grad_raster_time)

        assert self.inter_echo_delay >= 0, (
            f"inter_echo_delay is negative ({inter_echo_delay_raw*1e3:.4f} ms): "
            f"TE={self.TE*1e3:.0f} ms is too short for the echo train. Increase TE."
        )

        self.ESP = 2.0 * (
            pp.calc_duration(self.gx_pre)
            + pp.calc_duration(self.gy_pre_dummy)
            + self.delayTE2
            + self.adc.delay
            + self.readout_time / 2
        )

        self.logger.info(f"Echo spacing (ESP): {self.ESP*1e3:.3f} ms")
        self.logger.info(f"inter_echo_delay: {self.inter_echo_delay*1e3:.4f} ms")
        self.logger.info(f"N_shots: {self.Ny // self.ETL}")

    def _calc_se_tr_delay(self):
        # Each echo contributes: RF180 + gx_pre + gy + delayTE2 + gx+ADC + gy_rewind
        # Inter-echo gaps (ETL-1 of them): gx_pre[kx-rewind] + inter_echo_delay
        # The last echo has no kx-rewind or inter_echo_delay.
        # When ETL=1, the (ETL-1) terms vanish → identical to the single-echo SE.
        time_used = (
            pp.calc_duration(self.rf90, self.gz90)
            + pp.calc_duration(self.gz90_reph)
            + self.delayTE1
            + self.ETL * (
                pp.calc_duration(self.rf180, self.gz180)
                + pp.calc_duration(self.gx_pre)          # phase encode pre-phaser
                + pp.calc_duration(self.gy_pre_dummy)    # phase encode
                + self.delayTE2
                + pp.calc_duration(self.gx, self.adc)
                + pp.calc_duration(self.gy_pre_dummy)    # gy rewind
            )
            + (self.ETL - 1) * (
                pp.calc_duration(self.gx_pre)            # kx rewind
                + self.inter_echo_delay
            )
        )

        if self.end_spoilers:
            time_used += pp.calc_duration(self.spoiler_x, self.spoiler_y, self.spoiler_z)

        delayTR_exact = self.TR - time_used
        delayTR = align2rastertime_ceil(delayTR_exact, self.system.grad_raster_time)

        self.logger.info(f"Time used in TR (without delayTR): {time_used*1e3:.4f} ms of {self.TR*1e3:.4f} ms")
        self.logger.info(f"delayTR_exact: {delayTR_exact*1e3:.4f} ms -> delayTR: {delayTR*1e3:.4f} ms")

        assert delayTR > 0, (
            f"delayTR is negative ({delayTR:.6f}s)! "
            f"Increase TR or reduce ETL (currently ETL={self.ETL})."
        )
        self.delayTR = delayTR

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def init_message(self):
        self.logger.info(f"Initializing SE Multishot Pulseq Sequence: {self.name}")

    def metadata(self):
        meta = super().metadata().copy()
        meta.update(
            {
                "TE": self.TE,
                "ETL": self.ETL,
                "ESP": self.ESP,
                "N_shots": self.Ny // self.ETL,
            }
        )
        return meta

    def get_save_filename(self, full_path=False) -> str:
        filename = super().get_save_filename()
        suffix = f"_TE{self.TE*1000:.0f}ms_ETL{self.ETL}{'_spoilers' if self.end_spoilers else ''}"
        filename = filename.replace(".seq", f"{suffix}.seq")
        if full_path:
            return os.path.join(self.save_dir, filename)
        return filename

    def write(self, filename=None):
        filename = filename or self.get_save_filename(full_path=True)
        self.seq.write(filename, v141_compat=self.v141_compat)
        self.logger.info(f"Sequence written to {filename}")

    def build_seq(self) -> pp.Sequence:
        seq = pp.Sequence(self.system)
        N_shots = self.Ny // self.ETL
        gy_dur = pp.calc_duration(self.gy_pre_dummy)

        for shot in range(N_shots):
            pe_values = self.phase_encoding_gradients[shot * self.ETL : (shot + 1) * self.ETL]

            gy_list = [
                pp.make_trapezoid(channel="y", system=self.system, area=pe, duration=gy_dur)
                for pe in pe_values
            ]
            gy_rewind_list = [
                pp.make_trapezoid(channel="y", system=self.system, area=-pe, duration=gy_dur)
                for pe in pe_values
            ]

            seq.add_block(self.rf90, self.gz90)
            seq.add_block(self.gz90_reph)
            seq.add_block(pp.make_delay(self.delayTE1))

            for n in range(self.ETL):
                seq.add_block(self.rf180, self.gz180)
                seq.add_block(self.gx_pre)
                seq.add_block(gy_list[n])
                seq.add_block(pp.make_delay(self.delayTE2))
                seq.add_block(self.gx, self.adc)
                if n < self.ETL - 1:
                    # Rewind kx to zero so the next echo's gx_pre starts from kx=0.
                    # gx_pre has the same area magnitude as the required kx rewind.
                    seq.add_block(self.gx_pre)
                seq.add_block(gy_rewind_list[n])
                if n < self.ETL - 1:
                    seq.add_block(pp.make_delay(self.inter_echo_delay))

            if self.end_spoilers:
                seq.add_block(self.spoiler_x, self.spoiler_y, self.spoiler_z)

            seq.add_block(pp.make_delay(self.delayTR))

        self.seq = seq
        return seq


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(__file__))

    fov = 0.224
    res = 2.3333333

    results = {}
    for etl in [1, 4, 8]:
        se = SEMultishotPulseqSeq(
            name="SEMultishot",
            fov=fov,
            Nx=96,
            Ny=96,
            slice_thickness=res*1e-3,
            TR=5000,
            TE=90,
            ETL=etl,
            rf90_duration=0.003,
            dwell_time=5e-6,
            system_type=SystemLimitType.SAFE,
            end_spoilers=False,
            v141_compat=True,
        )
        se.build_seq()
        se.report()
        results[etl] = se.validate_sequence_properties()
        se.write()
        se.plot()
        se.plot_kspace_traj()
# %%
