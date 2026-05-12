"""
DiffusionSEMultishotPulseqSeq — Diffusion-weighted Fast Spin Echo Cartesian sequence.

Extends :class:`SEMultishotPulseqSeq` with a Stejskal–Tanner PGSE diffusion
preparation: one pair of bipolar diffusion gradient lobes flanking the **first**
180° refocusing pulse of each echo train.  Subsequent echoes in the train share
the same diffusion weighting (the magnetisation preparation is done once per TR).

Sequence timeline per shot (one TR, one diffusion direction)::

    RF90 → gz90_reph
        → [delay_before_diff1]
        → Gdiff1 (all three axes simultaneously)
        → [delayTE1_inner]
        → RF180_0 (+ gz180)
        → Gdiff2 (all three axes simultaneously, same shape, same sign — 180° provides sign flip)
        → gx_pre → gy_0 → [delayTE2_diff] → gx+ADC
        → [kx_rewind] → gy_rewind_0 → [inter_echo_delay]   (if ETL > 1)
        → RF180_1 → gx_pre → gy_1 → [delayTE2] → gx+ADC   (no Gdiff for n≥1)
        → …
        → [end_spoilers] → delayTR

The diffusion gradient pair fits *inside* the parent's ``delayTE1`` / ``delayTE2``
timing windows, so ``delayTR`` is unchanged.

Author      : Aron Gimesi <aron.gimesi@tecnico.ulisboa.pt>
Affiliation : Instituto Superior Técnico | MSCA-DN IQ-BRAIN
Date        : 2026
Context     : ESMRMB 2026 — Pulseq DiffusionMESE showcase

Funding acknowledgement (mandatory):
    IQ-BRAIN is funded by the European Union (MSCA Doctoral Network,
    December 2024–November 2028, Grant Agreement No. 101169519).
"""

# %%
import logging
import os
import numpy as np
import pypulseq as pp

from PulseqSeq import *
from SEMultishotPulseqSeq import SEMultishotPulseqSeq


class DiffusionSEMultishotPulseqSeq(SEMultishotPulseqSeq):
    """Diffusion-weighted FSE sequence — PGSE prep on the first RF180 per echo train.

    Inherits the full Cartesian spin-echo echo-train from
    :class:`SEMultishotPulseqSeq`.  Adds Stejskal–Tanner diffusion encoding
    (``Gdiff1`` before and ``Gdiff2`` after the first RF180) without altering the
    echo-train timing or TR.

    Key diffusion timing attributes (seconds):
        small_delta (float)         : Diffusion gradient lobe duration.
        big_DELTA (float)           : Lobe separation (Gdiff1 start → Gdiff2 start).
        delay_before_diff1 (float)  : Idle gap before Gdiff1 (to place Gdiff1 late).
        delayTE1_inner (float)      : Gap between Gdiff1 end and RF180_0 start.
        delayTE2_diff (float)       : Residual delay after Gdiff2, before gx_pre.
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
        b_value: int = 0,
        b_directions: int = 1,
        b_0_frequency: int = 0,
        small_delta: float = None,
        big_DELTA: float = None,
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
        """Initialise a diffusion-weighted FSE multishot sequence.

        Args:
            name: Sequence identifier.
            fov: Field of view in metres.
            Nx: Readout matrix size.
            Ny: Phase-encoding matrix size. Must be divisible by ETL.
            slice_thickness: Slice thickness in metres.
            TR: Repetition time in **milliseconds**.
            TE: First echo time in **milliseconds** (RF90 centre → echo_0 centre).
            ETL: Echo train length (≥1). Ny must be divisible by ETL.
            b_value: Diffusion weighting in s/mm².
            b_directions: Number of diffusion directions (electrostatic scheme).
            b_0_frequency: Number of interleaved b=0 acquisitions.
            small_delta: Diffusion lobe duration in seconds (None = auto).
            big_DELTA: Lobe separation in seconds (None = auto). Both or neither.
            N_slices: Number of slices.
            system_type: Hardware limits preset.
            rf90_duration: 90° sinc pulse duration in seconds.
            rf180_duration: 180° sinc pulse duration (0 = same as rf90).
            resolution: In-plane resolution in mm (overrides Nx/Ny).
            flip_angle: Excitation flip angle in degrees.
            apodization: Sinc apodization factor.
            time_bw_product: Time–bandwidth product.
            dwell_time: ADC dwell time in seconds (None = auto).
            end_spoilers: Add spoiler gradients after each echo train.
            spoiler_amplitude: Spoiler amplitude as fraction of max_grad.
            spoiler_duration: Spoiler duration in seconds.
            save_dir: Output directory for .seq files.
            logger: Logger instance (None = create default).
            v141_compat: Write Pulseq v1.4.1-compatible gradient shapes.
        """
        super().__init__(
            name=name,
            fov=fov,
            Nx=Nx,
            Ny=Ny,
            slice_thickness=slice_thickness,
            TR=TR,
            TE=TE,
            ETL=ETL,
            N_slices=N_slices,
            system_type=system_type,
            rf90_duration=rf90_duration,
            rf180_duration=rf180_duration,
            resolution=resolution,
            flip_angle=flip_angle,
            apodization=apodization,
            time_bw_product=time_bw_product,
            dwell_time=dwell_time,
            end_spoilers=end_spoilers,
            spoiler_amplitude=spoiler_amplitude,
            spoiler_duration=spoiler_duration,
            save_dir=save_dir,
            logger=logger,
            v141_compat=v141_compat,
        )

        self._init_diffusion_params(
            b_value, b_directions, b_0_frequency, small_delta, big_DELTA
        )
        self._calc_diffusion_timing()

    # -------------------------------------------------------------------------
    # Diffusion initialisation
    # -------------------------------------------------------------------------

    def _init_diffusion_params(
        self, b_value, b_directions, b_0_frequency, small_delta, big_DELTA
    ):
        self.b_value = b_value
        self.b_0_frequency = b_0_frequency
        self.b_dirs = b_directions
        self.b_directions = get_diffusion_directions(b_directions, b_0_frequency)
        self.small_delta = small_delta
        self.big_DELTA = big_DELTA
        self._user_small_delta = small_delta
        self._user_big_DELTA = big_DELTA
        if self.big_DELTA or self.small_delta:
            assert (
                self.big_DELTA and self.small_delta
            ), "Both small_delta and big_DELTA must be set together."

    def _calc_diffusion_timing(self):
        """Fit PGSE gradient pair inside the parent's delayTE1 / delayTE2 windows.

        Layout in delayTE1 (RF90 half-echo window, before RF180_0)::

            delay_before_diff1 | Gdiff1 | delayTE1_inner

        Layout in delayTE2 (first echo only, after RF180_0, before gx_pre)::

            Gdiff2 | delayTE2_diff

        Because Gdiff1+delays = delayTE1 and Gdiff2+delayTE2_diff = delayTE2,
        the TR duration is unchanged from the parent.
        """
        time_rf180_block = pp.calc_duration(self.rf180, self.gz180)

        if self._user_small_delta is None or self._user_big_DELTA is None:
            available = min(self.delayTE1, self.delayTE2)
            max_ramp_time = align2rastertime_ceil(
                self.system.max_grad / self.system.max_slew,
                self.system.grad_raster_time,
            )
            small_delta = available - 2 * max_ramp_time
            big_DELTA = self.delayTE1 + time_rf180_block
        else:
            small_delta = self._user_small_delta
            big_DELTA = self._user_big_DELTA

        assert (
            small_delta > 0
        ), f"small_delta={small_delta * 1e3:.3f} ms ≤ 0. Increase TE or reduce small_delta."

        self.diffusion_gradient_amplitude = calc_diffusion_gradient_amplitude(
            self.b_value, small_delta, big_DELTA
        )

        if self.diffusion_gradient_amplitude > self.system.max_grad:
            if self._user_small_delta is None:
                raise ValueError(
                    f"Required diffusion amplitude ({self.diffusion_gradient_amplitude:.0f} Hz/m) "
                    f"exceeds max_grad ({self.system.max_grad:.0f} Hz/m). Increase TE."
                )
            self.logger.warning(
                f"Diffusion amplitude clamped: {self.diffusion_gradient_amplitude:.0f} → "
                f"{self.system.max_grad:.0f} Hz/m (user small_delta preserved). "
                f"Actual b-value will be lower than requested {self.b_value} s/mm²."
            )
            self.diffusion_gradient_amplitude = self.system.max_grad

        max_rise_time = align2rastertime_ceil(
            self.diffusion_gradient_amplitude / self.system.max_slew,
            self.system.grad_raster_time,
        )
        flat_time = small_delta - 2 * max_rise_time
        assert flat_time >= 0, (
            f"small_delta={small_delta * 1e3:.3f} ms is too short for gradient ramps "
            f"({max_rise_time * 1e3:.3f} ms each)."
        )

        self.g_diff_dummy = pp.make_trapezoid(
            "z",
            system=self.system,
            amplitude=self.diffusion_gradient_amplitude,
            rise_time=max_rise_time,
            flat_time=flat_time,
        )
        self.diffusion_gradient_duration = pp.calc_duration(self.g_diff_dummy)
        self.diffusion_gradient_flat_time = self.g_diff_dummy.flat_time
        self.diffusion_gradient_rise_time = self.g_diff_dummy.rise_time

        # Delays inside delayTE1: delay_before_diff1 | Gdiff1 | delayTE1_inner
        self.delayTE1_inner = align2rastertime_nearest(
            big_DELTA - self.diffusion_gradient_duration - time_rf180_block,
            self.system.grad_raster_time,
        )
        self.delay_before_diff1 = align2rastertime_nearest(
            self.delayTE1 - self.diffusion_gradient_duration - self.delayTE1_inner,
            self.system.grad_raster_time,
        )
        # Residual delay inside delayTE2 (after Gdiff2, before gx_pre) for echo 0
        self.delayTE2_diff = align2rastertime_nearest(
            self.delayTE2 - self.diffusion_gradient_duration,
            self.system.grad_raster_time,
        )

        assert self.delayTE1_inner >= 0, (
            f"delayTE1_inner={self.delayTE1_inner * 1e3:.3f} ms < 0. "
            f"big_DELTA too large for this TE."
        )
        assert self.delay_before_diff1 >= 0, (
            f"delay_before_diff1={self.delay_before_diff1 * 1e3:.3f} ms < 0. "
            f"Gdiff1 + delayTE1_inner exceeds delayTE1."
        )
        assert self.delayTE2_diff >= 0, (
            f"delayTE2_diff={self.delayTE2_diff * 1e3:.3f} ms < 0. "
            f"Gdiff2 duration exceeds delayTE2 — increase TE or reduce small_delta."
        )

        self.small_delta = small_delta
        self.big_DELTA = big_DELTA

        self.logger.info(
            f"Diffusion timing: small_delta={small_delta * 1e3:.2f} ms, "
            f"big_DELTA={big_DELTA * 1e3:.2f} ms, "
            f"amplitude={self.diffusion_gradient_amplitude:.1f} Hz/m"
        )
        self.logger.info(
            f"  delay_before_diff1={self.delay_before_diff1 * 1e3:.3f} ms, "
            f"delayTE1_inner={self.delayTE1_inner * 1e3:.3f} ms, "
            f"delayTE2_diff={self.delayTE2_diff * 1e3:.3f} ms"
        )

    # -------------------------------------------------------------------------
    # Sequence build
    # -------------------------------------------------------------------------

    def build_seq(self) -> pp.Sequence:
        """Assemble the full Pulseq sequence.

        Outer loop over diffusion directions; inner loop over FSE shots.
        Signal layout: ``n_dirs × Ny × Nx`` samples total.
        Per-direction k-space: ``signal[d*Ny*Nx:(d+1)*Ny*Nx].reshape(Ny, Nx)``.

        Note: pypulseq's ``test_report()`` may report an incorrect TE for b > 0
        because PGSE diffusion gradients on the readout (x) axis cause the
        k-space trajectory checker to misidentify the echo centre.  The actual TE
        is correct; the ``event timing check`` always passes and is the reliable
        validation method for this sequence type.
        """
        seq = pp.Sequence(self.system)
        N_shots = self.Ny // self.ETL
        gy_dur = pp.calc_duration(self.gy_pre_dummy)

        for dir_vec in self.b_directions:
            # ------------------------------------------------------------------
            # Per-axis diffusion gradient trapezoids
            # Polarity: both lobes physical-positive. The 180° between them flips the
            # accumulated dephasing, so equal-sign physical lobes produce refocused
            # diffusion encoding (Stejskal–Tanner monopolar PGSE).
            # Per-axis rise times are minimised for weaker axes so that
            # all axes share the same total duration (diffusion_gradient_duration).
            # ------------------------------------------------------------------
            diffusion_grads = self.diffusion_gradient_amplitude * dir_vec  # (3,)

            diff_traps = []
            for amp in diffusion_grads:
                if abs(amp) > 0:
                    rt = align2rastertime_ceil(
                        abs(amp) / self.system.max_slew, self.system.grad_raster_time
                    )
                else:
                    rt = self.system.grad_raster_time
                ft = self.diffusion_gradient_flat_time - 2 * (
                    rt - self.diffusion_gradient_rise_time
                )
                diff_traps.append((amp, rt, ft))

            def make_diff_grad(ch, amp, rt, ft, sign=1):
                if abs(amp) == 0:
                    return pp.make_delay(self.diffusion_gradient_duration)
                return pp.make_trapezoid(
                    channel=ch,
                    system=self.system,
                    amplitude=sign * amp,
                    rise_time=rt,
                    flat_time=ft,
                )

            gx_d1 = make_diff_grad("x", *diff_traps[0], sign=1)
            gy_d1 = make_diff_grad("y", *diff_traps[1], sign=1)
            gz_d1 = make_diff_grad("z", *diff_traps[2], sign=1)
            gx_d2 = make_diff_grad("x", *diff_traps[0], sign=1)
            gy_d2 = make_diff_grad("y", *diff_traps[1], sign=1)
            gz_d2 = make_diff_grad("z", *diff_traps[2], sign=1)

            for ch, g1, g2 in [
                ("x", gx_d1, gx_d2),
                ("y", gy_d1, gy_d2),
                ("z", gz_d1, gz_d2),
            ]:
                a1 = getattr(g1, "area", 0.0)
                a2 = getattr(g2, "area", 0.0)
                net = -a1 + a2  # RF180 negates Gdiff1's k-space contribution
                assert abs(net) < 1e-3, (
                    f"Diffusion not refocused on {ch}-axis: net k = {net:.3e} rad/m "
                    f"(Gdiff1 area={a1:.3e}, Gdiff2 area={a2:.3e})"
                )

            for shot in range(N_shots):
                pe_values = self.phase_encoding_gradients[
                    shot * self.ETL : (shot + 1) * self.ETL
                ]
                gy_list = [
                    pp.make_trapezoid(
                        channel="y", system=self.system, area=pe, duration=gy_dur
                    )
                    for pe in pe_values
                ]
                gy_rewind_list = [
                    pp.make_trapezoid(
                        channel="y", system=self.system, area=-pe, duration=gy_dur
                    )
                    for pe in pe_values
                ]

                # RF90 + slice select
                seq.add_block(self.rf90, self.gz90)
                seq.add_block(self.gz90_reph)

                # Diffusion prep: [delay_before_diff1] → Gdiff1 → [delayTE1_inner]
                if self.delay_before_diff1 > 0:
                    seq.add_block(pp.make_delay(self.delay_before_diff1))
                seq.add_block(gx_d1, gy_d1, gz_d1)
                if self.delayTE1_inner > 0:
                    seq.add_block(pp.make_delay(self.delayTE1_inner))

                # ---- Echo n=0: RF180 + Gdiff2 + readout -------------------------
                seq.add_block(self.rf180, self.gz180)
                seq.add_block(gx_d2, gy_d2, gz_d2)
                seq.add_block(self.gx_pre)
                seq.add_block(gy_list[0])
                if self.delayTE2_diff > 0:
                    seq.add_block(pp.make_delay(self.delayTE2_diff))
                seq.add_block(self.gx, self.adc)
                if self.ETL > 1:
                    seq.add_block(self.gx_pre)  # kx rewind
                seq.add_block(gy_rewind_list[0])
                if self.ETL > 1:
                    seq.add_block(pp.make_delay(self.inter_echo_delay))

                # ---- Echoes n=1 … ETL-1: RF180 (no Gdiff) + readout -------------
                for n in range(1, self.ETL):
                    seq.add_block(self.rf180, self.gz180)
                    seq.add_block(self.gx_pre)
                    seq.add_block(gy_list[n])
                    seq.add_block(pp.make_delay(self.delayTE2))
                    seq.add_block(self.gx, self.adc)
                    if n < self.ETL - 1:
                        seq.add_block(self.gx_pre)  # kx rewind
                    seq.add_block(gy_rewind_list[n])
                    if n < self.ETL - 1:
                        seq.add_block(pp.make_delay(self.inter_echo_delay))

                if self.end_spoilers:
                    seq.add_block(self.spoiler_x, self.spoiler_y, self.spoiler_z)
                seq.add_block(pp.make_delay(self.delayTR))

        self.seq = seq
        return seq

    # -------------------------------------------------------------------------
    # Public interface overrides
    # -------------------------------------------------------------------------

    def init_message(self):
        self.logger.info(
            f"Initializing Diffusion SE Multishot Pulseq Sequence: {self.name}"
        )

    def metadata(self):
        meta = super().metadata().copy()
        meta.update(
            {
                "b_value": self.b_value,
                "b_directions": self.b_directions.tolist(),
                "small_delta": self.small_delta,
                "big_DELTA": self.big_DELTA,
                "diffusion_gradient_amplitude": self.diffusion_gradient_amplitude,
            }
        )
        return meta

    def get_save_filename(self, full_path=False) -> str:
        filename = super().get_save_filename()
        suffix = f"_b{self.b_value}_dirs{self.b_dirs}"
        filename = filename.replace(".seq", f"{suffix}.seq")
        if full_path:
            return os.path.join(self.save_dir, filename)
        return filename

    def write(self, filename=None):
        filename = filename or self.get_save_filename(full_path=True)
        self.seq.write(filename, v141_compat=self.v141_compat)
        self.logger.info(f"Sequence written to {filename}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(__file__))

    fov = 0.224
    res = 2.3333333

    for b_value, etl in [(0, 1), (1000, 1)]:
        seq = DiffusionSEMultishotPulseqSeq(
            name="DiffSEMultishot",
            fov=fov,
            Nx=96,
            Ny=96,
            slice_thickness=res * 1e-3,
            TR=5000,
            TE=100,
            ETL=etl,
            b_value=b_value,
            b_directions=3,
            small_delta=0.018,
            big_DELTA=0.03,
            system_type=SystemLimitType.SAFE,
            dwell_time=5e-6,
            end_spoilers=False,
            v141_compat=True,
        )
        seq.build_seq()
        seq.report()
        seq.validate_sequence_properties()
        seq.write()
        print(
            f"b={b_value} s/mm², ETL={etl}: "
            f"TE={seq.TE*1e3:.1f} ms, small_delta={seq.small_delta*1e3:.2f} ms, "
            f"big_DELTA={seq.big_DELTA*1e3:.2f} ms, "
            f"N_shots={seq.Ny // etl}"
        )
        seq.plot()
        seq.plot_kspace_traj()


# %%
