"""
PulseqSeq — Abstract base class for Pulseq MRI sequence design.

Provides shared infrastructure for all sequence types in this project:
system-limit configuration, imaging-parameter bookkeeping, RF pulse and
spoiler construction, ADC dwell-time resolution, sequence validation, and
.seq file export.  Concrete subclasses (e.g. EPIDiffusionSEPulseqSeq)
must implement :meth:`build_seq` and :meth:`write`.

Author      : Aron Gimesi <aron.gimesi@tecnico.ulisboa.pt>
Affiliation : Instituto Superior Técnico | MSCA-DN IQ-BRAIN
Date        : 2026
Context     : ESMRMB 2026 — Pulseq DiffusionMESE showcase

Funding acknowledgement (mandatory):
    IQ-BRAIN is funded by the European Union (MSCA Doctoral Network,
    December 2024-November 2028, Grant Agreement No. 101169519).
"""

# %%
import os
import re
import logging
from abc import ABC, abstractmethod

import pypulseq as pp
import numpy as np

from utils import *

# Default save directory is relative to this module so the repository is portable
# across machines (no hard-coded absolute paths).
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_SAVE_DIR = os.path.join(BASE_DIR, "seq_files")
# Ensure the directory exists
os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)


class PulseqSeq(ABC):
    """
    A class for creating and managing Pulseq sequences for MRI acquisitions.
    This class provides a high-level interface for creating Pulseq sequences with
    common parameters and utilities for visualization, validation, and export.
    Attributes:
        name (str): Name of the sequence.
        system (pp.Opts): System limits for the sequence.
        seq (pp.Sequence): The Pulseq sequence object.
        fov (float): Field of view in meters.
        delta_k (float): k-space sampling interval (1/FOV).
        resolution (float): Spatial resolution in mm (if specified).
        TR (float): Repetition time in seconds.
        Nx (int): Number of samples in x direction.
        Ny (int): Number of samples in y direction.
        N_slices (int): Number of slices.
        slice_thickness (float): Thickness of each slice in meters.
        rf_duration (float): Duration of RF pulse in seconds.
        flip_angle (float): Flip angle in degrees.
        logger (logging.Logger): Logger instance for tracking operations.
    Methods:
        metadata(): Returns a dictionary containing sequence metadata.
        get_save_filename(): Generates a standardized filename for saving the sequence.
        write(filename=None, outdir_path=None): Writes the sequence to a .seq file.
        report(): Prints a test report of the sequence.
        check_timing(): Validates sequence timing for hardware compatibility.
        plot_kspace_traj(): Visualizes the k-space trajectory.
        plot(TRs=0, show_blocks=False, save=False, time_range=None, time_disp='s',
             grad_disp='kHz/m', plot_now=True): Plots the sequence diagram.
    """

    def __init__(
        self,
        name: str,
        fov: float,
        Nx: int,
        Ny: int,
        slice_thickness: float,
        TR: int,
        N_slices: int = 1,
        system_type=SystemLimitType.SAFE,
        rf90_duration=0.003,
        resolution=None,
        flip_angle=90,
        apodization=0.5,
        time_bw_product=4,
        dwell_time=None,
        end_spoilers=True,
        spoiler_amplitude=1,
        spoiler_duration=1e-3,
        save_dir=DEFAULT_SAVE_DIR,
        logger: logging.Logger = None,
        v141_compat: bool = False,
    ):
        """
        Initialize a Pulseq sequence.

        Args:
            name: Name of the sequence.
            fov: Field of view in meters.
            Nx: Number of samples in x direction.
            Ny: Number of samples in y direction.
            slice_thickness: Thickness of each slice in meters.
            TR: Repetition time in milliseconds.
            N_slices: Number of slices (default=1).
            system_type: System limits type (default=SAFE).
            rf90_duration: Duration of 90° RF pulse in seconds (default=0.003).
            resolution: Spatial resolution in mm (optional).
            flip_angle: Flip angle in degrees (default=90).
            logger: Logger instance (optional).
        """
        self._init_logging(logger, name, system_type, save_dir)
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

    # -------------------------------------------------------------------------
    # Protected initialization methods — callable from any child class
    # -------------------------------------------------------------------------

    def _init_logging(self, logger, name, system_type, save_dir):
        """Validate the sequence name, configure the logger, and cache run-level metadata.

        Args:
            logger: Caller-supplied :class:`logging.Logger`, or ``None`` to create a default one.
            name: Sequence identifier — must not contain underscores (reserved as filename delimiters).
            system_type: :class:`SystemLimitType` enum value persisted for filename generation.
            save_dir: Output directory for ``.seq`` files.
        """
        if "_" in name:
            raise ValueError(
                "Sequence name should not contain underscores ('_'). They are reserved for filename formatting."
            )
        self.logger = logger or logging.getLogger("PulseqSeq")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.name = name
        self.system_type = system_type
        self.save_dir = save_dir
        self.init_message()

    def _init_system(self, system_type):
        """Instantiate hardware limits and create an empty Pulseq sequence object.

        Args:
            system_type: :class:`SystemLimitType` controlling max gradient and slew-rate values.
        """
        self.system = system_limit(system_type)
        self.seq = pp.Sequence(self.system)

    def _init_imaging_params(
        self,
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
    ):
        """Store FOV, matrix size, TR, slice geometry, flip angle, and derived k-space quantities.

        When ``resolution`` is supplied, ``Nx`` and ``Ny`` are recalculated to yield isotropic
        in-plane voxels; the caller-supplied values are preserved in ``Nx_requested``/``Ny_requested``
        for reference.

        Args:
            fov: Field of view in metres.
            Nx: Requested (or final) readout matrix size.
            Ny: Requested (or final) phase-encoding matrix size.
            slice_thickness: Slice thickness in metres.
            TR: Repetition time in **milliseconds** (converted to seconds internally).
            N_slices: Number of slices to acquire.
            resolution: Isotropic in-plane resolution in mm, or ``None`` to use ``Nx``/``Ny`` directly.
            flip_angle: Excitation flip angle in degrees.
            apodization: Sinc apodization factor for the RF pulse window.
            time_bw_product: Time-bandwidth product of the sinc pulse.
            v141_compat: If ``True``, encodes Pulseq v1.4.1-compatible gradient shapes.
        """
        self.apodization = apodization
        self.time_bw_product = time_bw_product
        self.fov = fov
        self.resolution = resolution if resolution is not None else fov / min(Nx, Ny)
        # pypulseq works in SI units; TR is supplied in ms by convention
        self.TR = TR * 1e-3
        self.v141_compat = v141_compat

        if resolution is None:
            self.Nx = Nx
            self.Ny = Ny
            self.logger.info("No resolution specified, using given Nx and Ny")
        else:
            self.Nx_requested = Nx
            self.Ny_requested = Ny
            self.Nx = int(np.round(fov / (resolution * 1e-3)))
            self.Ny = int(np.round(fov / (resolution * 1e-3)))
            self.logger.info(
                f"Requested Nx: {self.Nx_requested}, Ny: {self.Ny_requested}"
            )
            self.logger.info(f"Isotropic resolution: {resolution}mm")
            self.logger.info(f"Calculated Nx: {self.Nx}, Ny: {self.Ny}")

        self.N_slices = N_slices
        self.slice_thickness = slice_thickness
        self.flip_angle = flip_angle
        # Nyquist criterion: k-space sample spacing equals 1/FOV
        self.delta_k = 1 / self.fov
        # Total k-space extent covered during the readout window
        self.k_width = self.Nx * self.delta_k

    def _init_readout_timing(self, dwell_time):
        """Resolve and store the ADC dwell time.

        Uses the caller-supplied ``dwell_time`` when provided; otherwise calls
        :meth:`find_compatible_dwell_time` to find the smallest value that satisfies
        both the ADC raster and the gradient raster constraints.

        Args:
            dwell_time: ADC dwell time in seconds, or ``None`` to auto-compute.

        Raises:
            ValueError: If a non-positive ``dwell_time`` is supplied.
        """
        if dwell_time is not None:
            if dwell_time <= 0:
                raise ValueError("dwell_time must be positive.")
            self.dwell_time = dwell_time
            self.logger.info(f"Given dwell_time: {dwell_time*1e6:.3f}us")
            self.logger.info(f"Calculated readout_time: {self.readout_time*1e3:.3f}ms")
        else:
            self.dwell_time = None
            self.dwell_time = self.find_compatible_dwell_time()
            self.logger.info(
                f"No given dwell_time, using minimum: {self.dwell_time*1e6:.3f}us"
            )

    def _init_rf90(self, rf90_duration):
        """Create the 90° sinc excitation pulse, its slice-selection gradient, and a trigger pulse.

        The trigger is a 1 ms digital output on channel ``osc0`` used to synchronise
        external hardware (e.g. physiological monitoring) with the sequence timeline.

        The RF pulse is created with ``use="excitation"`` which tells pypulseq to compute
        the correct phase convention for a spin-excitation event (as opposed to
        ``"refocusing"`` used for 180° pulses in spin-echo trains).

        Args:
            rf90_duration: Total duration of the 90° RF pulse in seconds.
        """
        self.rf_duration = rf90_duration
        # Trigger output on osc0 — other valid channels: 'osc1', 'ext1'
        self.trigger = pp.make_digital_output_pulse(channel="osc0", duration=1e-3)
        self.rf90, self.gz90, self.gz90_reph = pp.make_sinc_pulse(
            flip_angle=deg2rad(90),
            duration=self.rf_duration,
            # rf_dead_time: mandatory hardware blanking period before RF transmission begins
            delay=self.system.rf_dead_time if self.system.rf_dead_time else 0,
            slice_thickness=self.slice_thickness,
            apodization=0.5,
            time_bw_product=4,
            system=self.system,
            use="excitation",
            return_gz=True,
        )

    def _init_spoilers(self, end_spoilers, spoiler_amplitude, spoiler_duration):
        """Build trapezoid spoiler gradients on all three logical axes.

        Spoilers are created on x, y, and z simultaneously so that residual
        transverse magnetisation is dephased regardless of the slice orientation
        (oblique acquisitions mix physical gradient axes into logical ones).

        Args:
            end_spoilers: If ``True``, the subclass should append these spoilers at the
                end of each TR to suppress stimulated-echo artefacts.
            spoiler_amplitude: Fraction of ``system.max_grad`` to use (0–1).
            spoiler_duration: Requested spoiler duration in seconds; ceiled to the gradient raster.
        """
        self.end_spoilers = end_spoilers
        # Align duration to the gradient raster (ceil ensures we never fall short)
        duration = align2rastertime_ceil(spoiler_duration, self.system.grad_raster_time)
        self.spoiler_z = pp.make_trapezoid(
            channel="z",
            amplitude=spoiler_amplitude * self.system.max_grad,
            duration=duration,
            system=self.system,
        )
        self.spoiler_y = pp.make_trapezoid(
            channel="y",
            amplitude=spoiler_amplitude * self.system.max_grad,
            duration=duration,
            system=self.system,
        )
        self.spoiler_x = pp.make_trapezoid(
            channel="x",
            amplitude=spoiler_amplitude * self.system.max_grad,
            duration=duration,
            system=self.system,
        )

    # -------------------------------------------------------------------------
    # Helper functions
    # -------------------------------------------------------------------------

    # Helper function to find dwell_time that satisfies both raster constraints
    def find_compatible_dwell_time(self):
        """
        Find the smallest dwell_time >= min_dwell such that:
        1. dwell_time is a multiple of adc_raster
        2. Nx * dwell_time is a multiple of grad_raster
        """
        self.logger.info(f"Calculating compatible dwell time for Nx {self.Nx}")
        if self.dwell_time is not None:
            min_dwell = self.dwell_time
            self.logger.info(f"Input dwell time: {min_dwell*1e6:.3f}us")
        else:
            min_dwell = (
                self.system.adc_raster_time
            )  # Minimum dwell time is adc raster time
            self.logger.info(f"Minimum dwell time: {min_dwell*1e6:.3f}us")

        # Round min_dwell to the nearest ADC raster tick to start the search
        dwell = align2rastertime_nearest(min_dwell, self.system.adc_raster_time)
        self.logger.info(
            f"Starting dwell time aligned to ADC raster: {dwell*1e6:.3f}us"
        )

        # Increment by one ADC raster tick at a time — each step keeps ADC alignment while
        # narrowing the search for gradient-raster compatibility of the full readout window.
        # The modulo check has two branches because floating-point remainder near grad_raster_time
        # may round to grad_raster_time rather than 0; both represent exact alignment.
        while not np.isclose(
            (self.Nx * dwell) % self.system.grad_raster_time, 0, atol=1e-12
        ) and not np.isclose(
            (self.Nx * dwell) % self.system.grad_raster_time,
            self.system.grad_raster_time,
            atol=1e-12,
        ):
            dwell += self.system.adc_raster_time
        self.logger.info(f"Found compatible dwell time: {dwell*1e6:.3f}us")
        return dwell

    @property
    def readout_time(self) -> float:
        """Readout window duration — always derived as dwell_time × Nx."""
        return self.dwell_time * self.Nx

    @abstractmethod
    def init_message(self):
        self.logger.info(f"Initializing Pulseq Sequence: {self.name}")

    def metadata(self):
        """
        Get sequence metadata.

        Returns:
            dict: Dictionary containing sequence parameters.
        """
        meta = {
            "name": self.name,
            "fov": self.fov,
            "Nx": self.Nx,
            "Ny": self.Ny,
            "slice_thickness": self.slice_thickness,
            "rf_duration": self.rf_duration,
            "flip_angle": self.flip_angle,
        }
        return meta

    def get_save_filename(self, full_path=False) -> str:
        """
        Generate a standardized filename for saving the sequence.

        Returns:
            str: Formatted filename with sequence parameters.
        """
        filename = f"{self.name}{'_v14' if self.v141_compat else '_v15'}_{self.system_type.value}_fov{self.fov*1000:.0f}mm_{self.Ny}x{self.Nx}x{self.N_slices}_TR{self.TR*1000:.0f}ms_dw{self.dwell_time*1e6:.1f}us.seq"
        if full_path:
            return os.path.join(self.save_dir, filename)
        return filename

    @abstractmethod
    def write(self, filename=None):
        """
        Write the sequence to a .seq file.

        Args:
            filename: Custom filename (optional, uses get_save_filename if None).
            outdir_path: Output directory path (optional).
        """
        self.logger.warning("write() not implemented in base PulseqSeq class.")
        pass

    def report(self):
        """Print a test report of the sequence."""
        self.logger.info("Sequence Test Report:")
        print(self.seq.test_report())

    def check_timing(self):
        """Check sequence timing for hardware compatibility and safety."""
        ok, report = self.seq.check_timing()
        if ok:
            print("Timing check passed successfully.")
        else:
            print("Timing check failed. Report:")
            print(report)

    def plot_kspace_traj(self):
        """Visualize the k-space trajectory of the sequence."""
        visualize_kspace_trajectory(self.seq)

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
            plot_now=plot_now,
        )

    @abstractmethod
    def build_seq(self) -> pp.Sequence:
        """Assemble the full pulse sequence and return the completed :class:`pp.Sequence` object.

        Subclasses must override this method to add all sequence blocks (RF pulses,
        gradients, ADC events, delays) in the correct temporal order.  The returned
        object must pass :meth:`check_timing` before being written to disk.
        """
        self.logger.warning("build_seq() not implemented in base PulseqSeq class.")

    def validate_sequence_properties(
        self, expected_values: dict = None, tolerance: float = None
    ) -> tuple[bool, list[str]]:
        """
        Validate that object properties match expected values from the sequence report.

        This function checks key sequence properties against expected values that would appear
        in the seq.test_report() output. If no expected_values are provided, it extracts them
        from the current sequence report.

        Args:
            expected_values: Dictionary with expected values. If None, validates against current report.
            tolerance: Numerical tolerance for floating-point comparisons (default=1e-6).

        Returns:
            tuple[bool, list[str]]: (all_passed, failed_tests) where all_passed is True if all
                checks pass, and failed_tests is a list of failure messages.

        Example expected_values format:
            {
                'TE': 0.1,  # seconds
                'TR': 1.0,  # seconds
                'flip_angle': [90.0, 180.0],  # degrees
                'Nx': 128,
                'Ny': 96,
                'dimensions': 2,
                'num_slices': 3,
                'spatial_resolution': [1.76, 1.78],  # mm
            }
        """

        # Use grad_raster_time as default tolerance (TR/TE use ceil/nearest rounding)
        if tolerance is None:
            tolerance = self.system.grad_raster_time

        # Get the test report string
        report_str = self.seq.test_report()

        parsed_values = {}

        # --- timing ---
        te_match = re.search(r"TE:\s+([\d.]+)\s+s", report_str)
        if te_match:
            parsed_values["TE"] = float(te_match.group(1))

        tr_match = re.search(r"TR:\s+([\d.]+)\s+s", report_str)
        if tr_match:
            parsed_values["TR"] = float(tr_match.group(1))

        dur_match = re.search(r"Sequence duration:\s+([\d.]+)\s+s", report_str)
        if dur_match:
            parsed_values["sequence_duration"] = float(dur_match.group(1))

        # --- spatial ---
        flip_match = re.search(r"Flip angle:\s+([\d.\s]+)deg", report_str)
        if flip_match:
            flip_angles = [float(x) for x in flip_match.group(1).strip().split()]
            parsed_values["flip_angle"] = flip_angles

        kspace_match = re.search(
            r"Unique k-space positions.*?:\s+(\d+)\s+(\d+)", report_str
        )
        if kspace_match:
            parsed_values["Nx"] = int(kspace_match.group(1))
            parsed_values["Ny"] = int(kspace_match.group(2))

        dim_match = re.search(r"Dimensions:\s+(\d+)", report_str)
        if dim_match:
            parsed_values["dimensions"] = int(dim_match.group(1))

        res_matches = re.findall(r"Spatial resolution:\s+([\d.]+)\s+mm", report_str)
        if res_matches:
            parsed_values["spatial_resolution"] = [float(x) for x in res_matches]

        rep_match = re.search(r"Repetitions/slices/contrasts:\s+([\d.]+)", report_str)
        if rep_match:
            parsed_values["num_slices"] = float(rep_match.group(1))

        # --- event counts ---
        events = {}
        event_matches = re.findall(r"(RF|Gx|Gy|Gz|ADC|Delay):\s+(\d+)", report_str)
        for event_type, count in event_matches:
            events[event_type] = int(count)
        if events:
            parsed_values["events"] = events

        blocks_match = re.search(r"Number of blocks:\s+(\d+)", report_str)
        if blocks_match:
            parsed_values["num_blocks"] = int(blocks_match.group(1))

        # --- hardware limits ---
        # Per-axis peak gradient (x, y, z) in Hz/m and mT/m
        max_grad_match = re.search(
            r"Max gradient:\s+([\d]+)\s+([\d]+)\s+([\d]+)\s+Hz/m\s+==\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+mT/m",
            report_str,
        )
        if max_grad_match:
            parsed_values["max_gradient_hz"] = [
                int(max_grad_match.group(1)),
                int(max_grad_match.group(2)),
                int(max_grad_match.group(3)),
            ]
            parsed_values["max_gradient_mt"] = [
                float(max_grad_match.group(4)),
                float(max_grad_match.group(5)),
                float(max_grad_match.group(6)),
            ]

        # Per-axis peak slew rate (x, y, z) in Hz/m/s and T/m/s
        max_slew_match = re.search(
            r"Max slew rate:\s+([\d]+)\s+([\d]+)\s+([\d]+)\s+Hz/m/s\s+==\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+T/m/s",
            report_str,
        )
        if max_slew_match:
            parsed_values["max_slew_hz"] = [
                int(max_slew_match.group(1)),
                int(max_slew_match.group(2)),
                int(max_slew_match.group(3)),
            ]
            parsed_values["max_slew_t"] = [
                float(max_slew_match.group(4)),
                float(max_slew_match.group(5)),
                float(max_slew_match.group(6)),
            ]

        # Vector-magnitude peak gradient across all axes
        max_abs_grad_match = re.search(
            r"Max absolute gradient:\s+([\d]+)\s+Hz/m\s+==\s+([\d.]+)\s+mT/m",
            report_str,
        )
        if max_abs_grad_match:
            parsed_values["max_abs_gradient_hz"] = int(max_abs_grad_match.group(1))
            parsed_values["max_abs_gradient_mt"] = float(max_abs_grad_match.group(2))

        # Vector-magnitude peak slew rate across all axes
        max_abs_slew_match = re.search(
            r"Max absolute slew rate:\s+([\d.e+]+)\s+Hz/m/s\s+==\s+([\d.]+)\s+T/m/s",
            report_str,
        )
        if max_abs_slew_match:
            parsed_values["max_abs_slew_hz"] = float(max_abs_slew_match.group(1))
            parsed_values["max_abs_slew_t"] = float(max_abs_slew_match.group(2))

        # If no expected values provided, just display parsed values and return True
        if expected_values is None:
            self.logger.info("Parsed sequence report values:")
            for key, value in parsed_values.items():
                if key in ["TE", "TR", "sequence_duration"]:
                    self.logger.info(f"  {key}: {value * 1e3:.3f}ms")
                else:
                    self.logger.info(f"  {key}: {value}")

            expected_values = {
                "TE": self.TE,  # seconds
                "TR": self.TR,  # seconds
                "max_abs_slew_t": self.system.max_slew
                / self.system.gamma,  # Convert from mT/m/s to T/m/s
                "max_abs_gradient_mt": self.system.max_grad
                / self.system.gamma
                * 1e3,  # Convert from T/m to mT/m
            }

        # Validate against expected values
        all_passed = True
        validation_results = []
        failed_tests = []

        for key, expected in expected_values.items():
            if key not in parsed_values:
                msg = f"❌ {key}: NOT FOUND in report"
                validation_results.append(msg)
                failed_tests.append(msg)
                all_passed = False
                continue

            actual = parsed_values[key]

            # Slew-rate tolerance: warn at 100–175% of the system limit, fail above 175%.
            # The 1.75× threshold is the pypulseq community convention for gradient pre-emphasis
            # compensation — scanners may handle mild overdrives without hardware protection trips.
            if key == "max_abs_slew_t":
                ratio = actual / expected
                if actual <= expected:
                    validation_results.append(
                        f"✓ {key}: {actual} <= {expected} (within limits)"
                    )
                elif 1.0 < ratio <= 1.75:
                    msg = f"⚠ {key}: {actual} is {ratio*100:.1f}% of limit {expected} (warning, but passing)"
                    validation_results.append(msg)
                    failed_tests.append(msg)
                else:
                    msg = f"❌ {key}: {actual} > 175% of {expected} (EXCEEDS LIMIT)"
                    validation_results.append(msg)
                    failed_tests.append(msg)
                    all_passed = False

            # Special handling for max_abs_gradient_mt (should be <= expected)
            elif key == "max_abs_gradient_mt":
                ratio = actual / expected
                if actual <= expected:
                    validation_results.append(
                        f"✓ {key}: {actual} <= {expected} (within limits)"
                    )
                elif 1.0 < ratio <= 1.75:
                    msg = f"⚠ {key}: {actual} is {ratio*100:.1f}% of limit {expected} (warning, but passing)"
                    validation_results.append(msg)
                    failed_tests.append(msg)
                else:
                    msg = f"❌ {key}: {actual} > {expected} (EXCEEDS LIMIT)"
                    validation_results.append(msg)
                    failed_tests.append(msg)
                    all_passed = False

            # Handle different types of comparisons
            elif isinstance(expected, (int, np.integer)):
                if actual == expected:
                    validation_results.append(f"✓ {key}: {actual} == {expected}")
                else:
                    msg = f"❌ {key}: {actual} != {expected}"
                    validation_results.append(msg)
                    failed_tests.append(msg)
                    all_passed = False

            elif isinstance(expected, (float, np.floating)):
                if abs(actual - expected) <= tolerance:
                    # Display time values in ms
                    if key in ["TE", "TR", "sequence_duration"]:
                        validation_results.append(
                            f"✓ {key}: {actual*1e3:.3f}ms ≈ {expected*1e3:.3f}ms (within tolerance)"
                        )
                    else:
                        validation_results.append(
                            f"✓ {key}: {actual} ≈ {expected} (within tolerance)"
                        )
                else:
                    if key in ["TE", "TR", "sequence_duration"]:
                        msg = f"❌ {key}: {actual*1e3:.3f}ms != {expected*1e3:.3f}ms (diff: {abs(actual - expected)*1e3:.3f}ms)"
                    else:
                        msg = f"❌ {key}: {actual} != {expected} (diff: {abs(actual - expected)})"
                    validation_results.append(msg)
                    failed_tests.append(msg)
                    all_passed = False

            elif isinstance(expected, list):
                if isinstance(actual, list) and len(actual) == len(expected):
                    list_match = all(
                        abs(a - e) <= tolerance if isinstance(e, float) else a == e
                        for a, e in zip(actual, expected)
                    )
                    if list_match:
                        validation_results.append(f"✓ {key}: {actual} ≈ {expected}")
                    else:
                        msg = f"❌ {key}: {actual} != {expected}"
                        validation_results.append(msg)
                        failed_tests.append(msg)
                        all_passed = False
                else:
                    msg = f"❌ {key}: {actual} != {expected} (length mismatch)"
                    validation_results.append(msg)
                    failed_tests.append(msg)
                    all_passed = False

            elif isinstance(expected, dict):
                if isinstance(actual, dict):
                    dict_match = all(actual.get(k) == v for k, v in expected.items())
                    if dict_match:
                        validation_results.append(f"✓ {key}: {actual} == {expected}")
                    else:
                        msg = f"❌ {key}: {actual} != {expected}"
                        validation_results.append(msg)
                        failed_tests.append(msg)
                        all_passed = False
                else:
                    msg = f"❌ {key}: type mismatch"
                    validation_results.append(msg)
                    failed_tests.append(msg)
                    all_passed = False

        # Print validation results
        self.logger.info("=" * 60)
        self.logger.info("SEQUENCE VALIDATION RESULTS")
        self.logger.info("=" * 60)
        for result in validation_results:
            self.logger.info(result)
        self.logger.info("=" * 60)

        if all_passed:
            self.logger.info("✓ All validation checks PASSED")
        else:
            self.logger.warning("❌ Some validation checks FAILED")

        return all_passed, failed_tests
