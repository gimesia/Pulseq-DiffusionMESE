"""
Simulation utility functions for Pulseq DiffusionMESE showcase.

Provides helpers for phantom manipulation, EPI Nyquist ghost correction, and
k-space-to-image reconstruction used by simulateSE.py and simulateMESE.py.

Author      : Aron Gimesi <aron.gimesi@tecnico.ulisboa.pt>
Affiliation : Instituto Superior Técnico | MSCA-DN IQ-BRAIN
Date        : 2026
Context     : ESMRMB 2026 — Pulseq DiffusionMESE showcase

Funding acknowledgement (mandatory):
    IQ-BRAIN is funded by the European Union (MSCA Doctoral Network,
    December 2024–November 2028, Grant Agreement No. 101169519).
"""

from typing import Any, Literal, Literal, Tuple, Union

from matplotlib import pyplot as plt
import numpy as np
from pypulseq import Sequence
import torch
import torch.nn.functional as F


def add_tumor_to_phantom(
    phantom: Any,  # Expects an object with a .D attribute (torch.Tensor)
    tumor_size: Tuple[int, int, int],
    tumor_location: Union[Tuple[int, int, int], Literal["tl", "bl", "tr", "br"]],
    adc_tumor_core: float = 1.5,
    adc_tumor_border: float = 2.5,
) -> Any:
    """
    Adds a tumor structure to the phantom's data (phantom.D) with specified size and location.

    Typical brain tumor ADC values are around ~1.5 * 10^-3 mm^2/s,
    which lies between GM/WM and CSF (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3000221)

    The tumor is characterized by an interior ADC (adc_tumor_core) and a 1-voxel border
    (adc_tumor_border). The function modifies the phantom.D tensor in-place.

    Args:
        phantom: An object (e.g., an MR0Phantom) with a 'D' attribute that is a torch.Tensor (2D or 3D).
        tumor_size: A tuple (sx, sy, sz) defining the size of the tumor in voxels.
        tumor_location: A tuple (cx, cy, cz) for specific center coordinates,
                        or 'tl' (top-left), 'bl', 'tr', 'br' for quarter placement.
        adc_tumor_core: The Apparent Diffusion Coefficient (ADC) value for the tumor core.
        adc_tumor_border: The ADC value for the 1-voxel thick tumor border.

    Returns:
        The modified phantom object (the modification is in-place).
    """
    D = phantom.D
    dims = tuple(D.shape)

    # 1. Determine dimensions
    if len(dims) == 3:
        nx, ny, nz = dims
    elif len(dims) == 2:
        nx, ny = dims
        nz = 1
    else:
        raise ValueError("Phantom data D must be 2D or 3D.")

    # 2. Determine tumor center coordinates (cx, cy, cz)
    sx, sy, sz = tumor_size

    # Tumor always centered on the Z-dimension slice
    center_z = nz // 2 if nz > 1 else 0

    if isinstance(tumor_location, str):
        # Calculate half-point for quarter division
        half_nx, half_ny = nx // 2, ny // 2

        # Determine center_x and center_y for the chosen quarter
        if tumor_location == "bl":
            # Top-Left: x from 0 to half_nx, y from 0 to half_ny
            center_x = half_nx // 2
            center_y = half_ny // 2
        elif tumor_location == "br":
            # Top-Right: x from half_nx to nx, y from 0 to half_ny
            center_x = (half_nx + nx) // 2
            center_y = half_ny // 2
        elif tumor_location == "tl":
            # Bottom-Left: x from 0 to half_nx, y from half_ny to ny
            center_x = half_nx // 2
            center_y = (half_ny + ny) // 2
        elif tumor_location == "tr":
            # Bottom-Right: x from half_nx to nx, y from half_ny to ny
            center_x = (half_nx + nx) // 2
            center_y = (half_ny + ny) // 2
        else:
            raise ValueError(
                f"Invalid tumor_location string '{tumor_location}'. Must be 'tl', 'bl', 'tr', or 'br'."
            )

        cx, cy, cz = (center_x, center_y, center_z)

    elif isinstance(tumor_location, tuple) and len(tumor_location) == 3:
        cx, cy, cz = tumor_location
    else:
        raise ValueError(
            "tumor_location must be a tuple (cx, cy, cz) or one of 'tl', 'bl', 'tr', 'br'."
        )

    # 3. Compute integer start/end indices and clamp to phantom bounds

    # X-dimension indices [x0, x1)
    x0 = max(0, int(cx - sx // 2))
    x1 = min(nx, x0 + int(sx))

    # Y-dimension indices [y0, y1)
    y0 = max(0, int(cy - sy // 2))
    y1 = min(ny, y0 + int(sy))

    # Prepare tensors for assignment, ensuring correct data type
    core_val = torch.tensor(adc_tumor_core, dtype=D.dtype, device=D.device)
    border_val = torch.tensor(adc_tumor_border, dtype=D.dtype, device=D.device)

    # 4. Apply tumor structure
    if D.ndim == 3:
        # Z-dimension indices [z0, z1)
        z0 = max(0, int(cz - sz // 2))
        z1 = min(nz, z0 + int(sz))

        # Fill interior with core ADC
        D[x0:x1, y0:y1, z0:z1] = core_val

        # Set 1-voxel thick border
        if x1 - x0 > 1 and y1 - y0 > 1 and z1 - z0 > 0:
            # x-faces
            D[x0, y0:y1, z0:z1] = border_val
            D[x1 - 1, y0:y1, z0:z1] = border_val
            # y-faces
            D[x0:x1, y0, z0:z1] = border_val
            D[x0:x1, y1 - 1, z0:z1] = border_val

            # Note: Original code did not explicitly border the z-faces (z0, z1-1).
            # If a full 3D box border is required, uncomment these lines:
            # D[x0:x1, y0:y1, z0] = border_val
            # D[x0:x1, y0:y1, z1 - 1] = border_val

    # 2D phantom (x,y)
    else:  # D.ndim == 2
        # Fill interior with core ADC
        D[x0:x1, y0:y1] = core_val

        # Set 1-voxel thick border
        if x1 - x0 > 1 and y1 - y0 > 1:
            D[x0, y0:y1] = border_val
            D[x1 - 1, y0:y1] = border_val
            D[x0:x1, y0] = border_val
            D[x0:x1, y1 - 1] = border_val

    # Update the phantom's data attribute (D was modified in-place)
    phantom.D = D

    return phantom


def epi_ghost_correction(
    calib_signal,
    epi_signal,
) -> np.ndarray:
    """Navigator-based EPI Nyquist ghost correction (linear phase model).

    Uses the navigator lines (acquired without phase-encode blips, all at ky≈0)
    to estimate a linear phase ramp + constant offset between forward (even) and
    reverse (odd) readout polarities. The correction is applied symmetrically:
    +Δφ/2 on even lines, −Δφ/2 on odd lines, working in the 1-D image domain
    along the readout direction. The result is transformed back to k-space so
    the output is a drop-in replacement for the input and is compatible with
    both FFT and NUFFT pipelines.

    Odd navigator/EPI lines are acquired with the readout gradient reversed
    (kx from +k_max to −k_max). They are flipped before the 1-D FFT so that
    all lines have kx increasing left-to-right, then restored to their original
    sample ordering in the output.

    Args:
        calib_signal: (N_nav, N_samples) complex array, navigator k-space lines.
        epi_signal:   (NY_acq, N_samples) complex array, EPI k-space to correct.

    Returns:
        (NY_acq, N_samples) complex64 numpy array, phase-corrected k-space.
    """
    if hasattr(calib_signal, "numpy"):
        calib_signal = calib_signal.numpy()
    if hasattr(epi_signal, "numpy"):
        epi_signal = epi_signal.numpy()

    calib_signal = np.asarray(calib_signal, dtype=np.complex64)
    epi_signal = np.asarray(epi_signal, dtype=np.complex64)
    N = calib_signal.shape[-1]
    x = np.arange(N, dtype=float)

    def _kspace_to_img(line, is_odd: bool) -> np.ndarray:
        """1-D k-space line → image space. Flips odd lines first."""
        kx = line[::-1].copy() if is_odd else line
        return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(kx)))

    def _img_to_kspace(img, is_odd: bool) -> np.ndarray:
        """Inverse of _kspace_to_img."""
        kx = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(img)))
        return kx[::-1].copy() if is_odd else kx

    # --- Phase estimation from navigator ---
    even_imgs = np.array(
        [_kspace_to_img(calib_signal[i], False) for i in range(0, len(calib_signal), 2)]
    )
    odd_imgs = np.array(
        [_kspace_to_img(calib_signal[i], True) for i in range(1, len(calib_signal), 2)]
    )

    even_avg = np.mean(even_imgs, axis=0)  # (N,) complex
    odd_avg = np.mean(odd_imgs, axis=0)

    delta_phi = np.unwrap(np.angle(even_avg * np.conj(odd_avg)))  # (N,) real
    a, b = np.polyfit(x, delta_phi, 1)
    phi_fit = (a * x + b).astype(np.float32)  # (N,) linear ghost phase

    # --- Apply correction to EPI lines ---
    corrected = np.empty_like(epi_signal)
    for i, line in enumerate(epi_signal):
        is_odd = bool(i % 2)
        img = _kspace_to_img(line, is_odd)
        sign = -1.0 if is_odd else +1.0
        img_corr = img * np.exp(1j * sign * phi_fit / 2)
        corrected[i] = _img_to_kspace(img_corr, is_odd)

    return corrected


def fft_reconstruct_image(kspace, use_gpu=False):
    """
    Reconstruct a magnitude image from 2-D Cartesian k-space via inverse FFT.

    Assumes the DC component is at the centre of the array (standard convention).
    Applies ``ifftshift`` before ``ifft2`` and ``fftshift`` after to place the image
    origin at the array centre.

    Args:
        kspace: (NY, NX) array-like or torch.Tensor, complex k-space data.
        use_gpu: Move computation to CUDA if available; falls back to CPU silently.

    Returns:
        Tuple of:
            magnitude (np.ndarray): ``|image|``, shape (NY, NX), float.
            image (np.ndarray): Complex image, shape (NY, NX), complex64.
    """
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    if not isinstance(kspace, torch.Tensor):
        kspace = torch.from_numpy(kspace)

    kspace = kspace.to(device)

    if not torch.is_complex(kspace):
        kspace = kspace.to(torch.complex64)

    # DC at centre of array (standard convention)
    kspace_shifted = torch.fft.ifftshift(kspace, dim=(-2, -1))

    image = torch.fft.ifft2(kspace_shifted, dim=(-2, -1))
    image = torch.fft.fftshift(image, dim=(-2, -1))

    magnitude = torch.abs(image)
    return magnitude.cpu().numpy(), image.cpu().numpy()


def visualize_kspace_trajectory(seq: Sequence):
    """Plot the continuous and ADC-sampled k-space trajectory for a sequence."""
    # Trajectories (new API returns 5 values)
    ktraj_adc, ktraj, _, _, t_adc = seq.calculate_kspace()

    # Build a time axis for the continuous k-space (sampled on grad raster)
    t_ktraj = np.arange(ktraj.shape[1]) * seq.grad_raster_time

    plt.figure()
    plt.plot(t_ktraj, ktraj.T)  # full k-space vs time (kx, ky, kz)
    plt.plot(t_adc, ktraj_adc[0, :], ".")  # ADC-sampled kx vs t
    plt.title("Full k-space vs time")
    plt.xlabel("t (s)")

    plt.figure()
    plt.plot(ktraj[0, :], ktraj[1, :], "b")  # continuous trajectory (2D view)
    plt.axis("equal")
    plt.plot(ktraj_adc[0, :], ktraj_adc[1, :], "r.")  # ADC samples
    plt.title("k-space (2D)")
    plt.xlabel("kx")
    plt.ylabel("ky")

    plt.show()


def pad_to_cube(tensor, target_size, pad_depth=False):
    """Pad tensor to cube with equal padding on both sides of each dimension.

    Args:
        tensor: shape (H, W, D) or (C, H, W, D)
        target_size: int target for spatial dimensions (applied to H and W,
            and to D only if pad_depth is True)
        pad_depth: if False, do not pad the depth (D) dimension.
    """
    # tensor shape: (H, W, D) or (C, H, W, D)
    spatial_dims = list(tensor.shape[-3:])
    # If not padding depth, set target for depth to current size
    if not pad_depth:
        depth_target = spatial_dims[-1]
    else:
        depth_target = target_size

    targets = [target_size, target_size, depth_target]
    pads = []
    for dim_size, tgt in zip(reversed(spatial_dims), reversed(targets)):
        total_pad = max(0, tgt - dim_size)
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        pads.extend([pad_before, pad_after])

    return F.pad(tensor, pads, mode="constant", value=0)
