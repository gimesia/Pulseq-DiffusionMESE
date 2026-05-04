from enum import Enum
import math

import numpy as np
import torch
import matplotlib.pyplot as plt

import pypulseq as pp

# ===============================================================================
#   General
# ===============================================================================

deg2rad = lambda deg: deg * np.pi / 180
align2rastertime_ceil = lambda x, rt: np.ceil(x / rt) * rt  # align time to raster time
align2rastertime_floor = lambda x, rt: np.floor(x / rt) * rt  # align time to raster time
align2rastertime_nearest = lambda x, rt: np.round(x / rt) * rt  # align time to raster time
milli = lambda x: x / 1000
micro = lambda x: x / 1000000
nano = lambda x: x / 1000000000

def tensor2image(tensor: torch.Tensor) -> np.ndarray:
    return tensor.cpu().numpy()


def image2tensor(image: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(image)


def normalize2float64(image: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    is_tensor = isinstance(image, torch.Tensor)
    if is_tensor:
        image = tensor2image(image)

    # Normalize to [0, 1] range
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val - min_val > 0:
        normalized_image = (image - min_val) / (max_val - min_val)
    else:
        normalized_image = np.zeros_like(image)
    normalized_image = normalized_image.astype(np.float64)

    if is_tensor:
        normalized_image = image2tensor(normalized_image)
    return normalized_image


# ===============================================================================
#   Pulseq
# ===============================================================================
class SystemLimitType(str, Enum):
    EXTRASAFE = "extrasafe"
    SAFE = "safe"
    RISKY = "risky"
    EXTREME = "extreme"


def system_limit(
    type: SystemLimitType, rf_ringdown_time=micro(10), rf_dead_time=micro(100), adc_dead_time=micro(10), adc_raster_time=micro(1)
) -> pp.Opts:
    # NOTE:`    
    #   ADC raster time is a multiple of:
    #       GE's 2 µs raster (the coarsest constraint)
    #       Siemens' 100 ns raster
    #      ` Philips' 500 ns raster
    if type == SystemLimitType.SAFE:
        return pp.Opts(
            max_grad=34,  # mT/m
            grad_unit="mT/m",  # Gradient unit
            max_slew=140,  # T/m/s
            slew_unit="T/m/s",  # Slew rate unit
            rf_ringdown_time=rf_ringdown_time,  # RF ringdown time in seconds
            rf_dead_time=rf_dead_time,  # RF dead time in seconds
            adc_dead_time=adc_dead_time,  # ADC dead time in seconds
            adc_raster_time=adc_raster_time,  # ADC raster time in seconds
            B0=2.89,  # Magnetic field strength in Tesla
        )

    elif type == SystemLimitType.EXTRASAFE:
        return pp.Opts(
            max_grad=32,  # mT/m
            grad_unit="mT/m",  # Gradient unit
            max_slew=130,  # T/m/s
            slew_unit="T/m/s",  # Slew rate unit
            rf_ringdown_time=rf_ringdown_time,  # RF ringdown time in seconds
            rf_dead_time=rf_dead_time,  # RF dead time in seconds
            adc_dead_time=adc_dead_time,  # ADC dead time in seconds
            adc_raster_time=adc_raster_time,  # ADC raster time in seconds
            B0=2.89,  # Magnetic field strength in Tesla
        )

    elif type == SystemLimitType.RISKY:
        return pp.Opts(
            max_grad=36,  # mT/m
            grad_unit="mT/m",  # Gradient unit
            max_slew=160,  # T/m/s
            slew_unit="T/m/s",  # Slew rate unit
            rf_ringdown_time=rf_ringdown_time,  # RF ringdown time in seconds
            rf_dead_time=rf_dead_time,  # RF dead time in seconds
            adc_dead_time=adc_dead_time,  # ADC dead time in seconds
            adc_raster_time=adc_raster_time,  # ADC raster time in seconds
            B0=2.89,  # Magnetic field strength in Tesla
        )

    elif type == SystemLimitType.EXTREME:
        return pp.Opts(
            max_grad=38,  # mT/m
            grad_unit="mT/m",  # Gradient unit
            max_slew=180,  # T/m/s
            slew_unit="T/m/s",  # Slew rate unit
            rf_ringdown_time=rf_ringdown_time,  # RF ringdown time in seconds
            rf_dead_time=rf_dead_time,  # RF dead time in seconds
            adc_dead_time=adc_dead_time,  # ADC dead time in seconds
            adc_raster_time=adc_raster_time,  # ADC raster time in seconds
            B0=2.89,  # Magnetic field strength in Tesla
        )

    else:
        raise NotImplementedError(f"{type} type has not been implemented yet")


# ===============================================================================
#   K-Space
# ===============================================================================
def visualize_kspace_trajectory(seq: pp.Sequence):
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

def fft_reconstruct_image(kspace, use_gpu=True, variant="v1"):
    """Reconstruct an image from k-space via inverse FFT.

    Parameters
    ----------
    kspace:
        Complex k-space array or tensor, shape ``(..., Ny, Nx)``.
    use_gpu:
        Use CUDA if available; silently falls back to CPU.
    variant:
        ``"v1"`` (default) applies ``ifftshift → ifft2 → fftshift``.
        ``"v2"`` applies ``fftshift → ifft2 → fftshift`` use this when the k-space DC component is already at the array corner.

    Returns
    -------
    magnitude : np.ndarray
    complex_img : np.ndarray
    """
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    if not isinstance(kspace, torch.Tensor):
        kspace = torch.from_numpy(kspace)

    kspace = kspace.to(device)

    if not torch.is_complex(kspace):
        kspace = kspace.to(torch.complex64)

    if variant == "v1":
        # DC at centre of array (standard convention)
        kspace_shifted = torch.fft.ifftshift(kspace, dim=(-2, -1))
    else:
        # DC at corner of array
        kspace_shifted = torch.fft.fftshift(kspace, dim=(-2, -1))

    image = torch.fft.ifft2(kspace_shifted, dim=(-2, -1))
    image = torch.fft.fftshift(image, dim=(-2, -1))

    magnitude = torch.abs(image)
    return magnitude.cpu().numpy(), image.cpu().numpy()


def visualize_kspace(
    kspace,
    reconstruction_fn=fft_reconstruct_image,
    use_gpu=True,
    fig_size=(12, 6),
    title: str | None = None,
):
    plt.figure(figsize=fig_size)

    magnitude, image = reconstruction_fn(kspace, use_gpu=use_gpu)
    normalized = normalize2float64(magnitude)

    # Plot K-space magnitude
    plt.subplot(1, 2, 1)
    plt.title("Reconstructed K-Space")
    if isinstance(kspace, torch.Tensor):
        kspace_vis = torch.log(torch.abs(kspace) + 1e-10).squeeze().cpu()
    else:
        kspace_vis = np.log(np.abs(kspace) + 1e-10).squeeze()
    plt.imshow(kspace_vis, aspect="equal")
    plt.xlabel("kx")
    plt.ylabel("ky")

    # Plot Image magnitude
    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Image")
    plt.imshow(normalized.squeeze(), origin="lower")
    plt.axis("off")

    if title:
        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        plt.tight_layout()

    plt.show()
    return normalized

# ===============================================================================
#   Diffusion
# ===============================================================================
def calc_bval(G, delta, Delta, gdiff_rt):
    """
    Calculates the achieved diffusion-weighting (b-value in s/mm2)

    Parameters
    ----------
    G : float
        amplitude of the diffusion gradient (Hz)
    delta : float
            duration of the diffusion gradients (s)
    Delta : float
            time between start of the first and second diffusion gradients (s)
    gdiff_rt : float
               diffusion gradient ramp time (s)
    Returns
    -------
    bval : float
           b-value in s/mm2
    """

    bval = (2 * math.pi * G) ** 2 * ((Delta - delta / 3) * (delta**2) + (gdiff_rt**3) / 30 - delta * (gdiff_rt**2) / 6)

    return bval

def bFactCalc(g, delta, DELTA):
    """
    Calculate b-value using the Stejskal-Tanner equation for rectangular gradients.
    Implementation matches make_dw_se_epi_rs_v3.py

    See Davy Sinnaeve, Concepts in Magn Reson Part A, 40A(2):39–65 (2012).
    b = (gamma^2) g^2 delta^2 sigma^2 (DELTA + 2 (kappa - lambda) delta)
    In Pulseq we use Hz/m gradients (gamma omitted), but diffusion uses phase in radians,
    so include 2*pi.
    For rectangular gradients: sigma=1, lambda=1/2, kappa=1/3  -> (kappa - lambda) = -1/6

    Parameters
    ----------
    g : float
        Gradient amplitude in Hz/m
    delta : float
        Gradient pulse duration in seconds
    DELTA : float
        Time between gradient pulse onsets in seconds

    Returns
    -------
    bval : float
        b-value in s/m²
    """
    sigma = 1.0
    kappa_minus_lambda = (1.0 / 3.0) - 0.5  # = -1/6
    return (2 * np.pi * g * delta * sigma) ** 2 * (DELTA + 2 * kappa_minus_lambda * delta)


def calc_diffusion_gradient_amplitude(b_value, delta, DELTA):
    """
    Calculate diffusion gradient amplitude using the correct Stejskal-Tanner equation.
    Implementation matches make_dw_se_epi_rs_v3.py

    Parameters
    ----------
    b_value : float
        Target b-value in s/m² (e.g., 1e9 for 1000 s/mm²)
    delta : float
        Gradient pulse duration in seconds
    DELTA : float
        Time between gradient pulse onsets in seconds

    Returns
    -------
    G : float
        Gradient amplitude in Hz/m
    """
    # Calculate g to hit requested b using bFactCalc
    g = np.sqrt(b_value * 1e6 / bFactCalc(1.0, delta, DELTA))
    return g



def calc_area_preserving_trapezoid(G_req, small_delta, max_grad, max_slew, grad_raster_time):
    """
    For a user-specified small_delta that produces G_req > max_grad, compute the
    best trapezoid achievable at max_grad within that fixed duration.

    The duration is kept at small_delta (honouring the user's timing intent).
    Amplitude is clamped to max_grad and flat_time is maximised within small_delta.
    The resulting gradient area and b-value will be lower than originally requested.

    For a symmetric trapezoidal gradient with amplitude G, rise_time r, flat_time f:
        area = G * (r + f)  =  G * (total_duration - r)

    Parameters
    ----------
    G_req : float
        Required amplitude in Hz/m (exceeds max_grad).
    small_delta : float
        User-specified total gradient duration in seconds (rise + flat + fall).
    max_grad : float
        System maximum gradient amplitude in Hz/m.
    max_slew : float
        System maximum slew rate in Hz/m/s.
    grad_raster_time : float
        Gradient raster time in seconds.

    Returns
    -------
    dict or None
        dict with keys:
            amplitude      – clamped gradient amplitude (= max_grad), Hz/m
            rise_time      – rise time at max_grad/max_slew (raster-aligned), s
            flat_time      – maximum flat time within small_delta (raster-aligned), s
            total_duration – 2*rise_time + flat_time  (≤ small_delta), s
            area           – achieved gradient area (Hz/m · s)
            delta_eff      – effective encoding duration (= rise_time + flat_time), s
        Returns None if the ramps at max_grad alone exceed small_delta (i.e. even
        a pure-triangle gradient at max_grad does not fit).
    """
    # Rise time at max_grad with max slew (shorter ramp than at G_req)
    rise_time_new = np.ceil(max_grad / max_slew / grad_raster_time) * grad_raster_time

    # Maximise flat_time within the fixed small_delta window
    flat_time_new = np.floor((small_delta - 2 * rise_time_new) / grad_raster_time) * grad_raster_time

    if flat_time_new < 0:
        return None  # max_grad ramps alone don't fit inside small_delta

    total_duration = 2 * rise_time_new + flat_time_new  # ≤ small_delta
    delta_eff = rise_time_new + flat_time_new
    area_achieved = max_grad * delta_eff

    return {
        "amplitude": max_grad,
        "rise_time": rise_time_new,
        "flat_time": flat_time_new,
        "total_duration": total_duration,
        "area": area_achieved,
        "delta_eff": delta_eff,
    }


def get_diffusion_directions(n_directions, insert_b0s_at=0):
    """
    Retrieves list of gradient directions and number of non\-DWI images.
    NOTE: obtained using gen_scheme (MRTrix)

    Parameters
    ----------
    n_directions : integer
            number of diffusion directions to sample
    Returns
    -------
    g : numpy.ndarray
        gradient components for each direction (gx,gy,gz)
    nb0s: integer
          number of non\-DWI volumes to acquire
    """

    if n_directions == 3:
        g = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    elif n_directions == 6:
        g = np.zeros((3, n_directions))
        g = np.array(
            [
                [-0.283341, -0.893706, -0.347862],
                [-0.434044, 0.799575, -0.415074],
                [0.961905, 0.095774, 0.256058],
                [-0.663896, 0.491506, 0.563616],
                [-0.570757, -0.554998, 0.605156],
                [-0.198848, -0.056534, -0.978399],
            ]
        )
    elif n_directions == 12:
        # obtained using gen_scheme (mrtrix)
        g = np.array(
            [
                [0.648514, 0.375307, 0.662249],
                [-0.560493, 0.824711, 0.075496],
                [-0.591977, -0.304283, 0.746307],
                [-0.084472, -0.976168, 0.199902],
                [-0.149626, -0.006494, -0.988721],
                [-0.988211, -0.056904, -0.142130],
                [0.864451, -0.274379, -0.421237],
                [-0.173549, -0.858586, -0.482401],
                [-0.039288, 0.644885, -0.763269],
                [0.729809, 0.673235, -0.118882],
                [0.698325, -0.455759, 0.551929],
                [-0.325340, 0.489774, 0.808873],
            ]
        )
    elif n_directions == 60:
        # obtained using gen_scheme (mrtrix)
        g = np.array(
            [
                [-0.811556, 0.245996, -0.529964],
                [-0.576784, -0.313126, 0.754502],
                [-0.167946, -0.899364, -0.403655],
                [0.755699, -0.512113, -0.408238],
                [0.116846, 0.962654, -0.244221],
                [0.495465, 0.208081, 0.843337],
                [0.901459, 0.385831, -0.196230],
                [-0.248754, 0.420519, -0.872516],
                [-0.047525, 0.444671, 0.894432],
                [-0.508593, 0.857494, -0.077699],
                [0.693558, 0.614042, 0.376737],
                [0.990394, -0.134781, -0.030898],
                [0.019140, -0.684235, 0.729010],
                [0.385221, -0.339346, -0.858166],
                [-0.440289, -0.853536, 0.278609],
                [0.680515, -0.559825, 0.472752],
                [-0.146029, 0.872237, 0.466774],
                [0.317352, 0.195118, -0.928018],
                [-0.796280, -0.129004, -0.591013],
                [-0.711299, 0.249255, 0.657211],
                [-0.838383, -0.538321, -0.085587],
                [0.202544, -0.966710, 0.156357],
                [-0.296747, -0.476761, -0.827430],
                [0.545225, 0.637023, -0.544914],
                [-0.887097, 0.451265, 0.097048],
                [0.034752, -0.124211, 0.991647],
                [0.469222, -0.766720, -0.438145],
                [-0.948457, -0.088803, 0.304209],
                [-0.354311, 0.664176, -0.658281],
                [-0.462117, -0.833550, -0.302724],
                [0.949202, -0.022353, 0.313871],
                [0.791248, 0.065354, -0.607994],
                [-0.004026, 0.992213, 0.124488],
                [-0.357034, 0.107223, -0.927917],
                [0.414504, 0.685422, 0.598651],
                [-0.331743, -0.552720, 0.764492],
                [-0.749058, 0.641807, -0.164305],
                [0.238666, -0.655691, -0.716315],
                [0.619125, 0.784982, 0.022082],
                [0.123966, -0.872070, 0.473419],
                [-0.185240, 0.122014, 0.975089],
                [-0.980282, -0.189151, -0.057166],
                [0.637873, -0.084335, 0.765510],
                [-0.668960, 0.723273, 0.171375],
                [-0.775822, -0.381543, 0.502519],
                [-0.636044, -0.425049, -0.644035],
                [0.229220, 0.809364, -0.540729],
                [-0.538340, 0.531550, 0.653946],
                [0.906105, -0.354129, 0.231445],
                [-0.166743, -0.191681, -0.967189],
                [0.324636, -0.927784, -0.183922],
                [0.551291, -0.398459, 0.733014],
                [0.537753, -0.032690, -0.842468],
                [-0.306182, -0.951456, -0.031381],
                [0.875976, 0.329209, 0.352545],
                [0.902989, -0.218632, -0.369879],
                [-0.456427, 0.801551, -0.386251],
                [0.089001, 0.716134, 0.692265],
                [-0.714965, -0.648438, 0.261444],
                [0.076308, 0.420804, -0.903936],
            ]
        )
    elif n_directions == 93:  # 93 -> GE DWI directions
        g = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.89801, 0.36058, 0.252109],
                [-0.88463, 0.351859, -0.305981],
                [-0.820113, 0.405285, 0.40393],
                [0.834013, 0.3071, -0.458381],
                [0.285365, 0.282254, -0.915915],
                [0.340583, 0.823984, -0.452829],
                [0.404599, 0.877445, 0.257661],
                [0.440576, 0.412017, 0.797581],
                [-0.426648, 0.300898, -0.852896],
                [-0.37152, 0.842631, -0.389802],
                [-0.306974, 0.896079, 0.320641],
                [-0.271745, 0.43067, 0.860627],
                [0.0, 0.0, 0.0],
                [0.89801, 0.36058, 0.252109],
                [-0.88463, 0.351859, -0.305981],
                [-0.820113, 0.405285, 0.40393],
                [0.834013, 0.3071, -0.458381],
                [0.285365, 0.282254, -0.915915],
                [0.340583, 0.823984, -0.452829],
                [0.404599, 0.877445, 0.257661],
                [0.440576, 0.412017, 0.797581],
                [-0.426648, 0.300898, -0.852896],
                [-0.37152, 0.842631, -0.389802],
                [-0.306974, 0.896079, 0.320641],
                [-0.271745, 0.43067, 0.860627],
                [0.0, 0.0, 0.0],
                [0.89801, 0.36058, 0.252109],
                [-0.88463, 0.351859, -0.305981],
                [-0.820113, 0.405285, 0.40393],
                [0.834013, 0.3071, -0.458381],
                [0.285365, 0.282254, -0.915915],
                [0.340583, 0.823984, -0.452829],
                [0.404599, 0.877445, 0.257661],
                [0.440576, 0.412017, 0.797581],
                [-0.426648, 0.300898, -0.852896],
                [-0.37152, 0.842631, -0.389802],
                [-0.306974, 0.896079, 0.320641],
                [-0.271745, 0.43067, 0.860627],
                [0.0, 0.0, 0.0],
                [0.89801, 0.36058, 0.252109],
                [-0.88463, 0.351859, -0.305981],
                [-0.820113, 0.405285, 0.40393],
                [0.834013, 0.3071, -0.458381],
                [0.285365, 0.282254, -0.915915],
                [0.340583, 0.823984, -0.452829],
                [0.404599, 0.877445, 0.257661],
                [0.440576, 0.412017, 0.797581],
                [-0.426648, 0.300898, -0.852896],
                [-0.37152, 0.842631, -0.389802],
                [-0.306974, 0.896079, 0.320641],
                [-0.271745, 0.43067, 0.860627],
            ]
        )
    elif n_directions == 1:
        g = np.array([[1, 0, 0]])
    else:
        print(f"Number of directions {n_directions} not implemented. Using 3 orthogonal directions instead.")
        g = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    if insert_b0s_at:
        if insert_b0s_at > n_directions:
            insert_b0s_at = n_directions

        positions = [i for i in range(0, n_directions, insert_b0s_at)]
        g = np.insert(g, positions, np.array([0, 0, 0]), axis=0)

    return g

