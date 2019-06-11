import numpy as np
from scipy.interpolate import CubicSpline

def cut_wavelength(wavelength, center = 670.9659, upper = 10, lower = 10):
    """Cuts the wavelength returns the values between center - lower and center + upper. Useful for plotting mostly because many functions return a cut line profile but not cut wavelength.

    Parameters
    ----------
    wavelength : List[Real] or 1darray
        Input wavelengths. Needs to be monotonically increasing.
    center : Real, optional
        The center of the wavelengths where the cut should be taken, in the same units as the wavelength. The 3 lithium lines are centered at 610.5298, 670.9659, and 812.8606 nm in the Balder results.
    upper : Positive Real, optional
        The amount to go above the center when taking the cut, in the same units as the wavelength.
    lower : Positive Real, optional
        The amount to go below the center when taking the cut, in the same units as the wavelength.

    Returns
    -------
    wl_cut : 2darray
        Cut wavelengths.
    """

    wavelength = np.array(wavelength)

    low = center - lower
    high = center + upper
    wl_cut = wavelength[(low <= wavelength) & (high >= wavelength)]
    return wl_cut

def cut(wavelength, line_profile, center = 670.9659, upper = 10, lower = 10):
    """Cuts the wavelength and line profile and returns the values between center - lower and center + upper.

    Parameters
    ----------
    wavelength : List[Real] or 1darray
        Input wavelengths. Needs to be monotonically increasing.
    line_profile : List[Real] or 1darray
        Input line profile.
    center : Real, optional
        The center of the wavelengths where the cut should be taken, in the same units as the wavelength. The 3 lithium lines are centered at 610.5298, 670.9659, and 812.8606 nm in the Balder results.
    upper : Positive Real, optional
        The amount to go above the center when taking the cut, in the same units as the wavelength.
    lower : Positive Real, optional
        The amount to go below the center when taking the cut, in the same units as the wavelength.

    Returns
    -------
    cut_data : 2darray
        Cut wavelengths and line profiles.
    """

    wavelength = np.array(wavelength)
    line_profile = np.array(line_profile)

    low = center - lower
    high = center + upper
    mask = (low <= wavelength) & (high >= wavelength)
    wl_cut = wavelength[mask]
    line_cut = line_profile[mask]
    cut_data = np.array([wl_cut, line_cut])
    return cut_data

def rew(wavelength, line_profile, center = 670.9659, upper = 10, lower = 10, num = 10000):
    """Calculates the reduced equivlanet width (REW) of the line profile between center - lower and center + upper.

    Parameters
    ----------
    wavelength : List[Real] or 1darray
        Input wavelengths. Needs to be monotonically increasing.
    line_profile : List[Real] or 1darray
        Input line profile.
    center : Real, optional
        The center of the wavelengths where the REW should be calculated from, in the same units as the wavelength. The 3 lithium lines are centered at 610.5298, 670.9659, and 812.8606 nm in the Balder results.
    upper : Positive Real, optional
        The amount to go above the center when taking calculating the REW, in the same units as the wavelength.
    lower : Positive Real, optional
        The amount to go below the center when calculating the REW, in the same units as the wavelength.
    num : Int, optional
        The number of points in the interpolation. Before calculating the REW, the line profile is interpolated to finer wavelength points.

    Returns
    -------
    rew : float
        The REW.
    """

    wl_cut, line_cut = cut(wavelength, line_profile, center, upper = upper, lower = lower)
    fine_wl = np.linspace(wl_cut[0], wl_cut[-1], num) # includes the last point
    fine_line = CubicSpline(wl_cut, line_cut)(fine_wl)
    area = np.trapz(1-fine_line, fine_wl)
    rew = np.log10(area/center)
    return rew
