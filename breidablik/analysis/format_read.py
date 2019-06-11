from breidablik.analysis import tools
import numpy as np

def pixel_format(data, wavelength, center = 670.9659, lower = 0.4, upper = 0.4, ftype = 'flux'):
    """Changes the data from read into a machine learning format. This function is for machine learning over pixels.

    Parameters
    ----------
    data : dict
        Flux data from the read functions. The outermost keys are the stellar parameters for the models. The next keys are the lithium abundances. The innermost keys are 'flux' which retreives the NLTE flux or 'fluxl' which retreives the LTE flux. All data must be located at the same wavelength points.
    wavelength : List[Real] or 1darray
        The wavelengths that correspond to the data. From read.get_wavelengths().
    center : Real, optional
        The center of the wavelengths where the cut should be taken, in the same units as the wavelength. The 3 lithium lines are centered at 610.5298, 670.9659, and 812.8606 nm in the Balder results.
    upper : Real, optional
        The amount to go above the center when taking the cut, in the same units as the wavelength.
    lower : Real, optional
        The amount to go below the center when taking the cut, in the same units as the wavelength.
    ftype : str, optional
        Which type of flux to convert from the data. Accepted options are: 'flux' for NLTE or 'fluxl' for LTE.

    Returns
    -------
    Xy : tuple of 2darrays
        The X and y data sets in the form (X, y). X contains [num of objects x num of parameters], and y contains [num of objects x num of pixels].
    """

    y = []
    X = []
    for row in list(data): # stellar parameters
        t, g, m = row
        model = data[t, g, m]
        ys = [tools.cut(wavelength, model[li][ftype], center = center, upper = upper, lower = lower)[1] for li in list(model)] # lithium abundances
        X.extend([[t,g,m,li] for li in list(model)])
        y.extend(ys)

    Xy = (np.array(X), np.array(y))
    return Xy

def rew_format(data, wavelength, predict = 'rew', center = 670.9659, upper = 10, lower = 10, ftype = 'flux', num = 10000):
    """Changes the data from read into a machine learning format. This function is for machine learning over REWs.

    Parameters
    ----------
    data : dict
        Flux data from the read functions. The outermost keys are the stellar parameters for the models. The next keys are the lithium abundances. The innermost keys are 'flux' which retreives the NLTE flux or 'fluxl' which retreives the LTE flux. All data must be located at the same wavelength points.
    wavelength : List[Real] or 1darray
        The wavelengths that correspond to the data. From read.get_wavelengths().
    predict : str, optional
        Determines what varlue is placed in the y data. Accepted options are 'rew' and 'li'.
    center : Real, optional
        The center of the wavelengths where the cut should be taken, in the same units as the wavelength. The 3 lithium lines are centered at 610.5298, 670.9659, and 812.8606 nm in the Balder results.
    upper : Real, optional
        The amount to go above the center when taking the cut, in the same units as the wavelength.
    lower : Real, optional
        The amount to go below the center when taking the cut, in the same units as the wavelength.
    ftype : str, optional
        Which type of flux to convert from the data. Possible options are: 'flux' for NLTE or 'fluxl' for LTE.
    num : Int, optional
        The number of points in the interpolation. Before calculating the REW, the line profile is interpolated to finer wavelength points.

    Returns
    -------
    Xy : tuple of 2darrays
        The X and y data sets in the form (X, y). X contains [num of objects x num of parameters], and y contains [num of objects].
    """

    if predict not in ('rew', 'li'):
        raise ValueError('Unaccepted input for predict, detected input: {}'.format(predict))

    X = []
    y = []
    for row in list(data): # stellar parameters
        t, g, m = row
        model = data[t, g, m]
        for li in list(model): # lithium abundance
            rew = tools.rew(wavelength, model[li][ftype], center = center, upper = upper, lower = lower, num = num)
            if predict == 'rew':
                X.append([t, g, m, li])
                y.append(rew)
            elif predict == 'li':
                X.append([t, g, m, rew])
                y.append(li)
    Xy = (np.array(X), np.array(y))
    return Xy
