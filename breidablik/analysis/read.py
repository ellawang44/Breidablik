## even though D == 'marcs' appears in this file. This data is not public. The functions in this does read in the private marcs data grid into the same format as the other grids though.

import math
import numpy as np
import os
from pathlib import Path
import random
import warnings

_base_path = Path(__file__).parent

def _name_add(name):
    """Add '+' to the front of positive values. Also keeps the '-' in front of negative values.
    """

    name = str(float(name))
    if name[0] != '-':
        name = '+' + name
    return name

def _get_dimension_path(D = '3D', a = 1.5, v = 1, data_path = _base_path.parent / 'Balder'):
    """Gets the path to the specified dimension.
    """

    if D == '3D' or D == 'marcs':
        D_path = os.path.join(data_path, D)
    elif D == '1D':
        D_path = os.path.join(data_path, '1D_a' + str(float(a)) + '_v' + str(float(v)))
    return D_path

def _closest_temp(temperature, surf_g, met, D = '3D', a = 1.5, v = 1, data_path = _base_path.parent / 'Balder'):
    """Finds the temperature closest to the input temperature given that the surface gravity and metallicity values match. The output will always be accurate to 2 decimal places (the output will match the names of the models).
    """

    D_path = _get_dimension_path(D = D, a = a, v = v, data_path = data_path)
    all_models = [f for f in os.listdir(D_path)]
    possible_temps = list(set([float(mod[1:8]) for mod in all_models if float(mod[9:13]) == surf_g and float(mod[14:]) == met])) # remove duplicates
    temp = str(possible_temps[np.argmin(np.abs(np.array(possible_temps) - temperature))])
    while len(temp) < 7:
        temp = temp + '0'
    if D != 'marcs':
        if abs(float(temp) - temperature) > 250:
            warnings.warn('closest temp is more than 250 K away from the input temperature. Returned closest model. The temperature snapped to: {}'.format(temp))
    return temp

def _get_model_path(eff_t, surf_g, met, D = '3D', a = 1.5, v = 1, data_path = _base_path.parent / 'Balder'):
    """Returns path of model given the stellar parameters. Will return closest model with the closest temp that matches the input temp. surf_g and met have to be exact.
    """

    D_path = _get_dimension_path(D = D, a = a, v = v, data_path = data_path)
    c_temp = _closest_temp(eff_t, surf_g, met, D = D, a = a, v = v, data_path = data_path)
    if D == 'marcs':
        if int(eff_t) != eff_t:
            raise ValueError('Decimal effective temperatures not accepted for the MARCS grid.')
        model_path = os.path.join(D_path, 't' + str(int(eff_t)) + '.00g' + _name_add(surf_g) + 'm' + _name_add(met))
    else:
        model_path = os.path.join(D_path, 't' + c_temp + 'g' + _name_add(surf_g) + 'm' + _name_add(met))
    return model_path

def read(eff_t, surf_g, met, abund, D = '3D', a = 1.5, v = 1, data_path = None):
    """Reads in the flux data for the specified dimension and stellar parameters.

    Parameters
    ----------
    eff_t : Real
        The temperature closest to the real temperature of the model. This input should to be less than 250 K away from the real temperature of the model. If the input temperature is more than 250 K from the real temperature, a warning will be raised, consult the warnings module to see how to change these warnings.
    surf_g : Real
        The surface gravity of the stellar model.
    met : Real
        The metallicity of the stellar model.
    abund : Real
        The lithium abundance of the stellar model.
    D : str, optional
        The dimension of the model. Accepted values are either '1D' or '3D'.
    a : Real or str, optional
        The mixing length parameter. Accepted values are 1, 1.5, or 2. The input can be expressed in any data type that can be converted into a floating point number.
    v : Real or str, optional
        The microturbulence parameter. Accepted values are 0, 1, or 2. The input can be expressed in any data type that can be converted into a floating point number.
    data_path : str, optional
        The folder that the data is stored in. By default, this path points to ``Balder`` in ``breidablik``.

    Returns
    -------
    flux_data : dict
        The NLTE and LTE flux of the specified dimension and stellar model. 'flux' contains the NLTE flux, and 'fluxl' contains the LTE flux.
    """

    # set default data_path
    data_path = data_path or _base_path.parent / 'Balder'

    model_path = _get_model_path(eff_t, surf_g, met, D = D, a = a, v = v, data_path = data_path)
    abund_path = os.path.join(model_path, 'a' + _name_add(abund) + '.dat')
    flux, fluxl = np.loadtxt(abund_path, unpack = True)
    flux_data = {'flux': flux, 'fluxl': fluxl}
    return flux_data

def get_wavelengths(data_path = None):
    """Returns the wavelengths for the flux data.

    Parameters
    ----------
    data_path : str, optional
        The folder that the data is stored in. By default, this path points to ``Balder`` in ``breidablik``.

    Returns
    -------
    wl : 1darray
        The wavelengths for the flux data in nm.
    """

    # set default data_path
    data_path = data_path or _base_path.parent / 'Balder'

    wl = np.loadtxt(os.path.join(data_path, 'wavelengths.dat'))

    # check monotonic
    if not np.all(wl[1:] > wl[:-1]):
        raise ValueError('Wavelengths needs to be monotonically increasing. The file containing wavelengths might be corrupted.')

    return wl

def read_all_abund(eff_t, surf_g, met, D = '3D', a = 1.5, v = 1, data_path = None):
    """Reads in the fluxes for all lithium abundances for a dimension and stellar model.

    Parameters
    ----------
    eff_t : Real
        The temperature closest to the real temperature of the model. This input should to be less than 250 K away from the real temperature of the model. If the input temperature is more than 250 K from the real temperature, a ValueError will be raised, consult the warnings module to see how to change these warnings.
    surf_g : Real
        The surface gravity of the stellar model.
    met : Real
        The metallicity of the stellar model.
    D : str, optional
        The dimension of the model. Accepted values are either '1D' or '3D'.
    a : Real or str, optional
        The mixing length parameter. Accepted values are 1, 1.5, or 2. The input can be expressed in any data type that can be converted into a floating point number.
    v : Real or str, optional
        The microturbulence parameter. Accepted values are 0, 1, or 2. The input can be expressed in any data type that can be converted into a floating point number.
    data_path : str, optional
        The folder that the data is stored in. By default, this path points to ``Balder`` in ``breidablik``.

    Returns
    -------
    data : dict of dict
        The NLTE and LTE fluxes for all lithium abundances of a model. The outermost keys are the lithium abundances. The inner keys are 'flux' (retreives the NLTE flux) and 'fluxl' (retreives the LTE flux).
    """

    # set default data_path
    data_path = data_path or _base_path.parent / 'Balder'

    model_path = _get_model_path(eff_t, surf_g, met, D = D, a = a, v = v, data_path = data_path)
    abunds = sorted([float(ab[1:-4]) for ab in os.listdir(model_path)])
    data = {}
    for abund in abunds:
        data[abund] = read(eff_t, surf_g, met, abund, D = D, a = a, v = v, data_path = data_path)
    return data

def read_all(D = '3D', a = 1.5, v = 1, data_path = None):
    """Read in all the data for some dimension.

    Parameters
    ----------
    D : str, optional
        The dimension of the model. Accepted values are either '1D' or '3D'.
    a : Real or str, optional
        The mixing length parameter. Accepted values are 1, 1.5, or 2. The input can be expressed in any data type that can be converted into a floating point number.
    v : Real or str, optional
        The microturbulence parameter. Accepted values are 0, 1, or 2. The input can be expressed in any data type that can be converted into a floating point number.
    data_path : str, optional
        The folder that the data is stored in. By default, this path points to ``Balder`` in ``breidablik``.

    Returns
    -------
    data : dict
        All the data stored in the specified dimension. The outermost keys are the stellar parameters for the models. The next keys are the lithium abundances. The innermost keys are 'flux' which retreives the NLTE flux or 'fluxl' which retreives the LTE flux. If split is used then the outermost keys are the split sets (either 'train' or 'test').
    """

    # set default data_path
    data_path = data_path or _base_path.parent / 'Balder'

    D_path = _get_dimension_path(D = D, a = a, v = v, data_path = data_path)
    models = os.listdir(D_path)
    data = {}
    for model in models:
        eff_t = float(model[1:8])
        surf_g = float(model[9:13])
        met = float(model[13+1:])
        data[eff_t, surf_g, met] = read_all_abund(eff_t, surf_g, met, D = D, a = a, v = v, data_path = data_path)
    return data

def split(data, split, seed = None):
    """Split up the data into a test and training set.

    Parameters
    ----------
    data : dict
        The read in data. The outermost keys are the stellar parameters for the models. The next keys are the lithium abundances. The innermost keys are 'flux' which retreives the NLTE flux or 'fluxl' which retreives the LTE flux.
    split : float, optional
        Specifies the ratio of test data to all data. No split is done if None.
    seed : int, optional
        Seed used to generate random numbers. Set for reproducibility of results.

    Returns
    -------
    split_sets : dict
        The split data. Outermost keys are the split sets (either 'train' or 'test'), the inner keys remain the same as the input data.
    """

    # shuffle based upon model
    models = sorted(list(data))
    random.seed(a = seed)
    random.shuffle(models)

    # figure out the stellar parameters that go into each set
    split_ind = math.floor(len(models)*split)
    train = list(models[split_ind:])
    test = list(models[:split_ind])

    # create the sets
    train_data = {}
    for (t, g, m) in train:
        train_data[t, g, m] = data[t, g, m]
    test_data = {}
    for (t, g, m) in test:
        test_data[t, g, m] = data[t, g, m]
    split_sets = {'train': train_data, 'test': test_data}
    return split_sets

def kfolds(data, k = 10, seed = None):
    """Split the data up into k-folds.

    Parameters
    ----------
    data : dict
        The read in data. The outermost keys are the stellar parameters for the models. The next keys are the lithium abundances. The innermost keys are 'flux' which retreives the NLTE flux or 'fluxl' which retreives the LTE flux.
    k : int, optional
        Specifies the number of folds generated. If the data doesn't divide fully along the folds, then a best split is done.
    seed : int, optional
        Seed used to generate random numbers. Set for reproducibility of results.

    Returns
    -------
    folds : dict
        The k-fold data. Outermost keys are the number of the sets (from 0 up to but not including k), the inner keys remain the same as the input data.
    """

    # shuffle based upon model
    models = sorted(list(data))
    random.seed(a = seed)
    random.shuffle(models)

    # figure out the stellar parameters that go into each set
    inds = list(map(int, np.round(np.linspace(0, len(models), k+1))))
    split_set = zip(inds[:-1], inds[1:])

    # create the folds
    folds = {}
    for i, (a, b) in enumerate(split_set):
        new_fold = {}
        for t, g, m in models[a:b]:
            new_fold[t, g, m] = data[t, g, m]
        folds[i] = new_fold
    return folds
