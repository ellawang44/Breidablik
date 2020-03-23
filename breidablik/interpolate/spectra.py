from breidablik.analysis import read
from breidablik.analysis import tools
from breidablik.interpolate.grid_check import _grid_check
from breidablik.interpolate.scalar import Scalar
import joblib
import numpy as np
import os
from pathlib import Path
from scipy.interpolate import CubicSpline
import warnings

_base_path = Path(__file__).parent

class Spectra:
    """Interpolation class for spectra. Used to interpolate between the stellar parameters. Can find the abundance of an input flux given the stellar parameters. Can also predict a flux from the stellar parameters and abundance.
    """

    def __init__(self, model_path = None, scalar_path = None, save_num = 1):
        """Initialise the data by loading the pickled models and scalar.

        Parameters
        ----------
        model_path : str, optional
            The path to the model to be used to predict the flux. By default, this path points to ``models/kri`` in ``breidablik``.
        scalar_path : str, optional
            The path to the scalar corresponding to the model. By default, this path points to ``models/kri/scalar.npy`` in ``breidablik``.
        save_num : int
            The number of cubic spline models to save. This makes the code run faster, but takes memory. Only worth increasing if you are analysing different observations of multiple stars.
        """

        # set default paths
        model_path = model_path or _base_path.parent / 'models/kri'
        scalar_path = scalar_path or _base_path.parent / 'models/kri/scalar.npy'
        # load models
        self.scalar = Scalar()
        self.scalar.load(scalar_path)
        self.models = joblib.load(os.path.join(model_path, 'kri.pkl'))
        self.relative_error = np.load(os.path.join(model_path, 'relative_err.npy'), allow_pickle = False)
        self.cut_models = None
        self.saved_cspline = {}
        self.saved_cut_cspline = {}

    def find_abund(self, wavelength, flux, flux_err, eff_t, surf_g, met, accuracy = 1e-4, method = 'bayes', min_abund = -0.5, max_abund = 4, initial_accuracy = 1e-1, abunds = None, prior = None):
        """Finds the abundance of the spectrum.

        Parameters
        ----------
        wavelength : List[Real] or 1darray
            The wavelengths that correspond to the flux. Needs to be monotonically increasing.
        flux : List[Real] or 1darray
            The flux that the abunance will be found for. Needs to be the same length as wavelength.
        flux_err : List[Real] or 1darray
            The error in each flux point. Needs to be the same length as wavelength.
        eff_t : Real
            The effective temperature of the star.
        surf_g : Real
            The log surface gravity of the star.
        met : Real
            The metallicity of the star.
        accuracy : Real, optional
            The decimal place you want the result to be accurate to.
        method : str, optional
            The method of finding the abundance. Accepted methods are: 'bayes' and 'chisq'.
        min_abund : Real, optional
            The minimum abundance that the algorithm should search to.
        max_abund : Real, optional
            The maximum abundance that the algorithm should search to.
        initial_accuracy : Real, optional
            The initial accuracy that the algorithm starts searching through. If 'bayes' is returning warnings try decreasing the initial accuracy. Note that this does make the algorithm run slower.
        abunds : List[Real], 1darray, optional
            Determine the abundances you want the probability caculated over. Overrides the min_abund and max_abund parameters. This parameter is ignored if prior is not set. Only used if method is 'bayes'.
        prior : List[Real], 1darray, optional
            Set the prior to the abundances specified via abunds. This parameter is ignored if abunds is not set. Only used if method is 'bayes'. If method is 'bayes' but no prior is set, uses uniform prior. Needs to be the same length as abunds.

        Returns
        -------
        (abundance, error) : (float, list)
            The abundance that matches best with the input flux and the error value associated with the abundance. The error list has 2 values, left is the negative error, right is the positive error. The error can be None if the method is 'chisq' or if abunds and prior were given but error cannot be calculated.
        """

        # change it all to numpy arrays
        wavelength = np.array(wavelength)
        flux = np.array(flux)
        flux_err = np.array(flux_err)

        # make sure wavelength is monotonically increasing
        if not np.all(wavelength[1:] > wavelength[:-1]):
            raise ValueError('Wavelength needs to be monotonically increasing.')
        # make sure wavelength, flux, flux_err all are 1D arrays with the same length
        if not (wavelength.shape == flux.shape) and (flux.shape == flux_err.shape) and (len(wavelength.shape) == 1):
            raise ValueError('Wavelength, flux, and flux_err needs to 1D array and have the same shape. Detected shapes: wavelength {}, flux {}, flux_err {}'.format(wavelength.shape, flux.shape, flux_err.shape))
        # make sure method is an accepted method
        if not ((method == 'bayes') or (method == 'chisq')):
            raise ValueError('Invalid method, detected input: {}'.format(method))
        # make sure min abund < max abund
        if min_abund > max_abund:
            raise ValueError('minimum abundance is bigger than maximum abundance, detected input: min_abund = {}, max_abund = {}'.format(min_abund, max_abund))

        # warn if prior/abunds is set but method is not bayes
        if (method == 'chisq') and ((abunds is not None) or (prior is not None)):
            warnings.warn('method is set to chisq but abunds or prior is not None, ignoring the abunds and prior inputs.')
        # warn if prior is defined but abunds is not or vice versa.
        if (abunds is not None) and (prior is None):
            warnings.warn('abunds is defined but prior is not. Both needs to be defined or else abunds is ignored.')
        if (prior is not None) and (abunds is None):
            warnings.warn('prior is defined but abunds is not. Both needs to be defined or else prior is ignored.')
        # warn if stellar parameters are too far outside the edge of the grid
        _grid_check(eff_t, surf_g, met)

        # makes things go vroom vroom. Predictions take a long time
        lower_wl = min(wavelength)
        upper_wl = max(wavelength)
        balder_wl = read.get_wavelengths()
        mask = (lower_wl <= balder_wl) & (balder_wl <= upper_wl)
        if (mask == False).all(): # check if the input wavelength is encompassed by our data
            raise ValueError('Input wavelength does not overlap with the model data. Minimum input wavelength : {}, maximum input wavelength, minimum model data wavelength, maximum model data wavelength: {}'.format(wavelength[0], wavelength[-1], balder_wl[0], balder_wl[-1]))
        self.cut_wl = balder_wl[mask]
        inds = np.where(mask == True)[0]
        ind_l = inds[0]
        ind_u = inds[-1]+1
        self.cut_models = {}
        for abund in list(self.models):
            self.cut_models[abund] = self.models[abund][ind_l:ind_u]

        err = None
        if method == 'bayes':
            # if there is prior and abunds
            if (prior is not None) and (abunds is not None):
                prior = np.array(prior)
                abunds = np.array(abunds)
                if (abunds.shape != prior.shape) or (len(abunds.shape) != 1) or (len(prior.shape) != 1): # each abundance needs a prior
                    raise ValueError('The length of abundance and prior should be the same, they should also be 1D arrays, detected shape: abunds: {}, prior: {}'.format(abunds.shape, prior.shape))
                probs = self._bayesian_inference(wavelength, flux, flux_err, eff_t, surf_g, met, abunds, prior = prior)
                # if we have the whole pdf
                tolerance = 1e-5
                if (probs > tolerance).any() and (probs[0] < tolerance) and (probs[-1] < tolerance):
                    abundance, err = self._calc_bayes_err(abunds, probs)
                # if we only have part of the pdf
                else:
                    warnings.warn('Only the abundance is returned. The probability distribution function is not fully covered by the input abunds and prior, therefore, the error in abundance cannot be calculated. Increase the range of the input abunds to cover the rest of the probability distribution function to get an error estimate. Minimum input abunds = {}, corresponding probability = {}; maximum input abunds = {}, corresponding probability = {}.'.format(abunds[0], probs[0], abunds[-1], probs[-1]))
                    abundance = abunds[np.argmax(probs)]
            # if there is no prior
            else:
                abundance, err = self._coarse_search(wavelength, flux, flux_err, eff_t, surf_g, met, min_abund = min_abund, max_abund = max_abund, accuracy = accuracy)
        else:
            abundance = self._window_search(wavelength, flux, flux_err, eff_t, surf_g, met, accuracy = accuracy, min_abund = min_abund, max_abund = max_abund, initial_accuracy = initial_accuracy)

        # warn if predicted Li is outside of grid
        if (abundance < -0.5) or (abundance > 4):
            warnings.warn('Predicted lithium abundance is outside of the grid, results are extrapolated and may not be reliable.')

        return (abundance, err)

    def _window_search(self, wavelength, flux, flux_err, eff_t, surf_g, met, accuracy = 1e-4, min_abund = -0.5, max_abund = 4, initial_accuracy = 1e-1):
        """An algorithm that finds the minimum - variation of ternary search. Used to make finding the abundances faster. Only for chisq method.
        """

        current_accuracy = initial_accuracy
        once = False
        while abs(current_accuracy - accuracy) > accuracy:
            abunds = np.arange(min_abund, max_abund + current_accuracy/2, current_accuracy)
            # find the index of the best abundance
            res_sq = self._chisq(wavelength, flux, flux_err, abunds, eff_t, surf_g, met)
            best_abund_index = np.argmin(res_sq)

            # update search window
            if best_abund_index == 0: # leftmost value
                max_abund = abunds[1]
            elif best_abund_index == (len(abunds) - 1): # rightmost value
                min_abund = abunds[-2]
            else:
                min_abund = abunds[best_abund_index - 1]
                max_abund = abunds[best_abund_index + 1]
            current_accuracy /= 10
        abundance = abunds[best_abund_index]
        return abundance

    def _coarse_search(self, wavelength, flux, flux_err, eff_t, surf_g, met, min_abund, max_abund, accuracy = 1e-4, initial_accuracy = 1e-1):

        # get initial probabilities
        abunds = np.arange(min_abund, max_abund + initial_accuracy/2, initial_accuracy)
        probs = self._bayesian_inference(wavelength, flux, flux_err, eff_t, surf_g, met, abunds)

        tolerance = 1e-5
        # not a flat line
        if (probs > tolerance).any():
            # if we have found the pdf
            if (probs[0] < tolerance) and (probs[-1] < tolerance):
                # find smallest range of abundances we can feed into bayesian inference
                max_ind = np.argmax(probs)
                left = probs[:max_ind]
                l_ind = len(left[left<=tolerance]) - 1
                right = probs[max_ind+1:]
                r_ind = len(probs) - len(right[right<=tolerance])
                # find pdf
                fine_abunds = np.arange(abunds[l_ind], abunds[r_ind] + accuracy/2, accuracy)
                fine_probs = self._bayesian_inference(wavelength, flux, flux_err, eff_t, surf_g, met, fine_abunds)
                return self._calc_bayes_err(fine_abunds, fine_probs)
            # if we haven't found the pdf
            half = (max_abund - min_abund)/2
            # if we have found left half the pdf
            if (probs[0] < tolerance) and (probs[-1] > tolerance):
                l_half = 0
                r_half = half
            # if we have found right half of the pdf
            elif (probs[0] > tolerance) and (probs[-1] < tolerance):
                l_half = half
                r_half = 0
            # if we have found the peak but not the edges
            else:
                l_half = half
                r_half = half
            return self._coarse_search(wavelength, flux, flux_err, eff_t, surf_g, met, min_abund - l_half, max_abund + r_half, accuracy = accuracy, initial_accuracy = initial_accuracy)
        # if we have not found the pdf
        else:
            raise ValueError('Either the actual abundance of the spectra is not between the min_abund and max_abund, or the initial_accuracy value is too large. Try finding the actual abundance through the chisq method, if this abundance is between the min_abund and max_abund, then make the initial_accuracy smaller. Detected inputs: min_abund = {}, max_abund = {}, initial_accuracy = {}'.format(min_abund, max_abund, initial_accuracy))

    def _calc_bayes_err(self, abunds, probs):
        """Find the best abundance through bayesian inference and also calculate the error associated with this measurement.
        """

        # find best abundance
        best_abund = abunds[np.argmax(probs)]
        # find errors
        cumsum = np.cumsum(probs)
        cdf = cumsum/cumsum[-1]
        l_sig = 0.5 - 0.34
        r_sig = 0.5 + 0.34
        l_abund = abunds[np.argmin(np.abs(cdf - l_sig))]
        r_abund = abunds[np.argmin(np.abs(cdf - r_sig))]
        l_err = l_abund - best_abund
        r_err = r_abund - best_abund
        return (best_abund, [l_err, r_err])

    def _bayesian_inference(self, wavelength, flux, flux_err, eff_t, surf_g, met, abunds, prior = None):
        """Use Bayesian optimisation to find the probability of the model (abundance) given data.
        """

        probs = []
        guess_fluxes = self._predict_flux(eff_t, surf_g, met, abunds)
        if prior is None:
            prior = np.full(len(guess_fluxes), 1)

        for g_flux, p, a in zip(guess_fluxes, prior, abunds):
            model = CubicSpline(self.cut_wl, g_flux)
            probability = self._p_model_given_data(wavelength, flux, flux_err, p, model)
            probs.append(probability)
        probs = np.array(probs)
        exp_probs = np.exp(probs - max(probs))
        width = abunds[1] - abunds[0]
        rescaled_probs = exp_probs/(np.sum(exp_probs)*width) # the probabilities were logged in _p_model_given_data, normalise before taking the exponential to prevent overflow
        return rescaled_probs

    def _p_model_given_data(self, xdata, ydata, sigma, prior, model):
        """Bayesian inference - probability of data given model. The prior is for the input model.
        """

        xdata = np.array(xdata)
        ydata = np.array(ydata)
        p_data_given_model = 1/np.sqrt(2*np.pi*sigma**2)*np.e**(-(ydata - model(xdata))**2/(2*sigma**2)) + 1e-5 # sometimes the probability is 0
        posterior = np.sum(np.log(p_data_given_model)) + np.log(prior + 1e-5) # we don't normalise the probabilities, so to avoid overflow we log
        return posterior

    def _chisq(self, wavelength, flux, flux_err, abunds, eff_t, surf_g, met):
        """Calculates the least squares of the input abundances.
        """

        guess_flux = self._predict_flux(eff_t, surf_g, met, abunds)
        int_flux = [CubicSpline(self.cut_wl, g_flux)(wavelength) for g_flux in guess_flux]
        diff = np.array(int_flux) - np.array(flux)
        chi_square_ind = np.square(diff)/np.square(flux_err)
        chi_sq = np.sum(chi_square_ind, axis = 1)
        return chi_sq

    def predict_flux(self, eff_t, surf_g, met, abundance):
        """Predicts the flux for the input stellar parameters and abundances.

        Parameters
        ----------
        eff_t : Real
            The effective temperature of the star.
        surf_g : Real
            The log surface gravity of the star.
        met : Real
            The metallicity of the star.
        abundance : Real
            The lithium abundance of the star.

        Returns
        -------
        predicted : 1darray
            The predicted flux given the input stellar parameters and lithium abundance.
        """

        # check the input stellar parameters and abundance
        if not ((np.array(eff_t).shape == ()) and (np.array(surf_g).shape == ()) and (np.array(met).shape == ()) and (np.array(abundance).shape == ())):
            raise ValueError('The input effective temperature, surface gravity, metallicity, or abundance is not in the right format, they all need to be scalar numbers, detected inputs: eff_t = {}, surf_g = {}, met = {}, and abund = {}'.format(eff_t, surf_g, met, abundance))
        # warn if stellar parameters are too far outside the edge of the grid
        _grid_check(eff_t, surf_g, met)
        # warn if abundance is outside the grid range
        if (abundance < -0.5) or (abundance > 4):
            warnings.warn('Input abundance is outside of the grid, results are extrapolated and may not be reliable.')

        return self._predict_flux(eff_t, surf_g, met, [abundance], user_call = True)[0]

    def _predict_flux(self, eff_t, surf_g, met, abundance, user_call = False):
        """Same as predict_flux. This is the hidden version without asserts for improved performance. The flux is only predicted in the region of the cut_models (where cut_models are determined by the input observed spectrum).

        Also, you can call this version with a list of abundances. It's faster if you call this function with a list of abundances vs calling the user visible one with a for-loop over abundances. Vroom vroom.
        """

        if (self.cut_models is not None) and (not user_call): # only predict a range of models
            saved_cspline = self.saved_cut_cspline
            cut = True
        else: # predict all models
            saved_cspline = self.saved_cspline
            cut = False

        if (eff_t, surf_g, met) in list(saved_cspline): # if the cubic spline is already saved
            splines = saved_cspline[eff_t, surf_g, met]
        else: # if new stellar parameters
            splines = self._create_cspline(eff_t, surf_g, met, cut = cut)

        # evaluate the splines at the desired abundances
        predicted = np.array([s(abundance) for s in splines]).T
        predicted = 1 + self.relative_error - 10**predicted

        return predicted

    def _create_cspline(self, eff_t, surf_g, met, cut = True):
        """Creates a cubic spline for each abundance for the input stellar parameters. The pixels for which these cubic splines are created over is determined by cut.
        """

        # get correct models and cubic splines
        if cut:
            models = self.cut_models
            cspline = self.saved_cut_cspline
        else:
            models = self.models
            cspline = self.saved_cspline

        # transform input
        transformed_input = self.scalar.transform([[eff_t, surf_g, met]])[0]

        # evaluating the models for the input stellar parameters
        abunds = list(models)
        grid_spec = []
        for a in abunds:
            spec = []
            for m in models[a]:
                try: # can't train on outputs that are all the same
                    point = m.execute('points', *transformed_input, backend = 'loop')[0][0]
                except AttributeError: # append the value in that case
                    point = m
                spec.append(point)
            grid_spec.append(spec)
        grid_spec = np.array(grid_spec).T

        # create splines over abundances
        splines = []
        for spec in grid_spec:
            splines.append(CubicSpline(abunds, spec))
        cspline[eff_t, surf_g, met] = splines

        return splines
