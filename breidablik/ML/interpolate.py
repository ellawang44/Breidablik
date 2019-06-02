from breidablik.analysis import read
from breidablik.analysis import tools
import joblib
import numpy as np
from pathlib import Path
from scipy.interpolate import CubicSpline
import warnings

_base_path = Path(__file__).parent

class Interpolate:
    """Interpolation class. Used to interpolate between the stellar parameters. Can find the abundance of an input flux given the stellar parameters. Can also predict a flux from the stellar parameters and abundance.
    """

    def __init__(self, model_path = _base_path.parent / 'models/mlp.pkl', scalar_path = _base_path.parent / 'models/mlp_scalar.pkl', rew_model_path = _base_path.parent / 'models/rew_3D.pkl', rew_scalar_path = _base_path.parent / 'models/rew_3D_scalar.pkl'):
        """Initialise the data by reading the pickled models and scalar.

        Parameters
        ----------
        model_path : str, optional
            The path to the model to be used to predict the flux.
        scalar_path : str, optional
            The path to the scalar corresponding to the model.
        """

        self.scalar = joblib.load(scalar_path)
        self.models = joblib.load(model_path)
        self.rew_models = [None, joblib.load(rew_model_path), None]
        self.rew_scalars = [None, joblib.load(rew_scalar_path), None]
        self.relative_error = 1e-14
        self.cut_models = None

    def _grid_check(self, eff_t, surf_g, met):
        """Check if the stellar parameters are too far outside the edge of the grid
        """

        with open(_base_path.parent / 'grid_snapshot.txt', 'r') as f:
            t_step, m_step = np.float_(f.readline().split())
            grid = np.loadtxt(f)
        input_model = np.array([eff_t*t_step, surf_g, met*m_step])
        input_tile = np.tile(input_model, (grid.shape[0], 1))
        min_dist = min(np.sqrt(np.sum(np.square(grid - input_tile), axis = 1)))
        if min_dist > np.sqrt(3*0.25**2):
            warnings.warn('Input stellar parameters are outside of the grid, results are extrapolated and may not be reliable.')

    def find_abund_rew(self, eff_t, surf_g, met, rew, center = 6709.659):
        """Find the abundance based on the stellar parameters and measured reduced equivalent width.

        rew : Real
            The reduced equivalent width for the lithium line at 670.9 nm.
        eff_t : Real
            The effective temperature of the star.
        surf_g : Real
            The log surface gravity of the star.
        met : Real
            The metallicity of the star.
        center : Real, optional
            The center of the lithium line that the input rew corresponds to, in angstroms. The three lithium lines we model are: 6105.298, 6709.659, and 8128.606 angstroms. The input center value will snap to the closest value out of those 3.
        """

        # TODO: add working with different line centers

        # check the input stellar parameters and abundance
        if not ((np.array(eff_t).shape == ()) and (np.array(surf_g).shape == ()) and (np.array(met).shape == ()) and (np.array(rew).shape == ())):
            raise ValueError('The input effective temperature, surface gravity, metallicity, or abundance is not in the right format, they all need to be scalar numbers, detected inputs: eff_t = {}, surf_g = {}, met = {}, and rew = {}'.format(eff_t, surf_g, met, rew))
        # warn if stellar parameters are too far outside the edge of the grid
        self._grid_check(eff_t, surf_g, met)

        predicted_li = self._find_abund_rew(eff_t, surf_g, met, [rew], center = 6709.659)[0]

        # warn if predicted Li is outside of grid
        if (predicted_li < -0.75) or (predicted_li > 4.25):
            warnings.warn('Predicted lithium abundance is outside of the grid, results are extrapolated and may not be reliable.')

        return predicted_li

    def _find_abund_rew(self, eff_t, surf_g, met, rew, center = 6709.659):
        """Same as find_abund_rew, hidden version without grid checks so extra warnings aren't thrown. This version can be used to quickly process many rew values.
        """

        line_centers = np.array([6105.298, 6709.659, 8128.606])
        ind = np.argmin(np.abs(line_centers - center))
        scalar = self.rew_scalars[ind]
        model = self.rew_models[ind]
        transformed_input = scalar.transform([[eff_t, surf_g, met, r] for r in rew])
        predicted_li = model.predict(transformed_input)

        return predicted_li

    def find_abund(self, wavelength, flux, flux_err, eff_t, surf_g, met, accuracy = 1e-5, method = 'bayes', min_abund = -0.5, max_abund = 4, initial_accuracy = 1e-1, abunds = None, prior = None):
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
            The method of finding the abundance. Accepted methods are: 'bayes' and 'leastsq'.
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
        abundance : float
            The abundance that matches best with the input flux.
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
        if not ((method == 'bayes') or (method == 'leastsq')):
            raise ValueError('Invalid method, detected input: {}'.format(method))
        # make sure min abund < max abund
        if min_abund > max_abund:
            raise ValueError('minimum abundance is bigger than maximum abundance, detected input: min_abund = {}, max_abund = {}'.format(min_abund, max_abund))

        # warn if prior/abunds is set but method is not bayes
        if (method == 'leastsq') and ((abunds is not None) or (prior is not None)):
            warnings.warn('method is set to leastsq but abunds or prior is not None, ignoring the abunds and prior inputs.')
        # warn if prior is defined but abunds is not or vice versa.
        if (abunds is not None) and (prior is None):
            warnings.warn('abunds is defined but prior is not. Both needs to be defined or else abunds is ignored.')
        if (prior is not None) and (abunds is None):
            warnings.warn('prior is defined but abunds is not. Both needs to be defined or else prior is ignored.')
        # warn if stellar parameters are too far outside the edge of the grid
        self._grid_check(eff_t, surf_g, met)

        # makes things go vroom vroom. Predictions take a long time
        lower_wl = min(wavelength)
        upper_wl = max(wavelength)
        balder_wl = read.get_wavelengths()
        mask = (lower_wl <= balder_wl) & (balder_wl <= upper_wl)
        if (mask == False).all(): # check if the input wavelength is encompassed by our data
            raise ValueError('Input wavelength does not overlap with the model data. Input wavelength : {}, model data wavelength: {}'.format(wavelength, balder_wl))
        self.cut_wl = balder_wl[mask]
        inds = np.where(mask == True)[0]
        ind_l = inds[0]
        ind_u = inds[-1]+1
        self.cut_models = self.models[ind_l:ind_u]

        if (method == 'bayes') and (prior is not None) and (abunds is not None): # if you want to input a prior to bayesian optimisation
            prior = np.array(prior)
            abunds = np.array(abunds)
            if (abunds.shape != prior.shape) or (len(abunds.shape) != 1) or (len(prior.shape) != 1): # each abundance needs a prior
                raise ValueError('The length of abundance and prior should be the same, they should also be 1D arrays, detected shape: abunds: {}, prior: {}'.format(abunds.shape, prior.shape))
            probs = self._bayesian_optimisation(wavelength, flux, flux_err, eff_t, surf_g, met, abunds, prior = prior)
            abundance = abunds[np.argmax(probs)]
        else:
            abundance = self._window_search(wavelength, flux, flux_err, eff_t, surf_g, met, accuracy = accuracy, min_abund = min_abund, max_abund = max_abund, initial_accuracy = initial_accuracy, method = method)

        # warn if predicted Li is outside of grid
        if (abundance < -0.75) or (abundance > 4.25):
            warnings.warn('Predicted lithium abundance is outside of the grid, results are extrapolated and may not be reliable.')

        return abundance

    def _window_search(self, wavelength, flux, flux_err, eff_t, surf_g, met, method = 'bayes', accuracy = 1e-5, min_abund = -0.5, max_abund = 4, initial_accuracy = 1e-1):
        """An algorithm that finds the maximum or minimum - variation of ternary search. Used to make finding the abundances faster.
        """

        current_accuracy = initial_accuracy
        once = False
        while abs(current_accuracy - accuracy) > accuracy:
            abunds = np.arange(min_abund, max_abund + current_accuracy/2, current_accuracy)
            # find the index of the best abundance
            if method == 'bayes':
                probs = self._bayesian_optimisation(wavelength, flux, flux_err, eff_t, surf_g, met, abunds)
                best_abund_index = np.argmax(probs)
            elif method == 'leastsq':
                res_sq = self._leastsq(wavelength, flux, abunds, eff_t, surf_g, met)
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

    def _bayesian_optimisation(self, wavelength, flux, flux_err, eff_t, surf_g, met, abunds, prior = None):
        """Use Bayesian optimisation to find the probability of the model (abundance) given data.
        """

        # convert to numpy array if list
        probs = []
        guess_fluxes = self._predict_flux(eff_t, surf_g, met, abunds)
        if prior is None:
            prior = np.full(len(guess_fluxes), 1)

        for g_flux, p, a in zip(guess_fluxes, prior, abunds):
            model = CubicSpline(self.cut_wl, g_flux)
            probability = self._p_model_given_data(wavelength, flux, flux_err, p, model)
            probs.append(probability)
        probs = np.array(probs)
        rescaled_probs = np.exp(probs - max(probs)) # the probabilities were logged in _p_model_given_data, normalise before taking the exponential to prevent overflow
        return rescaled_probs

    def _p_model_given_data(self, xdata, ydata, sigma, prior, model):
        """Bayesian inference - probability of data given model. The prior is for the input model.
        """

        xdata = np.array(xdata)
        ydata = np.array(ydata)
        p_data_given_model = 1/np.sqrt(2*np.pi*sigma**2)*np.e**(-(ydata - model(xdata))**2/(2*sigma**2)) + 1e-5 # sometimes the probability is 0
        posterior = np.sum(np.log(p_data_given_model)) + np.log(prior + 1e-5) # we don't normalise the probabilities, so to avoid overflow we log
        return posterior

    def _leastsq(self, wavelength, flux, abunds, eff_t, surf_g, met):
        """Calculates the least squares of the input abundances.
        """

        guess_flux = self._predict_flux(eff_t, surf_g, met, abunds)
        int_flux = [CubicSpline(self.cut_wl, g_flux)(wavelength) for g_flux in guess_flux]
        diff = np.array(int_flux) - np.array(flux)
        diff_square = np.square(diff)
        res_sq = np.sum(diff_square, axis = 1)
        return res_sq

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
        self._grid_check(eff_t, surf_g, met)
        # warn if abundance is too far outside the grid range
        if (abundance < -0.75) or (abundance > 4.25):
            warnings.warn('Input abundance is outside of the grid, results are extrapolated and may not be reliable.')

        return self._predict_flux(eff_t, surf_g, met, [abundance])[0]

    def _predict_flux(self, eff_t, surf_g, met, abundance):
        """Same as predict_flux. This is the hidden version without asserts for improved performance.

        Also, you can call this version with a list of abundances. It's faster if you call this function with a list of abundances vs calling the user visible one with a for-loop over abundances. Vroom vroom.
        """

        transformed_input = self.scalar.transform([[eff_t, surf_g, met, abund] for abund in abundance])
        if self.cut_models is not None: # only predict a range of models
            predicted = np.array([model.predict(transformed_input) for model in self.cut_models]).T
        else: # predict all models
            predicted = np.array([model.predict(transformed_input) for model in self.models]).T
        predicted = 1 + self.relative_error - 10**predicted
        return predicted
