from breidablik.interpolate.grid_check import _grid_check
import joblib
import numpy as np
from pathlib import Path
import warnings

_base_path = Path(__file__).parent

class Interpolate:
    """Interpolation class for REW. Used to interpolate between the stellar parameters. Can find the abundance given the REW and stellar parameters.
    """

    def __init__(self, model_path = _base_path.parent / 'models/rew_3D.pkl', scalar_path = _base_path.parent / 'models/rew_3D_scalar.pkl'):
        """Initialise the data by reading the pickled models and scalar.

        Parameters
        ----------
        model_path : str, optional
            The path to the rew model to be used to predict the lithium abundance.
        scalar_path : str, optional
            The path to the scalar corresponding to the rew model.
        """

        self.models = [None, joblib.load(model_path), None]
        self.scalars = [None, joblib.load(scalar_path), None]

    def find_abund(self, eff_t, surf_g, met, rew, center = 6709.659):
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
        _grid_check(eff_t, surf_g, met)

        predicted_li = self._find_abund(eff_t, surf_g, met, [rew], center = center)[0]

        # warn if predicted Li is outside of grid
        if (predicted_li < -0.75) or (predicted_li > 4.25):
            warnings.warn('Predicted lithium abundance is outside of the grid, results are extrapolated and may not be reliable.')

        return predicted_li

    def _find_abund(self, eff_t, surf_g, met, rew, center = 6709.659):
        """Same as find_abund_rew, hidden version without grid checks so extra warnings aren't thrown. This version can be used to quickly process many rew values.
        """

        line_centers = np.array([6105.298, 6709.659, 8128.606])
        ind = np.argmin(np.abs(line_centers - center))
        scalar = self.scalars[ind]
        model = self.models[ind]
        transformed_input = scalar.transform([[eff_t, surf_g, met, r] for r in rew])
        predicted_li = model.predict(transformed_input)

        return predicted_li