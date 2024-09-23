from breidablik.interpolate.load import FFNN
from breidablik.interpolate.grid_check import _grid_check
from breidablik.interpolate.scalar import Scalar
import numpy as np
from pathlib import Path
import warnings

_base_path = Path(__file__).parent

class Rew:
    """Interpolation class for REW. Used to interpolate between the stellar parameters. Can find the abundance given the REW and stellar parameters.
    """

    def __init__(self, dim = 3, model_path = None, scalar_path = None, model_path_610 = None, scalar_path_610 = None, model_path_810 = None, scalar_path_810 = None):
        """Initialise the data by reading the pickled models and scalar.

        Parameters
        ----------
        dim : int, optional
            The dimensionality of the model atmospheres. By default this is 3 for 3D, referring to Stagger model atmospheres. Alternatively 1 for 1D marcs model atmospheres - useful if you need consistent abundances for stellar parameters outside of the Stagger grid. Note that this only applies to the 670 line.
        model_path : str, optional
            The path to the rew model to be used to predict the lithium abundance. By default, this path points to ``models/rew`` in ``breidablik``.
        scalar_path : str, optional
            The path to the scalar corresponding to the rew model. By default, this path points to ``models/rew/scalar.npy`` in ``breidablik``.
        model_path_610 : str, optional
            Similar to model_path, except for the 610 line.
        scalar_path_610 : str, optional
            Similar to scalar_path, except for 610 line.
        model_path_810 : str, optional
            Similar to model_path, except for 810 line.
        scalar_path_810 : str, optional
            Similar to scalar_path, except for 810 line.
        """

        self.dim = dim
        # set default paths
        if self.dim == 3:
            model_path = model_path or _base_path.parent / 'models/rew'
            scalar_path = scalar_path or _base_path.parent / 'models/rew/scalar.npy'
        elif self.dim == 1:
            model_path = model_path or _base_path.parent / 'models/marcs'
            scalar_path = scalar_path or _base_path.parent / 'models/marcs/scalar.npy'
        else:
            raise ValueError('dim needs to be 3 or 1. Input dim: {}'.format(self.dim))
        model_path_610 = model_path_610 or _base_path.parent / 'models/rew_610'
        scalar_path_610 = scalar_path_610 or _base_path.parent / 'models/rew_610/scalar.npy'
        model_path_810 = model_path_810 or _base_path.parent / 'models/rew_810'
        scalar_path_810 = scalar_path_810 or _base_path.parent / 'models/rew_810/scalar.npy'
        # load models
        scalar = Scalar()
        scalar.load(scalar_path)
        scalar_610 = Scalar()
        scalar_610.load(scalar_path_610)
        scalar_810 = Scalar()
        scalar_810.load(scalar_path_810)
        self.models = [FFNN(model=str(model_path_610)), FFNN(model=str(model_path)), FFNN(model=str(model_path_810))]
        self.scalars = [scalar_610, scalar, scalar_810]

    def find_abund(self, eff_t, surf_g, met, rew, center = 670.9659):
        """Find the abundance based on the stellar parameters and measured reduced equivalent width.

        Parameters
        ----------
        rew : Real
            The reduced equivalent width for the lithium line at 670.9 nm.
        eff_t : Real
            The effective temperature of the star.
        surf_g : Real
            The log surface gravity of the star.
        met : Real
            The metallicity of the star.
        center : Real, optional
            The center of the lithium line that the input rew corresponds to, in angstroms. The 3 lithium lines are centered at 610.5298, 670.9659, and 812.8606 nm in the Balder results. The input center value will snap to the closest value out of those 3.

        Returns
        -------
        predcited_li : float
            The predicted lithium abundance.
        """

        # centers of the 3 lines we model
        line_centers = np.array([610.5298, 670.9659, 812.8606])
        # get which model is being used
        ind = np.argmin(np.abs(line_centers - center))
        # check the dim and line
        if self.dim == 1 and ind != 1:
            warnings.warn('The 1D abundances are only available for the 670 line. Output abundance is in 3D for the line closest to {}.'.format(center))

        # check the input stellar parameters and abundance
        if not ((np.array(eff_t).shape == ()) and (np.array(surf_g).shape == ()) and (np.array(met).shape == ()) and (np.array(rew).shape == ())):
            raise ValueError('The input effective temperature, surface gravity, metallicity, or abundance is not in the right format, they all need to be scalar numbers, detected inputs: eff_t = {}, surf_g = {}, met = {}, and rew = {}'.format(eff_t, surf_g, met, rew))
        # warn if stellar parameters are too far outside the edge of the grid
        _grid_check(eff_t, surf_g, met, self.dim)

        predicted_li = self._find_abund(eff_t, surf_g, met, [rew], ind = ind)[0][0]

        # warn if predicted Li is outside of grid
        if (predicted_li < -0.5) or (predicted_li > 4):
            warnings.warn('Predicted lithium abundance is outside of the grid, results are extrapolated and may not be reliable.')

        return predicted_li

    def _find_abund(self, eff_t, surf_g, met, rew, ind = 1):
        """Same as find_abund_rew, hidden version without grid checks so extra warnings aren't thrown. This version can be used to quickly process many rew values.
        """
        
        # predict lithium abundance
        scalar = self.scalars[ind]
        model = self.models[ind]
        transformed_input = scalar.transform([[eff_t, surf_g, met, r] for r in rew])
        predicted_li = model(transformed_input)

        return predicted_li
