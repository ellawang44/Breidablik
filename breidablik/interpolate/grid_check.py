import numpy as np
from pathlib import Path
import warnings

_base_path = Path(__file__).parent

def _grid_check(eff_t, surf_g, met):
    """Check if the stellar parameters are too far outside the edge of the grid. Essentially normalising the parameters and then drawing circles around the points. 
    """

    with open(_base_path.parent / 'grid_snapshot.txt', 'r') as f:
        t_step, m_step = np.float_(f.readline().split())
        grid = np.loadtxt(f)
    input_model = np.array([eff_t*t_step, surf_g, met*m_step])
    input_tile = np.tile(input_model, (grid.shape[0], 1))
    min_dist = min(np.sqrt(np.sum(np.square(grid - input_tile), axis = 1)))
    # the maximum minimum distance between points in the grid is 0.5063
    # np.sqrt(3*0.25**2) = 0.4330
    # there should be no holes in the grid
    if min_dist > np.sqrt(3*0.25**2):
        warnings.warn('Input stellar parameters are outside of the grid, results are extrapolated and may not be reliable. Stellar parameters inputted were: teff = {0:.2f}, surfg = {1:.2f}, met = {2:.2f}'.format(eff_t, surf_g, met))
