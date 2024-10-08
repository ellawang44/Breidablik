import numpy as np
from pathlib import Path
import warnings

_base_path = Path(__file__).parent

def _grid_check(eff_t, surf_g, met, dim = 3):
    """Check if the stellar parameters are too far outside the edge of the grid. Essentially normalising the parameters and then drawing circles around the points. 
    """

    if dim == 3:
        filename = ''
    elif dim == 1:
        filename = '_marcs'

    with open(_base_path.parent / 'grid_snapshot{}.txt'.format(filename), 'r') as f:
        t_step, m_step = np.float64(f.readline().split())
        grid = np.loadtxt(f)
    # round the teffs
    grid[:,0] = np.round(grid[:,0]*2)/2
    input_model = np.array([eff_t*t_step, surf_g, met*m_step])
    input_tile = np.tile(input_model, (grid.shape[0], 1))
    min_dist = min(np.sqrt(np.sum(np.square(grid - input_tile), axis = 1)))
    #TODO: squares? circles/squares = 2.72
    # there should be no holes in the grid
    if min_dist > np.sqrt(3*0.25**2):
        warnings.warn('Input stellar parameters are outside of the grid, results are extrapolated and may not be reliable. Stellar parameters inputted were: teff = {0:.2f}, surfg = {1:.2f}, met = {2:.2f}'.format(eff_t, surf_g, met))
