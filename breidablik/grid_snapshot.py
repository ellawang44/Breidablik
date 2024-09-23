from analysis import read
import numpy as np

data = read.read_all() # for marcs version, read in marcs grid

# make every model 0.5 in distance away from each other
# normalisation
t_step = 0.5/500
m_step = 0.5

models = np.array(list(data))

models[:, 0] *= t_step
models[:, 2] *= m_step

with open('grid_snapshot.txt', 'w') as f:
    f.write(str(t_step) + '\t' + str(m_step) + '\n')
    np.savetxt(f, models)

# note that some models are removed by hand in grid_snapshot_marcs.txt