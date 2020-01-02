import numpy as np
import os

class Scalar:
    """Scalar class used to scale data. Can create a scalar, scale input data, save and load previous scalars.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        """Create scalar
        """

        # make sure no crazy inputs
        try:
            data = np.array(data)
        except:
            raise ValueError('Data must be able to be converted into a numpy array.')

        # make sure the dimension of the data is correct
        if len(data.shape) != 2:
            raise ValueError('Data must be a 2D-array.')

        self.mean = np.mean(data, axis = 0)
        self.std = np.std(data, axis = 0)

    def transform(self, data):
        """Scale input data
        """

        # make sure there is a fitted scalar
        if (self.mean is None) or (self.std is None):
            raise AttributeError('A scalar must be created before data can be fitted. Call fit to fit a scalar.')

        # make sure no crazy inputs
        try:
            data = np.array(data)
        except:
            raise ValueError('Data must be able to be converted into a numpy array.')

        # make sure the dimension of the data is correct
        if len(data.shape) != 2:
            raise ValueError('Data must a 2D-array.')

        # make sure dimensions of data to be transformed and fitted data are the same
        if data.shape[1] != len(self.mean):
            raise ValueError('Data to be transformed must have the same number of columns as the fitted data.')

        scaled_data = (data - self.mean)/self.std
        return scaled_data

    def save(self, name):
        """Save scalar
        """

        if (self.mean is None) or (self.std is None):
            raise AttributeError('Need a fitted scalar before saving the scalar. Call fit to fit a scalar.')
        else:
            np.save(name, [self.mean, self.std])

    def load(self, name):
        """Load scalar
        """

        path = os.path.join(os.getcwd(), name)
        if os.path.isfile(path):
            raise FileNotFoundError('Attempted to load a scalar not found, path given: {}'.format(path))
        else:
            self.mean, self.std = np.load(name)
