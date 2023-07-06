import json
import numpy as np
import os
import warnings


def ReLU(x):
    return x * (x > 0)
    
def linear(x, weights, bias):
    return x @ weights.T + bias


class FFNN:
    '''Feed Forward Neural Network'''

    def __init__(self, model):
        self.load(path=model)

    def forward(self, x):
        '''Forward pass
        
        Parameters
        ----------
        x : ndarray
            The features matrix to predict results on. 
        
        Returns
        -------
        y : ndarray
            THe predicted results. 
        '''

        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = linear(x, w, b)
            x = ReLU(x)
        # last layer linear activation function
        x = linear(x, self.weights[-1], self.biases[-1])
        return x

    def __call__(self, x):
        return self.forward(x)

    def load(self, path, model='ffnn.json'):
        '''Load model from json file. Assumes that the model is saved with keys weight_x, bias_x, where x is the layer number. 
        
        Parameters
        ----------
        path : str
            Path to load the model from. Needs to be a folder.
        '''

        # open json
        with open(f'{path}/{model}', 'r') as f:
            json_state = json.load(f)
        state_dict = json.loads(json_state)
        
        # parse json contents into lists
        layers = len(state_dict.keys())//2
        self.weights = [np.array(state_dict[f'weight_{i+1}']) for i in range(layers)]
        self.biases = [np.array(state_dict[f'bias_{i+1}']) for i in range(layers)]

        # test if model returns the same results as when it was saved
        test = np.load(os.path.join(path, 'test.npy'), allow_pickle = False)
        if not (np.abs(self.forward(test[:,:4]).flatten() - test[:,4]) < 1e-5).all():
            warnings.warn('The saved model may be corrupted or major changes may have occured in pytorch. Re-download the model, roll back to an older version of pytorch, or retrain the model.')