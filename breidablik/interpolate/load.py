import torch
import torch.nn as nn
import numpy as np
import os
import warnings

class FFNN:
    '''Feedforward neural network.

    Not intended for big models!
    '''

    def __init__(self, layers, neurons, f_act=nn.ReLU(), model=None):
        '''
        Parameters
        ----------
        layers : int
            The number of layers in the ffnn. Includes the input and output layer. i.e. (in:4, 10), (10, out:1) is 2 layers.
        neurons : List[int] or int
            The number of neurons per layer. If int, then each layer uses the same number of neurons. If list, then each element specifies the layer and the layers input is ignored.
        f_act : List[class torch.nn] or class torch.nn, optional
            The activation function per layer. If not list, then each layer uses the same activation functions. If list, then each element specifies the activation function of the layer and the layers input is ignored. 
        model : class torch.nn or str, optional
            A fully custom model. If a str is given, then it is assumed that a trained model is at that location. If this is given then layers, neurons, and f_act is ignored.
        '''

        if type(model) is str:
            self.load(model)
        else:
            self.model = model

        if model is None:
            # neurons
            if type(neurons) is list: # custom
                self.neurons = neurons
            else: # create neuron list
                self.neurons = [neurons for _ in range(layers-1)]
            # f_act
            if type(f_act) is list: # custom
                self.f_act = f_act
            else: # create list based on other settings
                self.f_act = [f_act for _ in range(len(self.neurons))]

            # check that the neuron and activation layers are the same length
            if not ((len(self.neurons) == len(self.f_act)-1) or (len(self.neurons) == len(self.f_act))):
                raise ValueError('length of neurons and f_act is not compatible, need len(neurons) == len(f_act)-1 or len(neurons) == len(f_act) (last f_act is linear)')

    def _make_model(self, in_neurons, out_neurons):
        '''Make model.
        '''

        # make neuron blocks
        neurons = [in_neurons, *self.neurons, out_neurons]
        n_blocks = [nn.Linear(neurons[ind], neurons[ind+1]) for ind in range(len(neurons)-1)]
        
        # make model by folding n_blocks and f_blocks
        if len(n_blocks) == len(self.f_act): # f_act last layer
            model = []
            for n, f in zip(n_blocks, self.f_act):
                model.append(n)
                model.append(f)
        elif len(n_blocks)-1 == len(self.f_act): # linear f_act last
            model = [n_blocks[0]]
            for n, f in zip(n_blocks[1:], self.f_act):
                model.append(f)
                model.append(n)

        self.model = nn.Sequential(*model)

    def predict(self, X):
        '''Predict results. 
         
        Parameters
        ----------
        X : ndarray
            The features matrix to predict results on. 
        
        Returns
        -------
        y : ndarray
            THe predicted results. 
        '''

        self.model.eval() # put in test mode
        with torch.no_grad():
            return self.model(torch.Tensor(X).float()).detach().numpy()

    def load(self, path, model='ffnn'):
        '''Load a saved model.

        Parameters
        ----------
        path : str
            Path to load the model from. Needs to be a folder.
        '''

        self.model = torch.load(f'{path}/{model}')

        # test if model returns the same results as when it was saved
        test = np.load(os.path.join(path, 'test.npy'), allow_pickle = False)
        if not ((self.predict(test[:,:4]).flatten() - test[:,4]) < 1e-5).all():
            warnings.warn('The saved model may be corrupted or major changes may have occured in pytorch. Re-download the model, roll back to an older version of pytorch, or retrain the model.')

