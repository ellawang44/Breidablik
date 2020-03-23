import numpy as np
import warnings
import os
import sklearn.neural_network

def load(name):
    """Loads the saved model.

    Parameters
    ----------
    name : str
        The name of the model being loaded.

    Returns
    -------
    model : MLPRegressor
        The loaded model.
    """

    # load all the npy files
    coef_files = sorted([i for i in os.listdir(name) if i[:4] == 'coef'])
    coefs = [np.load(os.path.join(name, i), allow_pickle = False) for i in coef_files]
    int_files = sorted([i for i in os.listdir(name) if i[:3] == 'int'])
    intercepts = [np.load(os.path.join(name, i), allow_pickle = False) for i in int_files]
    n_outputs = np.load(os.path.join(name, 'n_outputs.npy'), allow_pickle = True)
    layers = np.load(os.path.join(name, 'layers.npy'), allow_pickle = True)
    n_layers = len(layers) + 2 # includes the input and output layer
    out_activation = np.load(os.path.join(name, 'out_activation.npy'), allow_pickle = True)
    alpha = np.load(os.path.join(name, 'alpha.npy'), allow_pickle = True)

    # create the model
    model = sklearn.neural_network.MLPRegressor(hidden_layer_sizes = layers, activation = 'relu', solver = 'lbfgs', max_iter = 100000, alpha = alpha, tol = 1e-6)

    # change variables in the model
    model.coefs_ = coefs
    model.intercepts_ = intercepts
    model.n_outputs_ = n_outputs
    model.n_layers_ = n_layers
    model.out_activation_ = str(out_activation)

    # test if model returns the same results as when it was saved
    test = np.load(os.path.join(name, 'test.npy'), allow_pickle = False)
    if not ((model.predict(test[:,:4]) - test[:,4]) < 1e-5).all():
        warnings.warn('The saved model may be corrupted or major changes may have occured between the version of scikit-learn the model was trained on compared to the version of scikit-learn the model is being loaded on. Re-download the model or retrain the model.')

    return model
