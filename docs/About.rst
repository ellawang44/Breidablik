About
=====

An interpolation routine and abundance predictor wrapper around stellar spectra for lithium generated from Balder (Amarsi et al. 2016) and the Stagger-grid (Magic et al. 2013). The raw synthetic spectra and models can be found at https://zenodo.org/records/10428805. We use radial basis functions and feedforward neural networks to interpolate between stellar parameters and lithium abundance inputs to generate interpolated synthetic spectra, these interpolation models are provided as part of the package. Using the interpolation routine, we can predict the lithium profile given any stellar parameters and lithium abundance input, we can also predict the lithium abundance given an observed spectrum and stellar parameters.
