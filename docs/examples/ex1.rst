Example 1: Predict abundance from spectra
=========================================

About
-----

This example will show you how to predict a lithium abundance from an observed spectrum by using the ``Spectra`` class in ``breidablik.interpolate.spectra``.

Example
-------

Lets say we have an observed spectrum of the sun and we want to determine the lithium abundance from this spectrum using the 670.9 nm lithium line.

Download the example spectrum, ``example_spec.txt``, from https://www.mso.anu.edu.au/~ellawang/. This example spectrum was generated from our simulation of the sun with noise added to the spectrum, how this is done is shown in example 3: generate mock observed spectrum

First, we will read in the spectrum.

::

  # read spectra
  import numpy as np
  wl, flux, flux_err = np.loadtxt('example_spec.txt', unpack = True)

Since we've observed the Sun, we will use the stellar parameters for the Sun.

::

  # stellar parameters for the Sun
  t_eff = 5777 # K
  log_g = 4.4 # cgs
  met = 0 # dex

Now we can see what lithium abundance the model predicts the Sun has - for this example spectrum, 1.1 dex :sup:`7`\Li was used.

::

  # determine the lithium abundance
  from breidablik.interpolate.spectra import Spectra
  # initialise models
  models = Spectra()
  # find the abundance of the observed spectrum
  abund = models.find_abund(wl, flux, flux_err, t_eff, log_g, met)
  print(abund)

Checking the value of ``abund``, we have that the output is 1.072 dex, which is not a bad prediction considering that 1.1 dex was simulated.
