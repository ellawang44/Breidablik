Getting Started
===============

Installation
------------

See the installation instructions in the README of the repository: https://github.com/ellawang44/Breidablik

Examples
--------

Predict abundance from spectra
++++++++++++++++++++++++++++++

Lets say we have an observed spectrum of the sun and we want to determine the lithium abundance from this spectrum using the 670.9 nm lithium line.

Download the example spectrum, `example.txt`, from https://www.mso.anu.edu.au/~ellawang/. This example spectrum was generated from our simulation of the sun with noise added to the spectrum, how this is done is shown in example 3: generate mock observed spectrum

First, we will read in the spectrum.

::

  # read spectra
  import numpy as np
  wl, flux, flux_err = np.loadtxt('example.txt', unpack = True)

Since we've observed the Sun, we will use the stellar parameters for the Sun.

::

  # stellar parameters for the Sun
  t_eff = 5777 # K
  log_g = 4.4 # dex
  met = 0 # dex

Now we can see what lithium abundance the model predicts the Sun has - for this example spectrum, 1.1 dex :sup:`7`\Li was used.

::

  # determine the lithium abundance
  from breidablik.interpolate import spectra
  # initialise models
  models = spectra.Interpolate()
  # find the abundance of the observed spectrum
  abund = models.find_abund(wl, flux, flux_err, t_eff, log_g, met)
  print(abund)

The output is 1.128 dex, which is not a bad prediction considering that 1.1 dex was simulated.

Check results
+++++++++++++

We have an output lithium abundance prediction, but it is good to check what the profile for this predicted abundance looks like, and if it matches the observed spectra.

We'll use the same models to predict what the spectra looks like. The stellar parameters will be the same, and the abundance will be the predicted abundance. These variables in the following code are the same as those in example 1: predict abundance from spectra.

::

  # predict flux
  pred_flux = models.predict_flux(t_eff, log_g, met, abund)
  # read in wavelengths for predicted flux
  from breidablik.analysis import read
  pred_wl = read.get_wavelengths()
  # all of the simulated spectra have the same wavelength points

We can plot the observed flux and the predicted flux to see if they look similar.

::

  # plot observed and predicted flux
  import matplotlib.pyplot as plt
  plt.plot(wl, flux, label = 'observed', color = 'C0')
  plt.errorbar(wl, flux, yerr = flux_err, ecolor = 'C0')
  plt.plot(pred_wl, pred_flux, label = 'predicted', color = 'C1')
  plt.legend()
  plt.show()

Generate mock observed spectrum
+++++++++++++++++++++++++++++++

We start with our simulated spectra for the sun with 1.1 dex :sup:`7`\Li, `sun.txt`, this can be downloaded from https://www.mso.anu.edu.au/~ellawang/

First, we'll read in the simulated spectra and wavelengths.

::

  # read spectra
  import numpy as np
  flux, _ = np.loadtxt('sun.txt', unpack = True)

  # read wavelengths
  from analysis import read
  wl = read.get_wavelengths()
  # all of the simulated spectra have the same wavelength points

Next, we'll narrow down the spectra to the region we care about, the original spectra ranges from about 600.0 nm to 820.0 nm.

::

  # cut spectra
  from analysis import tools
  wl_cut, flux_cut = tools.cut(wl, flux, center = 6709.659,
                               upper = 4, lower = 4)

Lastly, to mimic real observations, we'll need spectra observed at roughly equidistant points in wavelength, and some noise.

::

  # re-interpolate to equidistant wavelength points
  from scipy.interpolate import CubicSpline
  num_points = 500
  wl_equidist = np.linspace(wl_cut[0], wl_cut[-1], num_points)
  flux_equidist = CubicSpline(wl_cut, flux_cut)(wl_equidist)

  # add Gaussian noise
  from numpy.random import normal
  noise_scale = 0.003 # control how noisy the spectrum is
  noise = normal(scale = noise_scale, size = num_points)
  flux_noise = flux_equidist + noise
  # generate array for error in each pixel
  flux_err = np.full(len(flux_noise), noise_scale)

Now we can write this spectra to a file

::

  # write to file
  name = 'example.txt'
  data = np.array([wl_equidist, flux_noise, flux_err]).T
  header = 'wavelength (A) \t normalised flux \t flux error'
  np.savetxt(name, data, fmt = '%.5e', header = header)
