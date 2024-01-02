Example 3: Generate mock observed spectrum
==========================================

About
-----

This example will show you how to manipulate the raw data using functions in ``breidablik.analysis.read`` and ``breidablik.analysis.tools``.

Example
-------

We start with our simulated spectra for the sun with 1.1 dex :sup:`7`\Li, ``sun.txt``, this can be downloaded from https://zenodo.org/records/10428805

First, we'll read in the simulated spectra and wavelengths.

::

  # read spectra
  import numpy as np
  flux, _ = np.loadtxt('sun.txt', unpack = True)

  # read wavelengths
  from breidablik.analysis import read
  wl = read.get_wavelengths()
  # all of the simulated spectra have the same wavelength points

Next, we'll narrow down the spectra to the region we care about, the original spectra ranges from about 600.0 nm to 820.0 nm.

::

  # cut spectra
  from breidablik.analysis import tools
  wl_cut, flux_cut = tools.cut(wl, flux, center = 670.9659,
                               upper = 0.4, lower = 0.4)

Lastly, to mimic real observations, we'll need spectra observed at roughly equidistant points in wavelength, and some noise.

::

  # re-interpolate to equidistant wavelength points
  from scipy.interpolate import CubicSpline
  num_points = 500
  wl_equidist = np.linspace(wl_cut[0], wl_cut[-1], num_points)
  flux_equidist = CubicSpline(wl_cut, flux_cut)(wl_equidist)

  # add Gaussian noise
  from numpy.random import normal, seed
  seed(19) # set seed for replicability
  noise_scale = 0.003 # control how noisy the spectrum is
  noise = normal(scale = noise_scale, size = num_points)
  flux_noise = flux_equidist + noise
  # generate array for error in each pixel
  flux_err = np.full(len(flux_noise), noise_scale)

Now we can write this spectra to a file

::

  # write to file
  name = 'example_spec.txt'
  data = np.array([wl_equidist, flux_noise, flux_err]).T
  header = 'wavelength (nm) \t normalised flux \t flux error'
  np.savetxt(name, data, fmt = '%.5e', header = header)
