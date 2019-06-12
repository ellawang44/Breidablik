Example 5: Get NLTE correction
==============================

About
-----

This example will show you how to find the NLTE correction for a 1D LTE abundance by using the ``Nlte`` class in ``breidablik.interpolate.nlte``.

Example
-------

Lets assume we want to find the NLTE correction for a 1D LTE abundance of 2.2 dex for LP 815-43.

::

  abund_1DLTE = 2.2 # dex
  # stellar parameters for LP 815-43
  t_eff = 6400 # K
  log_g = 4.17 # cgs
  met = -2.74 # dex

Next, we load in the models by initialising the class, then call the function ``nlte_correction`` with the parameters we just defined.

::

  from breidablik.interpolate.nlte import Nlte
  # initialise model
  model = Nlte()
  # find NLTE correction
  nltec = model.nlte_correction(t_eff, log_g, met, abund_1DLTE)
  print(nltec)

Which returns a correction of FILLMEIN dex.
