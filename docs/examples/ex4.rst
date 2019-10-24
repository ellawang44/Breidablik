Example 4: Get abundance from REW
=========================================

About
-----

This example will show you how to derive the lithium abundance given a REW by using the ``Rew`` class in ``breidablik.interpolate.rew``.

Example
-------

Lets assume we want to find the 3D NLTE abundance for a REW of -6 for LP 815-43.

::

  REW = -6
  # stellar parameters for LP 815-43
  t_eff = 6400 # K
  log_g = 4.17 # cgs
  met = -2.74 # dex

Next, we load in the models by initialising the class, then call the function ``find_abund`` with the parameters we just defined.

::

  from breidablik.interpolate.rew import Rew
  # initialise model
  model = Rew()
  # find abundance
  abund = model.find_abund(t_eff, log_g, met, REW)
  print(abund)

Which returns an abundance of 1.702 dex.
