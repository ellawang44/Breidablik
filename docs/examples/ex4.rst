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

Which returns an abundance of 1.70 dex.

``find_abund`` accepts a parameter: ``center``, that determines which model is used: there are 3 models, 1 for each Li line. The 3 lithium lines are centered at 610.5298, 670.9659, and 812.8606 nm. The input ``center`` value will snap to the closest value out of those 3. By default, the 670.9 nm model is used.

The 3D stagger grid does not go to low temperatures. We can calculate 1D NLTE A(Li) trained on the marcs grid.

::

  # initialise model trained on 1D NLTE A(Li)
  model = Rew(dim = 1)

This model was only trained on the 670.9 nm line. If you call ``find_abund`` using a different Li line, a warning will be raised and the abundance that ``find_abund`` returns will be in 3D NLTE. 