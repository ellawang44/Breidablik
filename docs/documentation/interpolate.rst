interpolate
===========
Classes used to interpolate and predict lithium abundances and spectra.

spectra
-------
Class used to predict lithium abundance from spectra and predict spectra from lithium abundance.

.. autoclass:: breidablik.interpolate.spectra.Spectra
    :members:
    :special-members: __init__
    :undoc-members:

rew
---
Class used to predict lithium abundance from REW.

.. autoclass:: breidablik.interpolate.rew.Rew
    :members:
    :special-members: __init__
    :undoc-members:

nlte
----
Class used to find the NLTE correction for lithium abundance for a given 1D LTE lithium abundance.

.. autoclass:: breidablik.interpolate.nlte.Nlte
    :members:
    :special-members: __init__
    :undoc-members:
