Quickstart
==========

fcaz distributions are used like frozen :mod:`scipy.stats` distributions: construct one with
``scale``, then call ``pdf``, ``cdf``, ``sf``, ``ppf`` or ``rvs``.

.. doctest::

   >>> from fcaz.stats import invexpon
   >>> X = invexpon(scale=4)
   >>> round(float(X.cdf(6)), 4)
   0.5134

Hazard functions
----------------

:func:`fcaz.stats.hf` and :func:`fcaz.stats.chf` work on any frozen distribution, scipy's
included.

.. doctest::

   >>> from scipy import stats
   >>> from fcaz.stats import chf, hf
   >>> round(float(hf(stats.expon(scale=8), 2.5)), 4)
   0.125
   >>> round(float(chf(stats.expon(scale=8), 2.5)), 4)
   0.3125

Mixtures
--------

A random variable that is an exponential with mean 8 with probability 0.4 and an inverse
exponential with :math:`\theta = 3` with probability 0.6:

.. doctest::

   >>> from fcaz.stats import Mixture
   >>> X = Mixture([stats.expon(scale=8), invexpon(scale=3)], [0.4, 0.6])
   >>> round(float(X.sf(2.5)), 4)
   0.7119
   >>> round(float(X.chf(2.5)), 4)
   0.3398
   >>> X.rvs(size=3, random_state=0).shape
   (3,)
