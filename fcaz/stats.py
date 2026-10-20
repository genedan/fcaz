"""
Distribution tools that extend :mod:`scipy.stats` with actuarial transformations.

Everything here follows the scipy conventions: ``invexpon(scale=theta)`` is used like
``scipy.stats.expon(scale=theta)``, and :class:`Mixture` returns an object with the
frozen-distribution methods (``pdf``, ``cdf``, ``sf``, ``ppf``, ``rvs``, ``mean``) plus
``hf`` and ``chf``.
"""

import math

from collections.abc import Sequence
from typing import Any

import numpy as np

from numpy.typing import NDArray
from scipy import optimize
from scipy.stats import rv_continuous

from fcaz.typing import (
    ArrayLikeFloat,
    FloatOrArray,
    FrozenDistribution,
    SupportsLogSf,
    SupportsPdfSf,
)

__all__ = [
    'Mixture',
    'chf',
    'hf',
    'invexpon',
]


def hf(
    dist: SupportsPdfSf,
    x: ArrayLikeFloat,
) -> FloatOrArray:
    """
    Hazard function of a frozen distribution (i.e., one with parameters already fixed).

    Parameters
    ----------
    dist : frozen distribution
        Any object with ``pdf`` and ``sf`` methods, e.g. ``scipy.stats.expon(scale=8)``.
    x : array_like
        Points at which to evaluate the hazard.

    Returns
    -------
    ndarray
        :math:`h(x) = f(x) / S(x)`.

    Examples
    --------
    >>> from scipy import stats
    >>> from fcaz.stats import hf
    >>> round(float(hf(stats.expon(scale=8), 2.5)), 4)
    0.125
    """
    return np.asarray(dist.pdf(x) / dist.sf(x))[()]


def chf(
    dist: SupportsLogSf,
    x: ArrayLikeFloat,
) -> FloatOrArray:
    r"""
    Cumulative hazard function of a frozen distribution (i.e., one with parameters already fixed).

    Parameters
    ----------
    dist : frozen distribution
        Any object with a ``logsf`` method, e.g. ``scipy.stats.expon(scale=8)``.
    x : array_like
        Points at which to evaluate the cumulative hazard.

    Returns
    -------
    ndarray
        :math:`H(x) = -\\ln S(x)`.

    Examples
    --------
    >>> from scipy import stats
    >>> from fcaz.stats import chf
    >>> round(float(chf(stats.expon(scale=8), 2.5)), 4)
    0.3125
    """
    return np.asarray(-dist.logsf(x))[()]


# scipy convention
# noinspection PyPep8Naming
class _invexpon_gen(rv_continuous):
    r"""
    Inverse exponential continuous random variable.

    %(before_notes)s

    Notes
    -----
    The cumulative distribution function is

    .. math::

        F(x) = e^{-\theta / x}, \quad x > 0,

    where :math:`\theta` is the ``scale`` parameter, matching the *Loss Models*
    parameterization. This is :obj:`scipy.stats.invweibull` with ``c=1``.

    %(after_notes)s

    Examples
    --------
    >>> from fcaz.stats import invexpon
    >>> X = invexpon(scale=4)
    >>> round(float(X.cdf(6)), 4)
    0.5134
    >>> round(float(X.ppf(0.3)), 4)
    3.3223
    """

    def _pdf(
        self,
        x: Any,
        *args: Any,
    ) -> Any:
        """Probability density function of the standard form, ``exp(-1/x) / x**2``."""
        return np.exp(-1.0 / x) / x**2

    def _cdf(
        self,
        x: Any,
        *args: Any,
    ) -> Any:
        """Cumulative distribution function of the standard form, ``exp(-1/x)``."""
        return np.exp(-1.0 / x)

    def _sf(
        self,
        x: Any,
        *args: Any,
    ) -> Any:
        """Survival function of the standard form, ``1 - exp(-1/x)``."""
        return -np.expm1(-1.0 / x)

    def _ppf(
        self,
        q: Any,
        *args: Any,
    ) -> Any:
        """Percent point function of the standard form, ``-1 / log(q)``."""
        return -1.0 / np.log(q)


# Inverse exponential distribution from Loss Models.
invexpon: rv_continuous = _invexpon_gen(
    a=0.0,
    name='invexpon',
)


class Mixture:
    r"""
    Finite mixture of frozen distributions (i.e., those with parameters already fixed).

    Parameters
    ----------
    dists : sequence of frozen distributions
        Component distributions, e.g. ``scipy.stats.expon(scale=8)``.
    weights : array_like
        Non-negative mixing weights, one per component, summing to 1.

    Raises
    ------
    ValueError
        If ``dists`` and ``weights`` differ in length, or the weights are negative or do not
        sum to 1.

    Notes
    -----
    The survival function is :math:`S(x) = \sum_i w_i S_i(x)`; ``pdf``, ``cdf`` and
    ``mean`` mix in the same way. ``hf`` and ``chf`` are derived from the mixed ``pdf`` and
    ``sf``.

    Examples
    --------
    A 40% exponential, 60% inverse exponential mixture:

    >>> from scipy import stats
    >>> from fcaz.stats import invexpon, Mixture
    >>> X = Mixture([stats.expon(scale=8), invexpon(scale=3)], [0.4, 0.6])
    >>> round(float(X.chf(2.5)), 4)
    0.3398
    """

    def __init__(
        self,
        dists: Sequence[FrozenDistribution],
        weights: ArrayLikeFloat,
    ) -> None:
        w: NDArray[np.float64] = np.asarray(
            weights,
            dtype=np.float64,
        )
        if len(dists) != len(w):
            raise ValueError('dists and weights must have the same length')
        if bool(w.min() < 0) or not math.isclose(
            float(w.sum()),
            1.0,
            rel_tol=1e-5,
            abs_tol=1e-8,
        ):
            raise ValueError('weights must be non-negative and sum to 1')
        self.dists: list[FrozenDistribution] = list(dists)
        self.weights: NDArray[np.float64] = w

    def _mix(
        self,
        method: str,
        x: ArrayLikeFloat,
    ) -> FloatOrArray:
        r"""
        Weighted sum of a method evaluated on every component.

        Parameters
        ----------
        method : str
            Name of the component method to call, e.g. ``'pdf'``, ``'cdf'`` or ``'sf'``.
        x : array_like
            Points at which to evaluate the method.

        Returns
        -------
        float or ndarray
            :math:`\sum_i w_i \, g_i(x)`, where :math:`g_i` is ``method`` of component
            :math:`i`. Valid for any quantity that mixes linearly, which excludes ``ppf``
            and ``logsf``.
        """
        xs: NDArray[np.float64] = np.asarray(
            x,
            dtype=np.float64,
        )
        total: NDArray[np.float64] = np.zeros_like(xs)
        for w, d in zip(
            self.weights,
            self.dists,
            strict=True,
        ):
            total = total + w * np.asarray(
                getattr(
                    d,
                    method,
                )(xs),
                dtype=np.float64,
            )
        return total[()]

    def pdf(self, x: ArrayLikeFloat) -> FloatOrArray:
        """Probability density function."""
        return self._mix(
            'pdf',
            x,
        )

    def cdf(self, x: ArrayLikeFloat) -> FloatOrArray:
        """Cumulative distribution function."""
        return self._mix(
            'cdf',
            x,
        )

    def sf(self, x: ArrayLikeFloat) -> FloatOrArray:
        """Survival function, 1 - cdf."""
        return self._mix(
            'sf',
            x,
        )

    def logsf(self, x: ArrayLikeFloat) -> FloatOrArray:
        """Log of the survival function."""
        return np.log(self.sf(x))

    def hf(self, x: ArrayLikeFloat) -> FloatOrArray:
        """Hazard function, pdf / sf."""
        return hf(
            self,
            x,
        )

    def chf(self, x: ArrayLikeFloat) -> FloatOrArray:
        """Cumulative hazard function, -log sf."""
        return chf(
            self,
            x,
        )

    def mean(self) -> float:
        """Mean of the mixture, the weighted mean of the component means."""
        return float(
            sum(
                w * d.mean()
                for w, d in zip(
                    self.weights,
                    self.dists,
                    strict=True,
                )
            )
        )

    def ppf(self, q: ArrayLikeFloat) -> FloatOrArray:
        """Percent point function (inverse cdf), solved numerically."""

        def solve(p: float) -> float:
            lo = min(d.ppf(p) for d in self.dists)
            hi = max(d.ppf(p) for d in self.dists)
            return (
                lo
                if lo == hi
                else optimize.brentq(
                    lambda t: float(self.cdf(t)) - p,
                    lo,
                    hi,
                )
            )

        return np.vectorize(
            solve,
            otypes=[float],
        )(q)

    def rvs(
        self,
        size: int | tuple[int, ...] | None = None,
        random_state: Any = None,
    ) -> FloatOrArray:
        """Random variates; ``size`` and ``random_state`` as in scipy."""
        rng = np.random.default_rng(random_state)
        n = 1 if size is None else int(np.prod(size))
        edges = self.weights.cumsum()
        which = np.minimum(
            (rng.random(n)[:, None] >= edges).sum(axis=1),
            len(self.dists) - 1,
        )
        out: NDArray[np.float64] = np.empty(n)
        for i, d in enumerate(self.dists):
            m = which == i
            out[m] = d.rvs(
                size=int(m.sum()),
                random_state=rng,
            )
        if size is None:
            return out[0]
        shape: tuple[int, ...] = (
            (size,)
            if isinstance(
                size,
                int,
            )
            else size
        )
        return out.reshape(shape)  # pyright: ignore[reportUnknownMemberType]
