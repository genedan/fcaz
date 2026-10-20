"""
Type aliases and protocols shared across fcaz.

``FrozenDistribution`` and the smaller ``Supports*`` protocols describe the methods fcaz needs
from a frozen distribution, so any frozen ``scipy.stats`` distribution or fcaz object can be
passed where one is expected.
"""

from collections.abc import Sequence
from typing import (
    Any,
    Protocol,
    TypeAlias,
)

import numpy as np

from numpy.typing import NDArray

__all__ = [
    'ArrayLikeFloat',
    'FloatOrArray',
    'FrozenDistribution',
    'SupportsLogSf',
    'SupportsPdfSf',
]

ArrayLikeFloat: TypeAlias = (
    float
    | np.floating[Any]
    | NDArray[np.number[Any]]
    | Sequence[float | np.floating[Any] | NDArray[np.number[Any]]]
)
FloatOrArray: TypeAlias = np.float64 | NDArray[np.float64]


class SupportsPdfSf(Protocol):
    """Any object with ``pdf`` and ``sf`` methods, such as a frozen scipy distribution."""

    def pdf(self, x: Any) -> Any:
        """Probability density function."""
        ...

    def sf(self, x: Any) -> Any:
        """Survival function."""
        ...


class SupportsLogSf(Protocol):
    """Any object with a ``logsf`` method, such as a frozen scipy distribution."""

    def logsf(self, x: Any) -> Any:
        """Log of the survival function."""
        ...


class FrozenDistribution(SupportsPdfSf, SupportsLogSf, Protocol):
    """The frozen-distribution interface that Mixture needs from its components."""

    def cdf(self, x: Any) -> Any:
        """Cumulative distribution function."""
        ...

    def ppf(self, q: Any) -> Any:
        """Percent point function."""
        ...

    def mean(self) -> Any:
        """Mean."""
        ...

    def rvs(
        self,
        size: Any = None,
        random_state: Any = None,
    ) -> Any:
        """Random variates."""
        ...
