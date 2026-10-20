from typing import TypedDict

import numpy as np
import pytest

from numpy.typing import (
    ArrayLike,
    NDArray,
)
from scipy import stats

from fcaz.stats import (
    Mixture,
    chf,
    hf,
    invexpon,
)
from fcaz.typing import FrozenDistribution

SEEDS = range(20)


class Draw(TypedDict):
    """Random inputs shared by the tests."""

    theta1: float | int
    theta2: float | int
    w: float | int
    x: NDArray[np.float64]
    q: NDArray[np.float64]


def close(
    a: ArrayLike,
    b: ArrayLike,
) -> bool:
    """Whether every element of a is numerically close to b."""
    return bool(np.allclose(a, b))


def draw(seed: int) -> Draw:
    """Random scales, mixing weight, evaluation points and probabilities for a seed."""
    rng = np.random.default_rng(seed)
    return {
        'theta1': rng.uniform(
            0.5,
            20,
        ),
        'theta2': rng.uniform(
            0.5,
            20,
        ),
        'w': rng.uniform(
            0.05,
            0.95,
        ),
        'x': rng.uniform(
            0.1,
            30,
            size=5,
        ),
        'q': rng.uniform(
            0.01,
            0.99,
            size=5,
        ),
    }


@pytest.mark.parametrize(
    'seed',
    SEEDS,
)
def test_invexpon_matches_invweibull_and_closed_form(seed: int) -> None:
    """invexpon agrees with its closed-form cdf and with scipy's invweibull at c=1."""
    p = draw(seed)
    th, x, q = p['theta1'], p['x'], p['q']
    d: FrozenDistribution = invexpon(scale=th)
    ref: FrozenDistribution = stats.invweibull(
        c=1,
        scale=th,
    )
    assert close(
        d.cdf(x),
        np.exp(-th / x),
    )
    assert close(
        d.cdf(x),
        ref.cdf(x),
    )
    assert close(
        d.pdf(x),
        ref.pdf(x),
    )
    assert close(
        d.sf(x),
        ref.sf(x),
    )
    assert close(
        d.ppf(q),
        ref.ppf(q),
    )


@pytest.mark.parametrize(
    'seed',
    SEEDS,
)
def test_mixture_sf_and_chf_closed_form(seed: int) -> None:
    """Mixture sf, cdf, chf and hf match closed forms for an exponential and invexpon mix."""
    p = draw(seed)
    t1, t2, w, x = p['theta1'], p['theta2'], p['w'], p['x']
    m = Mixture(
        [stats.expon(scale=t1), invexpon(scale=t2)],
        [w, 1 - w],
    )
    s = w * np.exp(-x / t1) + (1 - w) * (1 - np.exp(-t2 / x))
    f = w * np.exp(-x / t1) / t1 + (1 - w) * np.exp(-t2 / x) * t2 / x**2
    assert close(
        m.sf(x),
        s,
    )
    assert close(
        m.cdf(x),
        1 - s,
    )
    assert close(
        m.chf(x),
        -np.log(s),
    )
    assert close(
        m.hf(x),
        f / s,
    )


@pytest.mark.parametrize(
    'seed',
    SEEDS,
)
def test_hf_chf_helpers_on_exponential(seed: int) -> None:
    """hf and chf of an exponential give the hazard 1/theta and cumulative hazard x/theta."""
    p = draw(seed)
    th, x = p['theta1'], p['x']
    e: FrozenDistribution = stats.expon(scale=th)
    assert close(
        chf(
            e,
            x,
        ),
        x / th,
    )
    assert close(
        hf(
            e,
            x,
        ),
        1 / th,
    )


@pytest.mark.parametrize(
    'seed',
    SEEDS,
)
def test_mixture_ppf_roundtrip_and_mean(seed: int) -> None:
    """Mixture ppf inverts cdf, and the mean and sample mean match the weighted component means."""
    p = draw(seed)
    t1, t2, w, q = p['theta1'], p['theta2'], p['w'], p['q']
    m = Mixture(
        [stats.expon(scale=t1), invexpon(scale=t2)],
        [w, 1 - w],
    )
    assert close(
        m.cdf(m.ppf(q)),
        q,
    )
    m2 = Mixture(
        [stats.expon(scale=t1), stats.expon(scale=t2)],
        [w, 1 - w],
    )
    assert close(
        m2.mean(),
        w * t1 + (1 - w) * t2,
    )
    r = m2.rvs(
        size=100_000,
        random_state=seed,
    )
    assert abs(r.mean() - m2.mean()) < 0.05 * m2.mean()


def test_mixture_rejects_bad_weights() -> None:
    """Mixture raises ValueError for weights that do not sum to 1 or have the wrong length."""
    e: FrozenDistribution = stats.expon()
    with pytest.raises(ValueError):
        Mixture(
            [e, e],
            [0.5, 0.6],
        )
    with pytest.raises(ValueError):
        Mixture(
            [e, e],
            [1.0],
        )
