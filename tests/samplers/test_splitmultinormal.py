import numpy as np
import pytest  # pytype: disable=import-error
from jax import random  # pytype: disable=import-error

from bmi.samplers.splitmultinormal import SplitMultinormal


@pytest.mark.parametrize("y", (2, 5, 10))
@pytest.mark.parametrize("x", (1, 2, 3))
@pytest.mark.parametrize("variance", (0.5, 1, 2))
def test_symmetric_gaussian(x: int, y: int, variance: float, n_samples: int = 2_000) -> None:
    """Tests if the standard Gaussian has MI = 0."""

    sampler = SplitMultinormal(
        dim_x=x,
        dim_y=y,
        mean=np.zeros(x + y),
        covariance=np.eye(x + y),
    )

    assert sampler.dim_x == x, f"x dim: {sampler.dim_x} != {x}"
    assert sampler.dim_y == y, f"y dim: {sampler.dim_x} != {x}"
    assert sampler.dim_total == x + y, f"total dim: {sampler.dim_total} != {x + y}"

    sampler_mi = sampler.mutual_information()
    assert sampler_mi == pytest.approx(0, abs=1e-3), f"MI wrong: {sampler_mi} != 0"

    rng = random.PRNGKey(111)
    x_sample, y_sample = sampler.sample(n_points=n_samples, rng=rng)

    assert x_sample.shape == (
        n_samples,
        x,
    ), f"X sample shape: {x_sample.shape} != {(n_samples, x)}"
    assert y_sample.shape == (
        n_samples,
        y,
    ), f"Y sample shape: {y_sample.shape} != {(n_samples, y)}"

    xy_sample = np.hstack([x_sample, y_sample])
    assert xy_sample.shape == (
        n_samples,
        x + y,
    ), f"XxY sample shape: {xy_sample.shape} != {(n_samples, x+y)}"

    assert np.allclose(
        np.zeros(x + y), xy_sample.mean(axis=0), atol=0.1 * max(1, variance)
    ), f"Arrays different: {np.zeros(x+y)} != {xy_sample.mean(axis=0)}"
