import numpy as np
import pytest  # pytype: disable=import-error
from jax import random  # pytype: disable=import-error

from bmi.samplers.splitmultinormal import SplitMultinormal  # pytype: disable=import-error


@pytest.mark.parametrize("y", (2, 5, 10))
@pytest.mark.parametrize("x", (1, 2, 3))
@pytest.mark.parametrize("variance", (0.5, 1, 2))
def test_symmetric_gaussian_mi(x: int, y: int, variance: float, n_samples: int = 2_000) -> None:
    """Tests if the standard Gaussian has MI = 0."""

    rng = random.PRNGKey(111)
    rng, mean_rng = random.split(rng)

    mean = random.uniform(mean_rng, shape=(x + y,))

    sampler = SplitMultinormal(
        dim_x=x,
        dim_y=y,
        mean=mean,
        covariance=variance * np.eye(x + y),
    )

    assert sampler.dim_x == x, f"x dim: {sampler.dim_x} != {x}"
    assert sampler.dim_y == y, f"y dim: {sampler.dim_x} != {x}"
    assert sampler.dim_total == x + y, f"total dim: {sampler.dim_total} != {x + y}"

    sampler_mi = sampler.mutual_information()
    assert sampler_mi == pytest.approx(0, abs=1e-3), f"MI wrong: {sampler_mi} != 0"

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
        mean, xy_sample.mean(axis=0), rtol=0.05, atol=0.07
    ), f"Arrays different: {mean} != {xy_sample.mean(axis=0)}"


@pytest.mark.parametrize("mean_xy", [(0.1, 0.4), (0.4, -0.9)])
@pytest.mark.parametrize("std_xy", [(0.5, 1.0)])
@pytest.mark.parametrize("correlation", [0.9, -0.5])
def test_2d_gaussian(
    mean_xy: tuple[float, float],
    std_xy: tuple[float, float],
    correlation: float,
    n_samples: int = 1500,
) -> None:
    """Tests whether the samples from 2D Gaussian look right. We analyze both means
    and the full covariance matrix.

    Args:
        mean_xy: tuple representing means of two 1D Gaussians: (mean_x, mean_y)
        std_xy: tuple representing std of two 1D Gaussians: (std_x, std_y)
        correlation: correlation between X and Y
    """

    var_x = std_xy[0] ** 2
    var_y = std_xy[1] ** 2
    cov_xy = std_xy[0] * std_xy[1] * correlation

    mean = np.asarray(mean_xy)
    covariance = np.asarray(
        [
            [var_x, cov_xy],
            [cov_xy, var_y],
        ]
    )

    sampler = SplitMultinormal(
        dim_x=1,
        dim_y=1,
        mean=mean,
        covariance=covariance,
    )

    rng = random.PRNGKey(10)
    x_sample, y_sample = sampler.sample(n_samples, rng=rng)

    assert x_sample.shape == (n_samples, 1), f"X shape wrong: {x_sample.shape}"
    assert y_sample.shape == (n_samples, 1), f"Y shape wrong: {x_sample.shape}"

    assert np.mean(x_sample) == pytest.approx(
        mean_xy[0], rel=0.1, abs=0.03
    ), f"X mean wrong: {np.mean(x_sample)} != {mean_xy[0]}"
    assert np.mean(y_sample) == pytest.approx(
        mean_xy[1], rel=0.1, abs=0.03
    ), f"Y mean wrong: {np.mean(y_sample)} != {mean_xy[1]}"

    assert np.std(x_sample) == pytest.approx(
        std_xy[0], rel=0.1, abs=0.03
    ), f"X std wrong: {np.std(x_sample)} != {std_xy[0]}"
    assert np.std(y_sample) == pytest.approx(
        std_xy[1], rel=0.1, abs=0.03
    ), f"Y std wrong: {np.std(y_sample)} != {std_xy[1]}"

    correlation_estimate = np.corrcoef(x_sample.ravel(), y_sample.ravel())[0, 1]
    assert correlation_estimate == pytest.approx(correlation, rel=0.1)
