import numpy as np
import pytest
from jax import random
from sklearn.feature_selection import mutual_info_regression

from bmi.samplers._splitmultinormal import BivariateNormalSampler, SplitMultinormal


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
    ), f"XxY sample shape: {xy_sample.shape} != {(n_samples, x + y)}"

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


@pytest.mark.parametrize("correlation", (0.2, 0.8))
@pytest.mark.parametrize("var_x", (1.0, 2.0))
def test_2d_mi(
    correlation: float, var_x: float, var_y: float = 1.0, n_samples: int = 1000
) -> None:
    cov_xy = correlation * np.sqrt(var_x * var_y)

    covariance = np.asarray(
        [
            [var_x, cov_xy],
            [cov_xy, var_y],
        ]
    )

    sampler = SplitMultinormal(
        dim_x=1,
        dim_y=1,
        covariance=covariance,
    )

    rng = random.PRNGKey(20)

    x, y = sampler.sample(n_points=n_samples, rng=rng)
    # We need to reshape `y` from (n, 1) to just (n,)
    mi_estimate = mutual_info_regression(x, y.ravel(), random_state=5)[0]

    assert sampler.mutual_information() == pytest.approx(mi_estimate, rel=0.05, abs=0.06)


@pytest.mark.parametrize("seed", (3, np.int64(5), np.int32(12)))
def test_rng_integer(seed: int, n_points: int = 10, dim_x: int = 3, dim_y: int = 2) -> None:
    """Tests whether we can pass an integer as a random seed without error."""
    sampler = SplitMultinormal(dim_x=dim_x, dim_y=dim_y, covariance=np.eye(dim_x + dim_y))

    x1, y1 = sampler.sample(n_points, rng=seed)
    x2, y2 = sampler.sample(n_points, rng=random.PRNGKey(seed))

    assert np.allclose(x1, x2)
    assert np.allclose(y1, y2)


@pytest.mark.parametrize("dim_x", (2, 3))
@pytest.mark.parametrize("dim_y", (1, 5))
def test_default_zero(dim_x: int, dim_y: int) -> None:
    """Tests whether the default mean vector is the zero vector."""
    sampler = SplitMultinormal(
        dim_x=dim_x,
        dim_y=dim_y,
        covariance=np.eye(dim_x + dim_y),
    )

    assert np.allclose(sampler._mean, np.zeros(dim_x + dim_y))


@pytest.mark.parametrize("correlation", [0.1, 0.6])
@pytest.mark.parametrize("std_x", [0.1, 2.0])
@pytest.mark.parametrize("std_y", [0.2, 3.0])
@pytest.mark.parametrize("mean_x", [0.1, 2.2])
@pytest.mark.parametrize("mean_y", [0.5, 1.1])
def test_bivariate_normal(
    correlation: float, std_x: float, std_y: float, mean_x: float, mean_y: float
) -> None:
    sampler = BivariateNormalSampler(
        correlation=correlation, std_x=std_x, std_y=std_y, mean_x=mean_x, mean_y=mean_y
    )

    # Test means
    x, y = sampler.sample(n_points=5000, rng=42)

    assert mean_x == pytest.approx(np.mean(x), abs=0.02 * std_x)
    assert mean_y == pytest.approx(np.mean(y), abs=0.02 * std_y)

    assert std_x == pytest.approx(np.std(x), rel=0.05)
    assert std_y == pytest.approx(np.std(y), rel=0.05)

    cov = np.asarray(
        [
            [1.0, correlation],
            [correlation, 1.0],
        ]
    )

    assert sampler.correlation() == pytest.approx(correlation)
    assert sampler.mutual_information() == pytest.approx(
        SplitMultinormal(dim_x=1, dim_y=1, covariance=cov).mutual_information()
    )
