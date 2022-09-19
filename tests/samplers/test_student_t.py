import numpy as np
import pytest
from jax import random
from sklearn.feature_selection import mutual_info_regression

import bmi.samplers.split_student_t as student


@pytest.mark.parametrize("x", (2, 4))
@pytest.mark.parametrize("y", (1, 2))
@pytest.mark.parametrize("nu", (4, 20))
@pytest.mark.parametrize("n_samples", (1000,))
def test_samples_produced(x: int, y: int, n_samples: int, nu: int) -> None:
    """Tests whether the sampling returns the right shapes."""

    rng = random.PRNGKey(111)
    mean = random.uniform(rng, shape=(x + y,))

    sampler = student.SplitStudentT(dim_x=x, dim_y=y, mean=mean, dispersion=np.eye(x + y), nu=nu)

    x_sample, y_sample = sampler.sample(n_points=n_samples, rng=42)

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
        mean, xy_sample.mean(axis=0), rtol=0.1, atol=0.05
    ), f"Arrays different: {mean} != {xy_sample.mean(axis=0)}"


@pytest.mark.parametrize("correlation", (0.2, 0.8))
@pytest.mark.parametrize("nu", (2, 4, 10))
def test_2d(
    correlation: float, nu: int, var_x: float = 1.0, var_y: float = 1.0, n_samples: int = 1000
) -> None:
    cov_xy = correlation * np.sqrt(var_x * var_y)

    dispersion = np.asarray(
        [
            [var_x, cov_xy],
            [cov_xy, var_y],
        ]
    )

    sampler = student.SplitStudentT(
        dim_x=1,
        dim_y=1,
        mean=np.zeros(2),
        dispersion=dispersion,
        nu=nu,
    )

    x, y = sampler.sample(n_points=n_samples, rng=10)
    # According to API, y must have shape (n,) rather than (n, 1)
    mi_estimate = mutual_info_regression(x, y.ravel(), random_state=5, n_neighbors=10)

    assert sampler.mutual_information() == pytest.approx(mi_estimate, rel=0.05)
