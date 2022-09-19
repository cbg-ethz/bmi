import numpy as np
import pytest
from jax import random

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
