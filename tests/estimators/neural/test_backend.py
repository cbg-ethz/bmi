"""Tests of different backends."""
from typing import Tuple

import jax.numpy as jnp
import pytest

import bmi.estimators.neural._backend_linear_memory as lin
import bmi.estimators.neural._backend_quadratic_memory as quad
from bmi.samplers import SplitMultinormal


@pytest.mark.parametrize(
    "losses",
    [
        (lin.infonce, quad.infonce),
        (lin.donsker_varadhan, quad.donsker_varadhan),
        (lin.nwj, quad.nwj),
    ],
)
@pytest.mark.parametrize(
    "points_x_y",
    [
        (10, 2, 3),
        (12, 1, 5),
        (8, 3, 1),
        (4, 2, 2),
    ],
)
def test_losses_agree(losses: tuple, points_x_y: Tuple[int, int, int]) -> None:
    """Checks if two losses calculated using different backends agree

    Args:
        losses: tuple of two losses of signature (critic, xs, ys) -> float
            will be compared against each other
        points_x_y: tuple (n_points, dim_x, dim_y)
    """
    # Unpack the arguments
    loss1, loss2 = losses
    n_points, dim_x, dim_y = points_x_y

    covariance = jnp.eye(dim_x + dim_y)

    xs, ys = SplitMultinormal(dim_x=dim_x, dim_y=dim_y, covariance=covariance).sample(
        n_points=n_points, rng=3
    )

    def critic(x, y):
        return jnp.mean(x**2) - jnp.mean(y**3)

    result1 = loss1(critic, xs, ys)
    result2 = loss2(critic, xs, ys)

    assert result1 == pytest.approx(result2, abs=1e-4, rel=1e-3)
