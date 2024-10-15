import jax
import jax.numpy as jnp
import pytest

import bmi.samplers
from bmi.samplers import bmm


def test_product_distribution(dim_x: int = 2, dim_y: int = 3, n_points: int = 10) -> None:
    assert dim_y >= dim_x, "We construct canonical correlation matrix, so we want this constraint."

    dist_dependent = bmm.MultivariateNormalDistribution(
        dim_x=dim_x,
        dim_y=dim_y,
        covariance=bmi.samplers.canonical_correlation(
            jnp.full((dim_x,), fill_value=0.5), additional_y=dim_y - dim_x
        ),
    )
    dist_independent = bmm.ProductDistribution(
        dist_x=dist_dependent.dist_x, dist_y=dist_dependent.dist_y
    )

    assert dist_independent.analytic_mi == pytest.approx(0.0)

    xs, ys = dist_independent.sample(n_points=n_points, key=jax.random.PRNGKey(0))
    pmis = dist_independent.pmi(xs, ys)

    assert pmis.min() == pytest.approx(0.0)
    assert pmis.max() == pytest.approx(0.0)

    # TODO(Pawel): Test whether some summary statistics of the marginal distributions are the same
