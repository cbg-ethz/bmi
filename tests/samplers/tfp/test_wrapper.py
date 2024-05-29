import jax.numpy as jnp
import pytest

from bmi.samplers import bmm


def test_can_create_sampler() -> None:
    dist = bmm.MultivariateNormalDistribution(
        dim_x=1, dim_y=1, covariance=jnp.asarray([[1, 0.5], [0.5, 1]])
    )
    mi = -0.5 * jnp.log(1 - 0.5**2)

    sampler = bmm.FineSampler(dist=dist, mi_estimate_seed=0, mi_estimate_sample=1_000)

    x_sample, y_sample = sampler.sample(n_points=10, rng=0)
    assert x_sample.shape == (10, 1)
    assert y_sample.shape == (10, 1)

    assert sampler.mutual_information() == pytest.approx(mi, abs=0.01)
