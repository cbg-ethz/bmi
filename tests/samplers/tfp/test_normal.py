import jax
import jax.numpy as jnp
import pytest

from bmi.samplers._tfp import MultivariateNormalDistribution
from bmi.samplers._splitmultinormal import BivariateNormalSampler

def test_sample_1v1(correlation: float = 0.5, n: int = 10):
    dist = MultivariateNormalDistribution(dim_x=1, dim_y=1, covariance=[[1, correlation], [correlation, 1]])

    key = jax.random.PRNGKey(0)

    x, y = dist.sample(key, n=n)

    assert x.shape == (n, 1)
    assert y.shape == (n, 1)

    assert pytest.approx(dist.analytic_mi) == BivariateNormalSampler(correlation=correlation).mutual_information()

