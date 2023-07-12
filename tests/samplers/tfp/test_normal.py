import jax
import pytest

from bmi.samplers import BivariateNormalSampler, SparseLVMParametrization, SplitMultinormal
from bmi.samplers._tfp import MultivariateNormalDistribution, monte_carlo_mi_estimate


def test_1v1(correlation: float = 0.5, n: int = 10):
    dist = MultivariateNormalDistribution(
        dim_x=1, dim_y=1, covariance=[[1, correlation], [correlation, 1]]
    )

    key = jax.random.PRNGKey(0)

    x, y = dist.sample(key, n=n)

    assert x.shape == (n, 1)
    assert y.shape == (n, 1)

    # Check whether the analytic MI formula is correct
    assert (
        pytest.approx(dist.analytic_mi)
        == BivariateNormalSampler(correlation=correlation).mutual_information()
    )
    # Check whether the Monte Carlo estimate is correct
    estimate, _ = monte_carlo_mi_estimate(key, dist, n=5_000)
    assert pytest.approx(estimate, abs=0.01) == dist.analytic_mi


@pytest.mark.parametrize("dim_x", (2, 3))
@pytest.mark.parametrize("dim_y", (5,))
def test_multivariate_normal(dim_x: int, dim_y: int) -> None:
    key = jax.random.PRNGKey(0)

    covariance = SparseLVMParametrization(
        dim_x=dim_x, dim_y=dim_y, n_interacting=min(dim_x, dim_y)
    ).covariance

    dist = MultivariateNormalDistribution(
        dim_x=dim_x,
        dim_y=dim_y,
        covariance=covariance,
    )

    # Check whether the analytic MI formula is correct
    assert (
        pytest.approx(dist.analytic_mi)
        == SplitMultinormal(dim_x=dim_x, dim_y=dim_y, covariance=covariance).mutual_information()
    )
    # Check whether the Monte Carlo estimate is correct
    estimate, _ = monte_carlo_mi_estimate(key, dist, n=5_000)
    assert pytest.approx(estimate, abs=0.01, rel=0.05) == dist.analytic_mi
