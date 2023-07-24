from typing import Union

import jax
import pytest

from bmi.samplers import SparseLVMParametrization, SplitStudentT
from bmi.samplers._tfp import MultivariateStudentDistribution, monte_carlo_mi_estimate


@pytest.mark.parametrize("dim_x", (2, 3))
@pytest.mark.parametrize("dim_y", (5,))
@pytest.mark.parametrize("df", (1, 2.0, 5))
def test_multivariate_normal(dim_x: int, dim_y: int, df: Union[int, float]) -> None:
    key = jax.random.PRNGKey(0)

    covariance = SparseLVMParametrization(
        dim_x=dim_x, dim_y=dim_y, n_interacting=min(dim_x, dim_y)
    ).covariance

    dist = MultivariateStudentDistribution(
        dim_x=dim_x,
        dim_y=dim_y,
        dispersion=covariance,
        df=df,
    )

    # Check whether the analytic MI formula is correct
    assert (
        pytest.approx(dist.analytic_mi)
        == SplitStudentT(
            dim_x=dim_x, dim_y=dim_y, dispersion=covariance, df=int(df)
        ).mutual_information()
    )
    # Check whether the Monte Carlo estimate is correct
    estimate, _ = monte_carlo_mi_estimate(key, dist, n=5_000)
    assert pytest.approx(estimate, abs=0.01, rel=0.05) == dist.analytic_mi
