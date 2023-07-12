import jax
import pytest
from tensorflow_probability.substrates import jax as tfp

import bmi.samplers._tfp as bmi_tfp
from bmi.samplers import SparseLVMParametrization

tfb = tfp.bijectors


def distributions(dim_x: int = 2, dim_y: int = 3) -> list[bmi_tfp.JointDistribution]:
    dispersion = SparseLVMParametrization(
        dim_x=dim_x, dim_y=dim_y, n_interacting=min(dim_x, dim_y)
    ).covariance

    normal = bmi_tfp.MultivariateNormalDistribution(
        dim_x=dim_x, dim_y=dim_y, covariance=dispersion
    )
    student = bmi_tfp.MultivariateStudentDistribution(
        dim_x=dim_x, dim_y=dim_y, dispersion=dispersion, df=3
    )

    return [
        bmi_tfp.MultivariateNormalDistribution(dim_x=1, dim_y=1, covariance=[[1, 0.5], [0.5, 1]]),
        normal,
        student,
        bmi_tfp.mixture(proportions=[0.3, 0.7], components=[normal, student]),
    ]


@pytest.mark.parametrize("dist", distributions())
def test_sample_and_pmi(dist: bmi_tfp.JointDistribution, n_samples: int = 10) -> None:
    """Checks whether we can sample from the distribution and calculate PMI."""
    x, y = dist.sample(jax.random.PRNGKey(0), n=n_samples)

    assert x.shape == (n_samples, dist.dim_x)
    assert y.shape == (n_samples, dist.dim_y)

    pmis = dist.pmi(x, y)
    assert pmis.shape == (n_samples,)


@pytest.mark.parametrize("dist", distributions())
def test_transformed(dist: bmi_tfp.JointDistribution, n_points: int = 1_000) -> None:
    base_dist = dist

    transformed = bmi_tfp.transform(
        dist=base_dist,
        x_transform=tfb.NormalCDF(),
        y_transform=tfb.Sinh(),
    )

    key = jax.random.PRNGKey(0)

    x_base, y_base = base_dist.sample(key, n=n_points)
    x_tran, y_tran = transformed.sample(key, n=n_points)

    # Check shapes
    assert x_base.shape == x_tran.shape
    assert y_base.shape == y_tran.shape
    assert x_tran.shape == (n_points, base_dist.dim_x)
    assert y_tran.shape == (n_points, base_dist.dim_y)

    # Check whether the analytic MI formula is correct
    assert pytest.approx(transformed.analytic_mi) == base_dist.analytic_mi
