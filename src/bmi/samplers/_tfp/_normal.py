from typing import Optional

import jax.numpy as jnp
from numpy.typing import ArrayLike
from tensorflow_probability.substrates import jax as tfp

from bmi.samplers._splitmultinormal import SplitMultinormal
from bmi.samplers._tfp._core import JointDistribution

jtf = tfp.tf2jax
tfd = tfp.distributions


def construct_multivariate_normal_distribution(
    mean: jnp.ndarray, covariance: jnp.ndarray
) -> tfd.MultivariateNormalLinearOperator:
    """Constructs a multivariate normal distribution."""
    # Lower triangular matrix such that `covariance = scale @ scale^T`
    scale = jnp.linalg.cholesky(covariance)
    return tfd.MultivariateNormalLinearOperator(
        loc=mean,
        scale=jtf.linalg.LinearOperatorLowerTriangular(scale),
    )


class MultivariateNormalDistribution(JointDistribution):
    """Multivariate normal distribution $P_{XY}$,
    such that $P_X$ is a multivariate normal distribution on the space
    of dimension `dim_x` and $P_Y$ is a multivariate normal distribution
    on the space of dimension `dim_y`."""

    def __init__(
        self, *, dim_x: int, dim_y: int, covariance: ArrayLike, mean: Optional[ArrayLike] = None
    ) -> None:
        """

        Args:
            dim_x: dimension of the $X$ support
            dim_y: dimension of the $Y$ support
            mean: mean vector, shape `(n,)` where `n = dim_x + dim_y`.
                Default: zero vector
            covariance: covariance matrix, shape (n, n)
        """
        # The default mean vector is zero
        if mean is None:
            mean = jnp.zeros(dim_x + dim_y)
        mean = jnp.array(mean)
        covariance = jnp.array(covariance)

        # Calculate MI and implicitly validate the shapes
        analytic_mi = SplitMultinormal(
            dim_x=dim_x, dim_y=dim_y, mean=mean, covariance=covariance
        ).mutual_information()

        # Now we need to define the TensorFlow Probability distributions
        # using the information provided

        dist_joint = construct_multivariate_normal_distribution(mean=mean, covariance=covariance)
        dist_x = construct_multivariate_normal_distribution(
            mean=mean[:dim_x], covariance=covariance[:dim_x, :dim_x]
        )
        dist_y = construct_multivariate_normal_distribution(
            mean=mean[dim_x:], covariance=covariance[dim_x:, dim_x:]
        )

        super().__init__(
            dim_x=dim_x,
            dim_y=dim_y,
            dist_joint=dist_joint,
            dist_x=dist_x,
            dist_y=dist_y,
            analytic_mi=analytic_mi,
        )

        self.mean = mean
        self.covariance = covariance
