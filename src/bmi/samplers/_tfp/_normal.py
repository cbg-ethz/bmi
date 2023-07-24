from typing import Optional

import jax.numpy as jnp
from numpy.typing import ArrayLike
from tensorflow_probability.substrates import jax as tfp

from bmi.samplers._splitmultinormal import SplitMultinormal
from bmi.samplers._tfp._core import JointDistribution

jtf = tfp.tf2jax
tfd = tfp.distributions


def _construct_multivariate_distribution(
    mean: jnp.ndarray, covariance: jnp.ndarray
) -> tfd.MultivariateNormalLinearOperator:
    # Lower triangular matrix such that `covariance = scale @ scale^T`
    scale = jnp.linalg.cholesky(covariance)
    return tfd.MultivariateNormalLinearOperator(
        loc=mean,
        scale=jtf.linalg.LinearOperatorLowerTriangular(scale),
    )


class MultivariateNormalDistribution(JointDistribution):
    def __init__(
        self, *, dim_x: int, dim_y: int, covariance: ArrayLike, mean: Optional[ArrayLike] = None
    ) -> None:
        """

        Args:
            dim_x: dimension of the X space
            dim_y: dimension of the Y space
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

        dist_joint = _construct_multivariate_distribution(mean=mean, covariance=covariance)
        dist_x = _construct_multivariate_distribution(
            mean=mean[:dim_x], covariance=covariance[:dim_x, :dim_x]
        )
        dist_y = _construct_multivariate_distribution(
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
