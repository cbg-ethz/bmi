from typing import Optional, Union

import jax.numpy as jnp
from numpy.typing import ArrayLike
from tensorflow_probability.substrates import jax as tfp

from bmi.samplers._split_student_t import SplitStudentT
from bmi.samplers._tfp._core import JointDistribution

jtf = tfp.tf2jax
tfb = tfp.bijectors
tfd = tfp.distributions


def construct_multivariate_student_distribution(
    mean: jnp.ndarray,
    dispersion: jnp.ndarray,
    df: Union[int, float],
) -> tfd.MultivariateStudentTLinearOperator:
    """Constructs a multivariate Student distribution.

    Args:
        mean: location vector, shape `(dim,)`
        dispersion: dispersion matrix, shape `(dim, dim)`
        df: degrees of freedom
    """
    # Lower triangular matrix such that `dispersion = scale @ scale^T`
    scale = jnp.linalg.cholesky(dispersion)
    return tfd.MultivariateStudentTLinearOperator(
        loc=mean,
        scale=jtf.linalg.LinearOperatorLowerTriangular(scale),
        df=float(df),
    )


class MultivariateStudentDistribution(JointDistribution):
    """Multivariate Student distribution $P_{XY}$,
    such that $P_X$ is a multivariate Student distribution on the space
    of dimension `dim_x` and $P_Y$ is a multivariate Student distribution
    on the space of dimension `dim_y`.

    Note that the degrees of freedom `df` are the same for all distributions.
    """

    def __init__(
        self,
        *,
        dim_x: int,
        dim_y: int,
        df: int,
        dispersion: ArrayLike,
        mean: Optional[ArrayLike] = None
    ) -> None:
        """

        Args:
            dim_x: dimension of the $X$ support
            dim_y: dimension of the $Y$ support
            df: degrees of freedom
            mean: mean vector, shape `(n,)` where `n = dim_x + dim_y`.
                Default: zero vector
            dispersion: dispersion matrix, shape `(n, n)`
        """
        # The default mean vector is zero
        if mean is None:
            mean = jnp.zeros(dim_x + dim_y)
        mean = jnp.array(mean)
        dispersion = jnp.array(dispersion)

        # Calculate MI and implicitly validate the shapes
        analytic_mi = SplitStudentT(
            dim_x=dim_x,
            dim_y=dim_y,
            df=df,
            dispersion=dispersion,
            mean=mean,
        ).mutual_information()

        # Now we need to define the TensorFlow Probability distributions
        # using the information provided

        _dist_joint = construct_multivariate_student_distribution(
            mean=mean, dispersion=dispersion, df=df
        )
        dist_joint = tfd.TransformedDistribution(
            distribution=_dist_joint,
            bijector=tfb.Split((dim_x, dim_y)),
        )

        dist_x = construct_multivariate_student_distribution(
            mean=mean[:dim_x], dispersion=dispersion[:dim_x, :dim_x], df=df
        )
        dist_y = construct_multivariate_student_distribution(
            mean=mean[dim_x:], dispersion=dispersion[dim_x:, dim_x:], df=df
        )

        super().__init__(
            dim_x=dim_x,
            dim_y=dim_y,
            dist_joint=dist_joint,
            dist_x=dist_x,
            dist_y=dist_y,
            analytic_mi=analytic_mi,
        )

        self.df = df
        self.mean = mean
        self.dispersion = dispersion
