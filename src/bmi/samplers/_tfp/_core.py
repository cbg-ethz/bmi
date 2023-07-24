import dataclasses
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


@dataclasses.dataclass
class JointDistribution:
    """Represents a joint distribution of X and Y with known marginals.
    This is the main object of this package.

    Attributes:
        dist: tfd.Distribution, joint distribution of X and Y
        dist_x: tfd.Distribution, marginal distribution of X
        dist_y: tfd.Distribution, marginal distribution of Y
        dim_x: dimension of the support of X
        dim_y: dimension of the support of Y
        analytic_mi: analytical mutual information.
          Use `None` if unknown (in most cases)
    """

    dist_joint: tfd.Distribution
    dist_x: tfd.Distribution
    dist_y: tfd.Distribution
    dim_x: int
    dim_y: int
    analytic_mi: Optional[float] = None

    def sample(self, key: jax.random.PRNGKeyArray, n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sample from the joint distribution.

        Args:
            key: JAX random key
            n: number of samples to draw
        """
        if n < 1:
            raise ValueError("n must be positive")

        xy = self.dist_joint.sample(seed=key, sample_shape=(n,))
        return xy[..., : self.dim_x], xy[..., self.dim_x :]  # noqa: E203 (formatting discrepancy)

    def pmi(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Calculates pointwise mutual information at specified points.

        Args:
            x: points in the X space, shape `(n_points, dim_x)`
            y: points in the Y space, shape `(n_points, dim_y)`

        Returns:
            pointwise mutual information evaluated at (x, y) points,
              shape `(n_points,)`

        Note:
            This function is vectorized, i.e. it can calculate PMI for multiple points at once.
        """
        log_pxy = self.dist_joint.log_prob(jnp.hstack([x, y]))
        log_px = self.dist_x.log_prob(x)
        log_py = self.dist_y.log_prob(y)

        return log_pxy - (log_px + log_py)


def mixture(
    proportions: jnp.ndarray,
    components: Sequence[JointDistribution],
) -> JointDistribution:
    """Constructs a mixture distribution.

    Args:
        proportions: mixture proportions should be positive and sum up to 1,
          shape `(n_components,)`
        components: sequence of `JointDistribution` objects which will be mixed

    Returns:
        mixture distribution
    """
    proportions = jnp.asarray(proportions)
    components = list(components)
    dim_xs = set(d.dim_x for d in components)
    dim_ys = set(d.dim_y for d in components)

    if len(dim_xs) != 1 or len(dim_ys) != 1:
        raise ValueError("All components must have the same dimensionality for x and y.")

    if len(components) != len(proportions):
        raise ValueError("Number of components must match the number of proportions.")

    dim_x = dim_xs.pop()
    dim_y = dim_ys.pop()

    dist_joint = tfd.Mixture(
        cat=tfd.Categorical(probs=proportions), components=[d.dist_joint for d in components]
    )
    dist_x = tfd.Mixture(
        cat=tfd.Categorical(probs=proportions), components=[d.dist_x for d in components]
    )
    dist_y = tfd.Mixture(
        cat=tfd.Categorical(probs=proportions), components=[d.dist_y for d in components]
    )

    return JointDistribution(
        dist_joint=dist_joint,
        dist_x=dist_x,
        dist_y=dist_y,
        dim_x=dim_x,
        dim_y=dim_y,
        analytic_mi=None,
    )


def transform(
    dist: JointDistribution,
    x_transform: Optional[tfb.Bijector] = None,
    y_transform: Optional[tfb.Bijector] = None,
) -> JointDistribution:
    """For given diffeomorphisms `f` and `g` transforms the joint distribution P_{XY}
    into P_{f(X)g(Y)}.

    Args:
        dist: distribution to be transformed
        x_transform: diffeomorphism to transform X. Defaults to identity.
        y_transform: diffeomorphism to transform Y. Defaults to identity.

    Returns:
        transformed distribution
    """
    if x_transform is None:
        x_transform = tfb.Identity()
    if y_transform is None:
        y_transform = tfb.Identity()

    product_bijector = tfb.Blockwise(
        bijectors=[x_transform, y_transform], block_sizes=[dist.dim_x, dist.dim_y]
    )
    return JointDistribution(
        dim_x=dist.dim_x,
        dim_y=dist.dim_y,
        dist_joint=tfd.TransformedDistribution(
            distribution=dist.dist_joint, bijector=product_bijector
        ),
        dist_x=tfd.TransformedDistribution(distribution=dist.dist_x, bijector=x_transform),
        dist_y=tfd.TransformedDistribution(distribution=dist.dist_y, bijector=y_transform),
        analytic_mi=dist.analytic_mi,
    )


def pmi_profile(key: jax.random.PRNGKeyArray, dist: JointDistribution, n: int) -> jnp.ndarray:
    """Monte Carlo draws a sample of size `n` from the PMI distribution.

    Args:
        key: JAX random key, used to generate the sample
        dist: distribution
        n: number of points to sample

    Returns:
        PMI profile, shape `(n,)`
    """
    x, y = dist.sample(key, n)
    return dist.pmi(x, y)


def monte_carlo_mi_estimate(
    key: jax.random.PRNGKeyArray, dist: JointDistribution, n: int
) -> tuple[float, float]:
    """Estimates the mutual information between X and Y using Monte Carlo sampling.

    Returns:
        float, mutual information estimate
        float, standard error estimate

    Note:
        It is worth to run this procedure multiple times and see whether
        the standard error estimate is accurate.
    """
    profile = pmi_profile(key=key, dist=dist, n=n)
    mi_estimate = jnp.mean(profile)
    standard_error = jnp.std(profile) / jnp.sqrt(n)

    return mi_estimate, standard_error
