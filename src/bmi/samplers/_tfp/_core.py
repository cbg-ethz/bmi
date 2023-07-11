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
    
    Attributes:
        dist: tfd.Distribution, joint distribution of X and Y
        dist_x: tfd.Distribution, marginal distribution of X
        dist_y: tfd.Distribution, marginal distribution of Y
        dim_x: dimension of the support of X
        dim_y: dimension of the support of Y
        analytic_mutual_information: analytical mutual information.
          Use `None` if unknown (in most cases)
    """
    dist_joint: tfd.Distribution
    dist_x: tfd.Distribution
    dist_y: tfd.Distribution
    dim_x: int
    dim_y: int
    analytic_mutual_information: Optional[float] = None

    def sample(key: jax.random.PRNGKeyArray, n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError
    
    def pmi(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError


def mixture(
    proportions: jnp.ndarray,
    components: Sequence[JointDistribution],
) -> JointDistribution:
    components = list(components)
    dim_xs = set(d.dim_x for d in components)
    dim_ys = set(d.dim_y for d in components)

    if len(dim_xs) != 1 or len(dim_ys) != 1:
        raise ValueError(
            "All components must have the same dimensionality for x and y.")

    dim_x = dim_xs.pop()
    dim_y = dim_ys.pop()

    dist_joint = tfd.Mixture(
        cat=tfd.Categorical(probs=proportions),
        components=[d.dist_joint for d in components])
    dist_x = tfd.Mixture(
        cat=tfd.Categorical(probs=proportions),
        components=[d.dist_x for d in components])
    dist_y = tfd.Mixture(
        cat=tfd.Categorical(proportions),
        components=[d.dist_y for d in components])

    return JointDistribution(
        dist_joint=dist_joint,
        dist_x=dist_x,
        dist_y=dist_y,
        dim_x=dim_x,
        dim_y=dim_y,
        analytic_mutual_information=None,
    )


def transform(
    dist: JointDistribution,
    x_transform: Optional[tfb.Bijector] = None,
    y_transform: Optional[tfb.Bijector] = None,
) -> JointDistribution:
    if x_transform is None:
        x_transform = tfb.Identity()
    if y_transform is None:
        y_transform = tfb.Identity()
    
    product_bijector = tfb.Blockwise(bijectors=[x_transform, y_transform], block_sizes=[dist.dim_x, dist.dim_y])
    return JointDistribution(
        dim_x=dist.dim_x,
        dim_y=dist.dim_y,
        dist_joint=tfd.TransformedDistribution(distribution=dist.dist_joint, bijector=product_bijector),
        dist_x=tfd.TransformedDistribution(distribution=dist.dist_x, bijector=x_transform),
        dist_y=tfd.TransformedDistribution(distribution=dist.dist_y, bijector=y_transform),
        analytic_mutual_information=dist.analytical_mutual_information
    )


def pmi_profile(key: jax.random.PRNGKeyArray, dist: JointDistribution, n: int) -> jnp.ndarray:
    x, y = dist.sample(key, n)
    return dist.pmi(x, y)  # TODO(Pawel): Check if vectorizes properly


def monte_carlo_mi_estimate(key: jax.random.PRNGKeyArray, dist: JointDistribution, n: int) -> tuple[float, float]:
    """Estimates the mutual information between X and Y using Monte Carlo sampling.
    
    Returns:
        float, mutual information estimate
        float, standard error estimate
    
    Note:
        It is worth to run this procedure multiple times and see whether
        the standard error estimate is stable.
    """
    profile = pmi_profile(key=key, dist=dist, n=n)
    mi_estimate = jnp.mean(profile)
    standard_error = jnp.std(profile) / jnp.sqrt(n)

    return mi_estimate, standard_error

