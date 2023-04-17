import jax.numpy as jnp
from jax.scipy.special import erf


def normal_cdf(x):
    return 0.5 * (1 + erf(x / 2**0.5))


def half_cube(x: float) -> float:
    """The mapping x * sqrt(x), i.e., a signed version of x^(3/2) mapping."""
    return x * jnp.sqrt(jnp.abs(x))
