import jax.numpy as jnp
from jax.scipy.special import erf


def normal_cdf(x):
    return 0.5 * (1 + erf(x / 2**0.5))


def half_cube(x: float) -> float:
    """The mapping x * sqrt(x), i.e., a signed version of x^(3/2) mapping."""
    return x * jnp.sqrt(jnp.abs(x))


def swissroll2d(x: jnp.ndarray) -> jnp.ndarray:
    """
    Args:
        x: array of shape (1,) representing number in range [0, 1]

    Returns:
        array of shape (2,)
    """
    # Rescale and shift the variable
    t = 1.5 * jnp.pi * (1 + 2 * x[0])
    # Return the Swiss-roll shape. Note the 21 in the denominator
    # to make the scale more similar (but not identical) to the original (0, 1)
    return jnp.stack([t * jnp.cos(t), t * jnp.sin(t)]) / 21.0
