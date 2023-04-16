from jax.scipy.special import erf


def normal_cdf(x):
    return 0.5 * (1 + erf(x / 2**0.5))
