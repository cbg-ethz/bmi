"""Implementations of different mutual information losses
as in https://github.com/ermongroup/smile-mi-estimator.

Note:
    They require quadratic, that is O(batch size ** 2), memory
    so they cannot be used with large batches.
"""
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from bmi.estimators.neural._types import BatchedPoints, Critic


def get_full_scores(f: Critic, xs: BatchedPoints, ys: BatchedPoints) -> jnp.ndarray:
    """Calculates the matrix f(x, y). Note that it's quadratic in terms of memory."""
    return jax.vmap(jax.vmap(f, in_axes=(None, 0)), in_axes=(0, None))(xs, ys)


def logmeanexp_nodiag(array: jnp.ndarray) -> float:
    """This calculates log(mean(exp(offdiagonal terms of `array`))).

    Args:
        array: shape (n, n)

    Returns:
        log-mean-exp of the off-diagonal terms

    Note:
        It's logsumexp with normalization log(n*(n-1)) subtracted
    """
    size = len(array)

    neg_inf_vector = jnp.full(shape=array.diagonal().shape, fill_value=-jnp.inf)
    sum_offdiagonal = logsumexp(
        array + jnp.diag(neg_inf_vector),
        axis=(0, 1),
    )
    n_elements = size * (size - 1)

    return sum_offdiagonal - jnp.log(n_elements)


def infonce(f: Critic, xs: BatchedPoints, ys: BatchedPoints) -> float:
    full_scores = get_full_scores(f, xs, ys)

    nll = full_scores.diagonal().mean() - logsumexp(full_scores, axis=1)
    mi = jnp.log(len(full_scores)) + nll

    return jnp.mean(mi)


def donsker_varadhan(f: Critic, xs: BatchedPoints, ys: BatchedPoints) -> float:
    full_scores = get_full_scores(f, xs, ys)
    return full_scores.diagonal().mean() - logmeanexp_nodiag(full_scores)


def nwj(f: Critic, xs: BatchedPoints, ys: BatchedPoints) -> float:
    full_scores = get_full_scores(f, xs, ys)
    full_scores_m1 = full_scores - 1.0

    return 1.0 + full_scores_m1.diagonal().mean() - jnp.exp(logmeanexp_nodiag(full_scores_m1))
