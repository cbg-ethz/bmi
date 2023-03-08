"""These are neural estimator losses implemented so that the memory
is O(batch size) rather than O(batch size ** 2).

In particular, we can use larger batches.
"""
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from bmi.estimators.neural._interfaces import BatchedPoints, Critic, Point


def logmeanexp_nodiag(f: Critic, xs: BatchedPoints, ys: BatchedPoints, offset: float = 0) -> float:
    """This calculates log(mean(exp( A[i, j] ))),
    where :math:`A[i, j] = f(x[i], y[j]) + offset`

    Args:
        f: function to be applied to get the matrix :math:`A[i, j] = f(x[i], y[j]) + offset`
        xs: Xs in the batch, shape (batch_size, dim_x)
        ys: Ys paired with Xs, shape (batch_size, dim_y)
        offset: offset to be applied

    Returns:
        log-mean-exp of the off-diagonal terms of A[i, j]
    """
    # Can be applied to (x, ys) for each x in xs
    f_vmap = jax.vmap(f, in_axes=(None, 0))

    def body_fun(i: int, x: Point) -> tuple[int, float]:
        # Calculate [f(x, y1), ... f(x, yn)]
        fs: jnp.ndarray = f_vmap(x, ys) + offset

        # Now we don't want to sum the diagonal element
        fs = fs.at[i].set(-jnp.inf)

        return i + 1, logsumexp(fs)

    # We have now logsumexp calculated along axis
    size, vals = jax.lax.scan(body_fun, 0, xs)

    # Now we will logsumexp it to have the total logsumexp...
    lse = logsumexp(vals)

    # ... and finally we subtract the log-denominator to have the mean
    n_elements = size * (size - 1)

    return lse - jnp.log(n_elements)


def infonce(f: Critic, xs: BatchedPoints, ys: BatchedPoints) -> float:
    # Can be applied to (x, ys) for each x in xs
    f_vmap = jax.vmap(f, in_axes=(None, 0))

    def body_fun(_, x):
        fs = f_vmap(x, ys)
        return None, logsumexp(fs)

    _, vals = jax.lax.scan(body_fun, None, xs)

    # This is mean of logsumexp over different x
    mean_negative = jnp.mean(vals)

    # This is the mean score for positive pairs
    mean_positive = jnp.mean(jax.vmap(f, in_axes=(0, 0))(xs, ys))
    nll = mean_positive - mean_negative
    return nll + jnp.log(len(xs))


def donsker_varadhan(f: Critic, xs: BatchedPoints, ys: BatchedPoints) -> float:
    f_vmap = jax.vmap(f, in_axes=(0, 0))
    return jnp.mean(f_vmap(xs, ys)) - logmeanexp_nodiag(f, xs, ys)


def nwj(f, xs, ys) -> float:
    f_vmap = jax.vmap(f, in_axes=(0, 0))

    positive = jnp.mean(f_vmap(xs, ys))
    negative = jnp.exp(logmeanexp_nodiag(f, xs, ys, offset=-1.0))
    return positive - negative
