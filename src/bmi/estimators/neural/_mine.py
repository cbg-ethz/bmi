"""Implementation of MINE:

M.I. Belghazi et al.,
MINE: Mutual Information Neural Estimation
https://arxiv.org/abs/1801.04062v5

The expression for the gradient
is given by Equation (12) in Section 3.2.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax.scipy.special import logsumexp
from jax.tree_util import tree_map

from bmi.estimators.neural._interfaces import BatchedPoints, Critic


def mine_value(
    f: Critic, xs: BatchedPoints, ys_paired: BatchedPoints, ys_unpaired: BatchedPoints
) -> float:
    """Calculates the MINE estimate on a batch.

    Args:
        f: critic
        xs: realizations X variable, shape (batch_size, dim_x)
        ys_paired: realizations of Y variable matched with X. In other words,
            zip(xs, ys_paired) is a sample from P(X, Y). Shape (batch_size, dim_y)
        ys_unpaired: samples from P(Y), shape (batch_size, dim_y)
            In other words, zip(xs, ys_unpaired) should be a sample from P(X) P(Y).
    """
    f_vmap = jax.vmap(f, in_axes=(0, 0))

    positive = jnp.mean(f_vmap(xs, ys_paired))
    negative = logsumexp(f_vmap(xs, ys_unpaired))

    return positive - negative + jnp.log(len(xs))


def _gradient_first_term(
    *, f: eqx.Module, xs: BatchedPoints, ys_paired: BatchedPoints
) -> eqx.Module:
    """Calculates the first term of Eq. (12).

    This is the gradient of f calculated with respect to its parameters
    and averaged over the batch of paired data.

    Note that it is the gradient of the first term of Eq. (10),
    so that we can first sum up values and then differentiate.
    """

    def auxiliary(t: eqx.Module, xs: BatchedPoints, ys: BatchedPoints) -> float:
        t_vmap = jax.vmap(t, in_axes=(0, 0))
        ts = t_vmap(xs, ys)
        return jnp.mean(ts)

    return jax.grad(auxiliary)(f, xs, ys_paired)


def _gradient_second_term(
    *, f: eqx.Module, xs: BatchedPoints, ys_unpaired: BatchedPoints
) -> eqx.Module:
    """This is the gradient:

    d log E[ exp f_theta(x, y) ]
    ------------------------------
             d theta

    where the expectation is taken over
    pX x pY, that is unpaired data.

    Compare with the second term of Eq. (10).
    """
    # We will use the fact that logsumexp
    # and logmeanexp differ by a constant
    # which vanishes after differentiation

    def auxiliary(t, xs, ys):
        t_vmap = jax.vmap(t, in_axes=(0, 0))
        ts = t_vmap(xs, ys)
        return logsumexp(ts)

    return jax.grad(auxiliary)(f, xs, ys_unpaired)


def _exponential_smoothing(
    log_new: float,
    log_old: float,
    alpha: float,
) -> float:
    """Consider exponential smoothing:
      alpha * new + (1-alpha) * old
    for alpha in (0, 1).

    We work apply the exponential smoothing having access
    to the log-values.


    Args:
        log_new: log(new)
        log_old: log(old)
        alpha: number from the open interval (0, 1). For
          values close to 1 the smoothing is weak. For values
          close to 0 the smoothing is strong

    Returns:
        float, log(smoothed new)
    """
    return jnp.logaddexp(
        jnp.log(alpha) + log_new,
        jnp.log(1.0 - alpha) + log_old,
    )


def _correction(
    f: eqx.Module,
    xs: BatchedPoints,
    ys_unpaired: BatchedPoints,
    log_previous: float,
    alpha: float,
) -> tuple[float, float]:
    """
    The ``_gradient_second_term`` is biased.

    We will slightly rescale it by a factor

    A / B,

    where
    A = E[ exp f_theta(x, y) ]
    and
    B = exponential moving average
      = alpha * A + (1-alpha) * C,

    where C is B obtained at the previous step

    To improve numerical stability, we use
    log(A) and log(B) instead.

    Args:
        f: critic
        xs: points, shape (batch_size, dim_x)
        ys_unpaired: unmatched points, shape (batch_size, dim_y)
        log_previous: log(C) which is previous log(B)

    Returns:
        A/B
        log(B), to be used at the next step
    """
    fs = jax.vmap(f, in_axes=(0, 0))(xs, ys_unpaired)
    # A = log( E[exp f(x,y) )
    log_a = logsumexp(fs) - jnp.log(len(fs))

    # B = smoothed version of A
    log_b = _exponential_smoothing(
        log_new=log_a,
        log_old=log_previous,
        alpha=alpha,
    )

    # Calculate A/B
    a_div_b = jnp.exp(log_a - log_b)

    # Return A/B and log(B)
    return a_div_b, log_b


def mine_negative_grads_and_carry(
    f: eqx.Module,
    xs: BatchedPoints,
    ys_paired: BatchedPoints,
    ys_unpaired: BatchedPoints,
    log_previous: float,
    alpha: float,
) -> tuple[eqx.Module, float]:
    """Calculates the *negative* gradient of MINE objective
    according to Eq. (12) of the paper.

    Args:
        f: Equinox module representing the critic
        xs: shape (batch_size, dim_x)
        ys_paired: matched samples of shape (batch_size, dim_y)
        ys_unpaired: *randomly resampled* samples of shape (batch_size, dim_y)
        log_previous: the log(denominator) used at the previous step,
          used to debias the gradient (see Section 3.2 of the paper)
          via exponential smoothing
        alpha: float between (0, 1). Values close to 1 result in little
          smoothing. Values close to 0 result in strong smoothing

    Returns:
        (negative) gradient to be applied to f parameters, PyTree
          of the same structure as f
        updated denominator, to be used to adjust the gradient
          at the next training step

    Note:
        MINE objective is to be *maximized* which can be
        accomplished using gradient ascent.

        However, most of the optimizers works with minimization,
        so that we return *negative gradients*.
        Gradient descent using negative gradients is equivalent
        to the gradient ascent using normal ones.
    """
    # TODO(Frederic, Pawel): Consider computing the second term directly.
    #   with exp(fs) weights divided by the denominator for numerical stability.
    first_term = _gradient_first_term(f=f, xs=xs, ys_paired=ys_paired)
    second_term = _gradient_second_term(f=f, xs=xs, ys_unpaired=ys_unpaired)

    correction, log_carried = _correction(
        f=f, xs=xs, ys_unpaired=ys_unpaired, log_previous=log_previous, alpha=alpha
    )

    def negative_grad_func(first: jnp.ndarray, second: jnp.ndarray) -> jnp.ndarray:
        """Auxiliary function to be applied to the leaves in PyTree
        representing the critic ``f``."""
        grad = first - correction * second
        return -grad

    negative_grad = tree_map(negative_grad_func, first_term, second_term)

    return negative_grad, log_carried


def _sample_paired(
    key: jax.random.PRNGKeyArray, xs: BatchedPoints, ys: BatchedPoints, batch_size: int
) -> tuple[BatchedPoints, BatchedPoints]:
    assert len(xs) == len(ys), f"Length mismatch: {len(xs)} != {len(ys)}"

    paired_indices = jax.random.choice(
        key,
        len(xs),
        shape=(batch_size,),
        replace=False,
    )
    xs_paired = xs[paired_indices, ...]
    ys_paired = ys[paired_indices, ...]

    return xs_paired, ys_paired


def _sample_unpaired(
    key: jax.random.PRNGKeyArray,
    ys: BatchedPoints,
    batch_size: int,
) -> BatchedPoints:
    # TODO(Pawel, Frederic): Consider using roll(1)
    #  to shuffle (and shuffle only once, at the beginning).
    #  (Currently we can accidentally get a positive pair, but
    #  it does not look like a serious issue.)
    indices = jax.random.choice(
        key,
        len(ys),
        shape=(batch_size,),
        replace=False,
    )
    return ys[indices]


def training_loop(
    rng: jax.random.PRNGKeyArray,
    critic: eqx.Module,
    xs: BatchedPoints,
    ys: BatchedPoints,
    max_n_steps: int,
    learning_rate: float = 0.1,
    batch_size: int = 256,
    alpha: float = 0.9,
):

    # Initialize the optimized
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(critic)

    @jax.jit
    def step_func(
        *,
        critic,
        opt_state,
        xs: BatchedPoints,
        ys_paired: BatchedPoints,
        ys_unpaired: BatchedPoints,
        log_carry: float,
        alpha: float,
    ):
        neg_grads, new_log_carry = mine_negative_grads_and_carry(
            f=critic,
            xs=xs,
            ys_paired=ys_paired,
            ys_unpaired=ys_unpaired,
            alpha=alpha,
            log_previous=log_carry,
        )
        updates, opt_state = optimizer.update(neg_grads, opt_state, critic)
        params = optax.apply_updates(critic, updates)
        return params, opt_state, new_log_carry

    # We initialize this value to be very small at the beginning,
    # so that the smoothing effect of it is negligible
    log_carry = -jnp.inf

    final_eval_key, *step_keys = jax.random.split(rng, max_n_steps + 1)

    for step, key in enumerate(step_keys, 1):
        key_paired, key_unpaired = jax.random.split(key)

        xs_paired, ys_paired = _sample_paired(key_paired, xs=xs, ys=ys, batch_size=batch_size)
        ys_unpaired = _sample_unpaired(key_unpaired, ys=ys, batch_size=batch_size)

        critic, opt_state, log_carry = step_func(
            critic=critic,
            opt_state=opt_state,
            xs=xs_paired,
            ys_paired=ys_paired,
            ys_unpaired=ys_unpaired,
            log_carry=log_carry,
            alpha=alpha,
        )

        ys_unpaired = jax.random.shuffle(final_eval_key, ys, axis=1)
        mine_val = mine_value(f=critic, xs=xs, ys_paired=ys, ys_unpaired=ys_unpaired)
        if step % 50 == 0:
            print(f"Step {step}:\t{mine_val:.2f}")

    ys_unpaired = jax.random.shuffle(final_eval_key, ys, axis=1)
    return mine_value(f=critic, xs=xs, ys_paired=ys, ys_unpaired=ys_unpaired)
