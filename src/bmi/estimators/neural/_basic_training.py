"""Basic training loop used for most neural estimators."""
from typing import Callable, Optional

import equinox as eqx
import jax
import optax

from bmi.estimators.neural._training_log import TrainingLog
from bmi.estimators.neural._types import BatchedPoints, Critic, Point


def get_batch(
    xs: BatchedPoints, ys: BatchedPoints, key: jax.random.PRNGKeyArray, batch_size: Optional[int]
):
    if batch_size is not None:
        batch_indices = jax.random.choice(
            key,
            len(xs),
            shape=(batch_size,),
            replace=False,
        )
        return xs[batch_indices], ys[batch_indices]
    else:
        return xs, ys


def basic_training(
    rng: jax.random.PRNGKeyArray,
    critic: eqx.Module,
    mi_formula: Callable[[Critic, Point, Point], float],
    xs: BatchedPoints,
    ys: BatchedPoints,
    mi_formula_test: Optional[Callable[[Critic, Point, Point], float]] = None,
    xs_test: Optional[BatchedPoints] = None,
    ys_test: Optional[BatchedPoints] = None,
    batch_size: Optional[int] = 256,
    test_every_n_steps: int = 250,
    max_n_steps: int = 2_000,
    early_stopping: bool = True,
    learning_rate: float = 0.1,
    verbose: bool = True,
) -> TrainingLog:
    """Simple training loop, which samples mini-batches
    from (xs, ys) and maximizes mutual information according to
    ``mi_formula`` using trainable ``critic``.
    """
    mi_formula_test = mi_formula_test or mi_formula
    xs_test = xs_test if xs_test is not None else xs
    ys_test = ys_test if ys_test is not None else ys

    # initialize the optimizer
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(critic)

    # compile the training step
    @jax.jit
    def step(critic, opt_state, xs: BatchedPoints, ys: BatchedPoints):
        """One training step."""

        def loss(critic, xs, ys):
            return -mi_formula(critic, xs, ys)

        loss_step, grads = jax.value_and_grad(loss)(critic, xs, ys)
        updates, opt_state = optimizer.update(grads, opt_state, critic)
        critic = optax.apply_updates(critic, updates)
        return critic, opt_state, -loss_step

    # main training loop
    training_log = TrainingLog(
        max_n_steps=max_n_steps, early_stopping=early_stopping, verbose=verbose
    )
    keys = jax.random.split(rng, max_n_steps)
    for n_step, key in enumerate(keys, start=1):
        # run step
        batch_xs, batch_ys = get_batch(xs, ys, key, batch_size)
        critic, opt_state, mi_train = step(critic, opt_state, batch_xs, batch_ys)

        # logging train
        training_log.log_train_mi(n_step, mi_train)

        # logging test
        if n_step % test_every_n_steps == 0:
            mi_test = mi_formula_test(critic, xs_test, ys_test)
            training_log.log_test_mi(n_step, mi_test)

        # early stop?
        if training_log.early_stop():
            break

    training_log.finish()

    return training_log
