"""Basic training loop used for most neural estimators."""
from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import tqdm

from bmi.estimators.neural._types import BatchedPoints, Critic, Point


class TrainingLog:
    def __init__(
        self,
        max_n_steps: int,
        train_smooth_factor: float = 0.1,
        verbose: bool = True,
        enable_tqdm: bool = True,
    ):
        self.max_n_steps = max_n_steps
        self.train_smooth_window = int(max_n_steps * train_smooth_factor)
        self.verbose = verbose

        self._mi_train_history = []
        self._mi_test_history = []
        self._mi_test_best = None
        self._logs_since_mi_test_best = 0
        self._tqdm = None

        if verbose and enable_tqdm:
            self._tqdm_init()

    def log_train_mi(self, n_step: int, mi: float):
        self._mi_train_history.append((n_step, mi))
        self._tqdm_update()

    def log_test_mi(self, n_step: int, mi: float):
        if self._mi_test_best is None or self._mi_test_best < mi:
            self._mi_test_best = mi
            self._logs_since_mi_test_best = 0
        else:
            self._logs_since_mi_test_best += 1

        self._mi_test_history.append((n_step, mi))

        if self.verbose and self._tqdm is None:
            print(f"MI test: {mi:.2f} (step={n_step})")

        self._tqdm_refresh()

    @property
    def final_mi(self):
        return self._mi_test_best

    def early_stop(self) -> bool:
        return self._logs_since_mi_test_best > 1

    def finish(self):
        self._tqdm_close()

        if self.verbose:
            self.print_warnings()

    def print_warnings(self):
        # analyze training
        train_mi = jnp.array([mi for _step, mi in self._mi_train_history])
        w = self.train_smooth_window
        cs = jnp.cumsum(train_mi)
        train_mi_smooth = (cs[w:] - cs[:-w]) / w

        if len(train_mi_smooth) > 0:
            train_mi_smooth_max = float(train_mi_smooth.max())
            train_mi_smooth_fin = float(train_mi_smooth[-1])
            if train_mi_smooth_max > 1.05 * train_mi_smooth_fin:
                print(
                    f"WARNING: Smoothed training MI fell compared to highest value: "
                    f"max={train_mi_smooth_max:.3f} vs "
                    f"final={train_mi_smooth_fin:.3f}"
                )

        w = self.train_smooth_window
        if len(train_mi_smooth) >= w:
            train_mi_smooth_fin = float(train_mi_smooth[-1])
            train_mi_smooth_prv = float(train_mi_smooth[-w])
            if train_mi_smooth_fin > 1.05 * train_mi_smooth_prv:
                print(
                    f"WARNING: Smoothed raining MI was still increasing when training stopped: "
                    f"final={train_mi_smooth_fin:.3f} vs "
                    f"{w} step(s) ago={train_mi_smooth_prv:.3f}"
                )

    def _tqdm_init(self):
        self._tqdm = tqdm.tqdm(
            total=self.max_n_steps,
            unit="step",
            ncols=120,
        )

    def _tqdm_update_prefix(self):
        if self._tqdm is None:
            return

        if self._mi_train_history:
            train_str = f"{self._mi_train_history[-1][-1]:.2f}"
        else:
            train_str = "???"

        if self._mi_test_history:
            test_str = f"{self._mi_test_history[-1][-1]:.2f}"
        else:
            test_str = "???"

        self._tqdm.set_postfix(train=train_str, test=test_str)

    def _tqdm_update(self):
        if self._tqdm is not None:
            self._tqdm_update_prefix()
            self._tqdm.update()

    def _tqdm_refresh(self):
        if self._tqdm is not None:
            self._tqdm_update_prefix()
            self._tqdm.refresh()

    def _tqdm_close(self):
        if self._tqdm is None:
            return

        self._tqdm.close()
        self._tqdm = None


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

    # compile training step
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
    training_log = TrainingLog(max_n_steps=max_n_steps)
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
        if early_stopping and training_log.early_stop():
            break

    training_log.finish()

    if verbose:
        if early_stopping and not training_log.early_stop():
            print("WARNING: Early stopping enabled but max_n_steps reached.")

    return training_log
