"""Module with some neural network critics implemented
as well as basic training loop."""
import dataclasses
from typing import Callable, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from bmi.estimators.neural._interfaces import BatchedPoints, Critic, Point


class MLP(eqx.Module):
    """Multi-layer perceptron with ReLU layers."""

    layers: list
    extra_bias: jax.numpy.ndarray

    def __init__(
        self,
        key: jax.random.PRNGKeyArray,
        dim_x: int,
        dim_y: int,
        hidden_layers: Sequence[int] = (5,),
    ) -> None:
        """

        Args:
            key: JAX random key to initialize the network
            dim_x: dimension of the X space
            dim_y: dimension of the Y space
            hidden_layers: dimensionalities of hidden layers

        Example:
            Let ``hidden_layers = (5,)``
            There will be two layers in this network:
              - dim_x + dim_y -> 5
              - 5 -> 1

            Let ``hidden_layers = (8, 12)``
            There will be three layers in this neural network:
              - dim_x + dim_y 0 -> 8
              - 8 -> 12
              - 12 -> 1
        """
        # We have in total the following dimensionalities:
        dim_sizes = [dim_x + dim_y] + list(hidden_layers) + [1]
        # ... and one layer less:
        keys = jax.random.split(key, len(hidden_layers) + 1)

        self.layers = []

        for i, key in enumerate(keys):
            self.layers.append(eqx.nn.Linear(dim_sizes[i], dim_sizes[i + 1], key=key))

        # This is ann additional trainable parameter.
        self.extra_bias = jax.numpy.ones(1)

    def __call__(self, x: Point, y: Point) -> float:
        z = jnp.concatenate([x, y])

        for layer in self.layers[:-1]:
            z = jax.nn.relu(layer(z))
        return jnp.mean(self.layers[-1](z) + self.extra_bias)


@dataclasses.dataclass
class TrainHistory:
    loss_history: list[float]
    test_history: Optional[list[float]]
    final_mi: float


def mi_divergence_check(xs: jnp.ndarray) -> Optional[tuple[float, float]]:
    assert xs.ndim == 1
    xs_argmax = jnp.argmax(xs)
    xs_max = xs[xs_argmax]
    xs_after_max = xs[xs_argmax:]
    xs_min_after_max = xs_after_max.min()
    if xs_max > 0.01 and xs_min_after_max < 0.9 * xs_max:
        return float(xs_max), float(xs_min_after_max)

<<<<<<< HEAD
=======

>>>>>>> d6b6aef (add train/test split and some convergence diagnostics)
def basic_fit(
    rng: jax.random.PRNGKeyArray,
    critic: eqx.Module,
    mi_formula: Callable[[Critic, Point, Point], float],
    xs: BatchedPoints,
    ys: BatchedPoints,
    mi_formula_test: Optional[Callable[[Critic, Point, Point], float]] = None,
    xs_test: Optional[BatchedPoints] = None,
    ys_test: Optional[BatchedPoints] = None,
    test_every_n_steps: int = 250,
    batch_size: int = 256,
    max_n_steps: int = 2_000,
    learning_rate: float = 0.1,
    verbose: bool = False,
) -> TrainHistory:
    """Simplest training loop, which samples mini-batches
    from (xs, ys) and maximizes mutual information according to
    ``mi_formula`` using trainable ``critic``.
    """
    mi_formula_test = mi_formula_test or mi_formula
    testing = xs_test is not None and ys_test is not None and test_every_n_steps is not None

    def loss(f: Critic, xs: BatchedPoints, ys: BatchedPoints) -> float:
        """We maximize mutual information by *minimizing* loss."""
        return -mi_formula(f, xs, ys)

    # Initialize the optimized
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(critic)

    loss_history = []
    test_history = []

    @jax.jit
    def step(params, opt_state, xs: BatchedPoints, ys: BatchedPoints):
        """One training step."""
        loss_value, grads = jax.value_and_grad(loss)(params, xs, ys)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    keys = jax.random.split(rng, max_n_steps)

    for epoch, key in enumerate(keys):
        batch_indices = jax.random.choice(
            key,
            len(xs),
            shape=(batch_size,),
            replace=False,
        )
        batch_xs = xs[batch_indices, ...]
        batch_ys = ys[batch_indices, ...]

        critic, opt_state, loss_value = step(critic, opt_state, batch_xs, batch_ys)
        if verbose and epoch % 500 == 0:
            print(f"Epoch {epoch}, train MI: {-loss_value:.2f}")

        if testing and epoch % test_every_n_steps == 0:
            loss_test = -mi_formula_test(critic, xs_test, ys_test)
            if verbose:
                print(f"Epoch {epoch}, test MI: {-loss_test:.2f}")
            test_history.append(loss_test)

        loss_history.append(loss_value)

    if testing:
        final_mi = -min(test_history)
    else:
        # if no testing is used, compute final mi using
        # testing mi formula on whole dataset
        final_mi = mi_formula_test(critic, xs, ys)

    return TrainHistory(
        loss_history=loss_history,
        test_history=test_history if testing else None,
        final_mi=final_mi,
    )
