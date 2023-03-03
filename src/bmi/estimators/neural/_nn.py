"""Module with some neural network critics implemented
as well as basic training loop."""
import dataclasses
from typing import Callable, Sequence

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
    final_mi: float


def basic_fit(
    rng: jax.random.PRNGKeyArray,
    critic: eqx.Module,
    mi_formula: Callable[[Critic, Point, Point], float],
    xs: BatchedPoints,
    ys: BatchedPoints,
    batch_size: int = 256,
    max_n_steps: int = 2_000,
    learning_rate: float = 0.1,
) -> TrainHistory:
    """Simplest training loop, which samples mini-batches
    from (xs, ys) and maximizes mutual information according to
    ``mi_formula`` using trainable ``critic``.
    """

    def loss(f: Critic, xs: BatchedPoints, ys: BatchedPoints) -> float:
        """We maximize mutual information by *minimizing* loss."""
        return -mi_formula(f, xs, ys)

    # Initialize the optimized
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(critic)

    loss_history = []

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
            jnp.arange(len(xs)),
            shape=(batch_size,),
            replace=False,
        )
        batch_xs = xs[batch_indices, ...]
        batch_ys = ys[batch_indices, ...]

        critic, opt_state, loss_value = step(critic, opt_state, batch_xs, batch_ys)
        # TODO(Pawel, Frederic): Think about making this non-verbose.
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, MI: {-loss_value:.2f}")

        loss_history.append(loss_value)

    return TrainHistory(
        loss_history=loss_history,
        final_mi=-loss_history[-1],
    )
