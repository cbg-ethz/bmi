"""Module with neural networks used as critics."""

from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp

from bmi.estimators.neural._types import Point


class MLP(eqx.Module):
    """Multi-layer perceptron with ReLU layers."""

    layers: list
    extra_bias: jax.numpy.ndarray

    def __init__(
        self,
        key: jax.Array,
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
