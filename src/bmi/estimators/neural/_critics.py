"""Module with neural networks used as critics."""

from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp

from bmi.estimators.neural._types import Point


class MLP(eqx.Module):
    """Multi-layer perceptron with ReLU layers."""

    layers: list

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

        key_final, key_hidden = jax.random.split(key)
        keys_hidden = jax.random.split(key_hidden, len(hidden_layers))

        dims = [dim_x + dim_y] + list(hidden_layers)

        self.layers = []
        for dim_in, dim_out, key in zip(dims, dims[1:], keys_hidden):
            self.layers.append(eqx.nn.Linear(dim_in, dim_out, key=key))

        self.layers.append(eqx.nn.Linear(dims[-1], 1, key=key_final))

    def __call__(self, x: Point, y: Point) -> jax.Array:
        z = jnp.concatenate([x, y])

        for layer in self.layers[:-1]:
            z = layer(z)
            z = jax.nn.relu(z)

        # last layer without activation
        z = self.layers[-1](z)

        return z[..., 0]  # return scalar
