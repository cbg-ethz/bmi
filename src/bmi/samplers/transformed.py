from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

import bmi.samplers.base as base
from bmi.interface import ISampler, KeyArray

SomeArray = Union[jnp.ndarray, np.ndarray]
Transform = Callable[[SomeArray], jnp.ndarray]


def identity(x: SomeArray) -> SomeArray:
    return x


class TransformedSampler(base.BaseSampler):
    def __init__(
        self,
        base_sampler: ISampler,
        transform_x: Optional[Callable] = None,
        transform_y: Optional[Callable] = None,
        add_dim_x: int = 0,
        add_dim_y: int = 0,
    ) -> None:
        super().__init__(
            dim_x=base_sampler.dim_x + add_dim_x, dim_y=base_sampler.dim_y + add_dim_y
        )

        if transform_x is None:
            transform_x = identity
        if transform_y is None:
            transform_y = identity

        self._vectorized_transform_x = jax.vmap(transform_x)
        self._vectorized_transform_y = jax.vmap(transform_y)
        self._base_sampler = base_sampler

    def transform(self, x: SomeArray, y: SomeArray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self._vectorized_transform_x(x), self._vectorized_transform_y(y)

    def sample(self, n_points: int, rng: Union[int, KeyArray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        x, y = self._base_sampler.sample(n_points=n_points, rng=rng)
        return self.transform(x, y)

    def mutual_information(self) -> float:
        return self._base_sampler.mutual_information()
