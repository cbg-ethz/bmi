from typing import Sequence, Union

import jax.numpy as jnp
from jax import random

from bmi.interface import ISampler, KeyArray
from bmi.samplers.base import BaseSampler, cast_to_rng


class IndependentConcatenationSampler(BaseSampler):
    """Consider a sequence of samplers Sk, k = 1, ..., m
    and variables
    (Xk, Yk) ~ Sk

    If the variables are sampled independently, we can concatenate them
    to X = (X1, ..., Xm) and Y = (Y1, ..., Ym)

    and have

    I(X; Y) = I(X1; Y1) + ... + I(Xm; Ym)
    """

    def __init__(self, samplers: Sequence[ISampler]) -> None:
        """
        Args:
            samplers: sequence of samplers to concatenate
        """
        self._samplers = list(samplers)
        dim_x = sum(sampler.dim_x for sampler in self._samplers)
        dim_y = sum(sampler.dim_y for sampler in self._samplers)

        super().__init__(dim_x=dim_x, dim_y=dim_y)

    def sample(self, n_points: int, rng: Union[int, KeyArray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        rng = cast_to_rng(rng)
        keys = random.split(rng, len(self._samplers))
        xs = []
        ys = []
        for key, sampler in zip(keys, self._samplers):
            x, y = sampler.sample(n_points, key)
            xs.append(x)
            ys.append(y)

        return jnp.hstack(xs), jnp.hstack(ys)

    def mutual_information(self) -> float:
        return jnp.sum([sampler.mutual_information() for sampler in self._samplers])
