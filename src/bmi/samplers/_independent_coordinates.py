from typing import Sequence, Union

import jax.numpy as jnp
from jax import random

from bmi.interface import ISampler, KeyArray
from bmi.samplers.base import BaseSampler, cast_to_rng


class IndependentConcatenationSampler(BaseSampler):
    """Consider a sequence of samplers $S_k$, where $k \\in \\{1, \\dotsc, m \\}$
    and variables

    $$(X_k, Y_k) \\sim S_k.$$

    If the variables are sampled independently, we can concatenate them
    to $X = (X_1, \\dotsc, X_m)$ and $Y = (Y_1, \\dotsc, Y_m)$

    and have

    $$I(X; Y) = I(X_1; Y_1) + \\dotsc + I(X_m; Y_m).$$
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
        return float(
            jnp.sum(jnp.asarray([sampler.mutual_information() for sampler in self._samplers]))
        )
