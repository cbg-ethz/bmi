from typing import Union

import jax.numpy as jnp
import jax.random
import numpy as np

from bmi.interface import KeyArray
from bmi.samplers.base import BaseSampler, cast_to_rng


class AdditiveUniformSampler(BaseSampler):
    def __init__(self, epsilon: float) -> None:
        """Represents the distribution $P_{XY}$ under the following model:

        $$X \\sim \\mathrm{Uniform}(0, 1)$$

        $$N \\sim \\mathrm{Uniform}(-\\epsilon, \\epsilon)$$

        $$Y = X + N$$

        The MI in this case is:

        $$I(X; Y) = \\begin{cases}
             \\frac{1}{4 \\epsilon} \\mathrm{~if~} \\epsilon > 0.5    \\\\
             \\epsilon - \\log(2 \\epsilon) \\mathrm{~otherwise}
            \\end{cases}$$

        and can be derived analytically.
        """
        super().__init__(dim_x=1, dim_y=1)

        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, was {epsilon}.")
        self._epsilon = epsilon

    def sample(self, n_points: int, rng: Union[int, KeyArray]) -> tuple[np.ndarray, np.ndarray]:
        rng = cast_to_rng(rng)
        key_x, key_n = jax.random.split(rng)
        x = jax.random.uniform(key_x, shape=(n_points, 1), minval=0.0, maxval=1.0)
        n = jax.random.uniform(
            key_n, shape=(n_points, 1), minval=-self._epsilon, maxval=self._epsilon
        )
        y = x + n
        return x, y

    @staticmethod
    def mutual_information_function(epsilon: float) -> float:
        if epsilon > 0.5:
            return 0.25 / epsilon
        else:
            return epsilon - jnp.log(2 * epsilon)

    def mutual_information(self) -> float:
        return self.mutual_information_function(self._epsilon)
