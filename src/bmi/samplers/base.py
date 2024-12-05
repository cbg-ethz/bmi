"""Partial implementation of the ISampler interface, convenient to inherit from."""

from typing import Union

import numpy as np
from jax import random

from bmi.interface import ISampler, KeyArray


def _validate_dimensions(dim_x: int, dim_y: int) -> None:
    """Method used to validate dimensions."""
    if dim_x < 1:
        raise ValueError(f"X variable space must be of positive dimension. Was {dim_x}.")
    if dim_y < 1:
        raise ValueError(f"Y variable space must be of positive dimension. Was {dim_x}.")


class BaseSampler(ISampler):
    """Partial implementation of the ISampler interface, convenient to inherit from."""

    def __init__(self, *, dim_x: int, dim_y: int) -> None:
        _validate_dimensions(dim_x, dim_y)
        self._dim_x = dim_x
        self._dim_y = dim_y

    @property
    def dim_x(self) -> int:
        return self._dim_x

    @property
    def dim_y(self) -> int:
        return self._dim_y

    def mutual_information_std(self) -> float:
        return 0.0


def cast_to_rng(seed: Union[KeyArray, int]) -> KeyArray:
    """Casts `int` to a KeyArray."""
    if isinstance(seed, int) or isinstance(seed, np.integer):
        seed = int(seed)
        return random.PRNGKey(seed)
    else:
        return seed
