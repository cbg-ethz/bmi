"""Partial implementation of the ISampler interface, convenient to inherit from."""
from bmi.interface import ISampler


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
