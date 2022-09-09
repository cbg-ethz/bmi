"""Partial implementation of the IDistribution interface, convenient to inherit from."""
from bmi.interface import ISampler


class BaseSampler(ISampler):
    """Partial implementation of the IDistribution interface, convenient to inherit from."""

    @staticmethod
    def _validate_dimensions(dim_x: int, dim_y: int) -> None:
        """Method used to validate dimensions."""
        if dim_x < 1:
            raise ValueError(f"X variable space must be of positive dimension. Was {dim_x}.")
        if dim_y < 1:
            raise ValueError(f"Y variable space must be of positive dimension. Was {dim_x}.")
