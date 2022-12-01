"""Utilities for creating dispersion matrices."""
import numpy as np


def one_and_one_correlation_matrix(dim_x: int, dim_y: int, correlation: float = 0.5) -> np.ndarray:
    """Creates a correlation (which can act as a covariance) matrix where the first dimension of X
    is correlated with the first dimension of Y.

    Returns:
        correlation matrix, shape (dim_x + dim_y, dim_x + dim_y)
    """
    if not -1 < correlation < 1:
        raise ValueError(f"Correlation {correlation} must be in the interval (-1, 1).")

    corr_matrix = np.eye(dim_x + dim_y)
    corr_matrix[0, dim_x] = correlation
    corr_matrix[dim_x, 0] = correlation

    return corr_matrix
