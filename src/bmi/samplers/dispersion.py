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

def correlation_matrix(dim_x: int, dim_y: int, k: int = 1, correlation: float = 0.5, correlation_x: float = 0.0, correlation_y: float = 0.0) -> np.ndarray:
    """

    Args:

    Returns:
        correlation matrix, shape (dim_x + dim_y, dim_x + dim_y)

    Raises:
        ValueError, if `k` is greater than `dim_x` or `dim_y`
    """
    if min(dim_x, dim_y, k) < 1:
        raise ValueError(f"dim_x, dim_y and k must be at least 1. Were {dim_x}, {dim_y}, {k}.")

    if k > dim_x or k > dim_y:
        raise ValueError(f"dim_x={dim_x} and dim_y={dim_y} must be greater or equal than k={k}.")

    if min(correlation_x, correlation_y, correlation) < -1 or max(correlation_x, correlation_y, correlation) > 1:
        raise ValueError("Correlations must be between -1 and 1.")

    corr_matrix = np.eye(dim_x + dim_y)
    for i in range(k):
        corr_matrix[i, dim_x + i] = correlation
        corr_matrix[dim_x + i, i] = correlation

    for i in range(dim_x):
        for j in range(i):
            corr_matrix[i, j] = correlation_x
            corr_matrix[j, i] = correlation_x

    for i in range(dim_x, dim_x + dim_y):
        for j in range(dim_x, i):
            corr_matrix[i, j] = correlation_y
            corr_matrix[j, i] = correlation_y

    return corr_matrix
