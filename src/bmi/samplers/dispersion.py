"""Utilities for creating dispersion matrices."""
import numpy as np


def _validate(
    dim_x: int, dim_y: int, k: int, correlation: float, correlation_x: float, correlation_y: float
) -> None:
    if min(dim_x, dim_y, k) < 1:
        raise ValueError(f"dim_x, dim_y and k must be at least 1. Were {dim_x}, {dim_y}, {k}.")

    if k > dim_x or k > dim_y:
        raise ValueError(f"dim_x={dim_x} and dim_y={dim_y} must be greater or equal than k={k}.")

    if (
        min(correlation_x, correlation_y, correlation) < -1
        or max(correlation_x, correlation_y, correlation) > 1
    ):
        raise ValueError("Correlations must be between -1 and 1.")


def parametrised_correlation_matrix(
    dim_x: int,
    dim_y: int,
    k: int = 1,
    correlation: float = 0.5,
    correlation_x: float = 0.0,
    correlation_y: float = 0.0,
) -> np.ndarray:
    """

    Args:
        dim_x: dimension of X variable
        dim_y: dimension of Y variable
        k: number of dimensions of X which should be correlated with a number of dimensions of Y
        correlation: correlation between X1 and Y1, X2 and Y2, ..., Xk and Yk
        correlation_x: correlation between Xi and Xj
        correlation_y: corrrelation between Yi and Yj

    Returns:
        correlation matrix, shape (dim_x + dim_y, dim_x + dim_y)

    Raises:
        ValueError, if `k` is greater than `dim_x` or `dim_y`
    """
    _validate(
        dim_x=dim_x,
        dim_y=dim_y,
        k=k,
        correlation=correlation,
        correlation_x=correlation_x,
        correlation_y=correlation_y,
    )

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
