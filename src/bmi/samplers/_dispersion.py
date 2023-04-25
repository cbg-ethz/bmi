"""Utilities for creating dispersion matrices."""
import dataclasses
from typing import Optional

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


@dataclasses.dataclass
class GaussianLVMParametrization:
    """Parameters of a Gaussian latent variable models.

    Attrs:
        dim_x: the dimensionality of the X variable
        dim_y: the dimensionality of the Y variable
        n_interacting: number of strongly interacting components of X vs Y
          (strength controlled by parameter `lambd`)
        alpha: dense interactions (between every pair of variables)
        beta_x: dense interactions between X components
        beta_y: dense interactions between Y components
        lambd: strength of strongly interacting components of X vs Y
        epsilon_x: individual noise, increasing variances of X components
        epsilon_y: individual noise, increasing variances of Y components
        eta_x: increases variance of X components which are not strongly interacting
          via `n_interacting` and `lambd`
        eta_y: increases variance of Y components which are not strongly interacting
          via `n_interacting` and `lambd`
    """

    dim_x: int
    dim_y: int
    n_interacting: int = 0
    alpha: float = 0.0
    beta_x: float = 0.0
    beta_y: float = 0.0
    lambd: float = 0.0
    epsilon_x: float = 1.0
    epsilon_y: Optional[float] = None
    eta_x: float = 0.0
    eta_y: Optional[float] = None

    def mixing(self) -> np.ndarray:
        """"""
        pass

    def covariance(self) -> np.ndarray:
        pass

    def correlation(self) -> np.ndarray:
        pass

    def latent_variable_labels(self) -> list[str]:
        pass

    def xy_labels(self) -> list[str]:
        pass


class DenseLVMParametrization(GaussianLVMParametrization):
    """Dense interaction scheme: between every two (distinct) variables
    the covariance is the same.
    """

    def __init__(self, dim_x: int, dim_y: int, alpha: float, epsilon: float) -> None:
        super().__init__(
            dim_x=dim_x,
            dim_y=dim_y,
            n_interacting=0,
            alpha=alpha,
            beta_x=0.0,
            beta_y=0.0,
            lambd=0.0,
            epsilon_x=epsilon,
            epsilon_y=epsilon,
            eta_x=0.0,
            eta_y=0.0,
        )

    def correlation_strength(self) -> float:
        """Apart from correlations Cor(X_i, X_i)=Cor(Y_i, Y_i)=1
        all the correlations are the same and given by this value.
        """
        return self.alpha**2 / (self.alpha**2 + self.epsilon_x**2)


class SparseLVMParametrization(GaussianLVMParametrization):
    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        n_interacting: int,
        beta: float = 0.0,
        lambd: float = 1.0,
        epsilon: float = 1.0,
        eta: Optional[float] = None,
    ) -> None:
        """

        Args:
            eta: by default `lambd` will be used, so that
              all variables have the same variance
        """
        eta = lambd if eta is None else eta

        super().__init__(
            dim_x=dim_x,
            dim_y=dim_y,
            n_interacting=n_interacting,
            alpha=0.0,
            beta_x=beta,
            beta_y=beta,
            lambd=lambd,
            epsilon_x=epsilon,
            epsilon_y=epsilon,
            eta_x=eta,
            eta_y=eta,
        )

    def xx_correlation(self) -> float:
        """Correlation between X_i and X_j for i *different from* j
        (for i=j the correlation is obviously 1).
        """
        pass

    def yy_correlation(self) -> float:
        """Correlation between X_i and X_j for i *different from* j
        (for i=j the correlation is obviously 1).
        """
        pass

    def xy_interacting_correlation(self) -> float:
        """Correlation between X_i and Y_i for i < `n_interacting`.
        (All the other correlations X_i and Y_j are 0).
        """
        pass
