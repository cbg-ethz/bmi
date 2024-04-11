"""Utilities for creating dispersion matrices."""

import dataclasses
from typing import List, Optional, Union

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


def canonical_correlation(
    rho: Union[np.ndarray, List[float]], additional_y: int = 0
) -> np.ndarray:
    """Constructs a covariance matrix given by canonical correlations.

    Namely,

        var(Xi) = var(Yj) = 1
        cov(Xi, Yi) = rho[i]

    and for i != j
        cov(Xi, Xj) = cov(Yi, Yj) = cov(Xi, Yj) = 0

    Args:
        rho: canonical correlations, shape (dim_x,)
        additional_y: controls the dimension of y,
          namely `dim_y = dim_x + additional_y`

    Returns:
        covariance matrix, shape (dim_x + dim_y, dim_x + dim_y)
    """
    dim_x = len(rho)
    dim_y = dim_x + additional_y

    covariance = np.zeros((dim_x + dim_y, dim_x + dim_y), dtype=float)
    covariance[:dim_x, :dim_x] = np.eye(dim_x, dtype=float)
    covariance[dim_x:, dim_x:] = np.eye(dim_y, dtype=float)

    for i in range(dim_x):
        covariance[i, dim_x + i] = rho[i]
        covariance[dim_x + i, i] = rho[i]

    return covariance


@dataclasses.dataclass
class GaussianLVMParametrization:
    """Parameters of a particular Gaussian latent variable model
    resulting in interesting covariance structures.

    For `n_interacting=K` pairs of "strongly interacting" coordinates
    (X_1, Y_1), ..., (X_K, Y_K) we imagine the following generative model with

    3 + K + dim_x + dim_y + (dim_x - K) + (dim_y - K) = 3 + 2*(dim_x + dim_y) - K

    latent variables. They are independent and identically distributed according to
    Normal(0,1):

      - U_all: added to all variables X_i and Y_i, inducing "dense" interactions
      - U_X: added to all X_i variables, inducing dense interactions within X coordinates
      - U_Y: analogous to U_X
      - Z_1, ..., Z_K: Z_i is added to X_i and Y_i, inducing "strongly interacting" coordinates
      - E_1, ..., E_{dim_x}: E_i is added to X_i, increasing the variance
        and decreasing correlations
      - F_1, ..., F_{dim_y}: analogous to E_i, but for the Y variable
      - V_{dim_x-K+1}, ..., V_{dim_x}: V_i is added to X_i,
        increasing the variance of these entries.
        The motivation is that variables X_1, ..., X_K have increased variance due to Z_i
        and we may want to compensate for it
      - W_{dim_y-K+1}, ..., V_{dim_y}: analogous to V_i but for the Y variable

    Attrs:
        dim_x: the dimensionality of the X variable
        dim_y: the dimensionality of the Y variable
        n_interacting: number of strongly interacting components of X vs Y
          (strength controlled by parameter `lambd`)
        alpha: dense interactions between every pair of variables (U_all weight)
        beta_x: dense interactions between X components (U_X weight)
        beta_y: dense interactions between Y components (U_Y weight).
            If None, defaults to `beta_x`
        lambd: strength of strongly interacting components of X vs Y (Z_i weight)
        epsilon_x: individual noise, increasing variances of X components (E_i weight)
        epsilon_y: individual noise, increasing variances of Y components (F_i weight)
            If None, defaults to `epsilon_x`
        eta_x: increases variance of X components which are not strongly interacting
          via `n_interacting` and `lambd` (V_i weight)
        eta_y: increases variance of Y components which are not strongly interacting
          via `n_interacting` and `lambd` (W_i weight)
            If None, defaults to `eta_x`
    """

    dim_x: int
    dim_y: int
    n_interacting: int
    alpha: float
    lambd: float
    epsilon_x: float = 1.0
    epsilon_y: Optional[float] = None
    beta_x: float = 0.0
    beta_y: Optional[float] = None
    eta_x: Optional[float] = 0.0
    eta_y: Optional[float] = None

    def __post_init__(self) -> None:
        if not 0 <= self.n_interacting <= min(self.dim_x, self.dim_y):
            raise ValueError(
                f"n_interacting={self.n_interacting} must be between 0 and the smaller"
                f"of the variable dimensionalities ({self.dim_x} and {self.dim_y})."
            )
        self.epsilon_y = self.epsilon_x if self.epsilon_y is None else self.epsilon_y
        self.beta_y = self.beta_x if self.beta_y is None else self.beta_y
        self.eta_y = self.eta_x if self.eta_y is None else self.eta_y

    def _z_coefficients(self, i: int) -> np.ndarray:
        """For an index i (starting at 0) returns the array
        of coefficients corresponding to contributions of Z_j
        variables to either X_i or Y_i.

        Returns:
            array of shape (n_interacting,)
        """
        z_coef = np.zeros(self.n_interacting, dtype=float)
        if i < self.n_interacting:
            z_coef[i] = self.lambd
        return z_coef

    def _mixing_x(self, i: int) -> np.ndarray:
        """The array of shape (n_latent,) describing X_i.

        Args:
            i: index from the set {0, ..., dim_x-1}
        """
        u_coef = np.asarray([self.alpha, self.beta_x, 0.0], dtype=float)
        z_coef = self._z_coefficients(i)

        e_coef = np.zeros(self.dim_x, dtype=float)
        e_coef[i] = self.epsilon_x

        f_coef = np.zeros(self.dim_y, dtype=float)

        v_coef = np.zeros(self.dim_x - self.n_interacting, dtype=float)
        if i >= self.n_interacting:
            v_coef[i - self.n_interacting] = self.eta_x

        w_coef = np.zeros(self.dim_y - self.n_interacting, dtype=float)

        return np.concatenate(
            [
                u_coef,
                z_coef,
                e_coef,
                f_coef,
                v_coef,
                w_coef,
            ]
        )

    def _mixing_y(self, i: int) -> np.ndarray:
        """The array of shape (n_latent,) describing Y_i.

        Args:
            i: index from the set {0, ..., dim_y-1}
        """
        u_coef = np.asarray([self.alpha, 0.0, self.beta_y], dtype=float)
        z_coef = self._z_coefficients(i)

        e_coef = np.zeros(self.dim_x, dtype=float)

        f_coef = np.zeros(self.dim_y, dtype=float)
        f_coef[i] = self.epsilon_y

        v_coef = np.zeros(self.dim_x - self.n_interacting, dtype=float)

        w_coef = np.zeros(self.dim_y - self.n_interacting, dtype=float)
        if i >= self.n_interacting:
            w_coef[i - self.n_interacting] = self.eta_y

        return np.concatenate(
            [
                u_coef,
                z_coef,
                e_coef,
                f_coef,
                v_coef,
                w_coef,
            ]
        )

    @property
    def mixing(self) -> np.ndarray:
        """Matrix desscribing the linear mapping
        from the described latent variables to (X, Y).

        Returns:
            matrix of shape (dim_x + dim_y, n_latent)
            where `n_latent` is given by:
            3 + n_interacting + dim_x + dim_y + (dim_x - n_interacting) + (dim_y - n_interacting)
        """
        return np.vstack(
            [self._mixing_x(i) for i in range(self.dim_x)]
            + [self._mixing_y(i) for i in range(self.dim_y)]
        )

    @property
    def n_latent(self) -> int:
        return (
            3
            + self.n_interacting
            + self.dim_x
            + self.dim_y
            + (self.dim_x - self.n_interacting)
            + (self.dim_y - self.n_interacting)
        )

    @property
    def covariance(self) -> np.ndarray:
        for variable in [self.beta_y, self.epsilon_y, self.eta_y]:
            assert variable is not None

        n_all = self.dim_x + self.dim_y
        dim_x = self.dim_x

        # Add alpha^2 everywhere
        covariance = np.full(
            (n_all, n_all),
            fill_value=self.alpha**2,
            dtype=float,
        )

        # Add beta^2 to the blocks of Xs and Ys
        covariance[0:dim_x, 0:dim_x] += self.beta_x**2
        covariance[dim_x:n_all, dim_x:n_all] += self.beta_y**2

        # Add epsilon^2 to the diagonal
        for i in range(dim_x):
            covariance[i, i] += self.epsilon_x**2
        for j in range(dim_x, n_all):
            covariance[j, j] += self.epsilon_y**2

        # Add lambda^2 to Cov(Xi, Yi) (and Cov(Yi, Xi)) for i < n_interacting
        for i in range(self.n_interacting):
            j = dim_x + i
            covariance[i, j] += self.lambd**2
            covariance[j, i] += self.lambd**2
            covariance[i, i] += self.lambd**2
            covariance[j, j] += self.lambd**2

        # Add eta^2
        for i in range(self.n_interacting, dim_x):
            covariance[i, i] += self.eta_x**2

        for j in range(self.n_interacting + self.dim_x, n_all):
            covariance[j, j] += self.eta_y**2

        return covariance

    @property
    def correlation(self) -> np.ndarray:
        covariance = self.covariance
        variance = np.diag(covariance)
        return covariance / np.sqrt(np.outer(variance, variance))

    @property
    def latent_variable_labels(self) -> list[str]:
        return (
            ["$U_\\mathrm{all}$", "$U_X$", "$U_Y$"]
            + [f"$Z_{i+1}$" for i in range(self.n_interacting)]
            + [f"$E_{i+1}$" for i in range(self.dim_x)]
            + [f"$F_{i+1}$" for i in range(self.dim_y)]
            + [f"$V_{i+1}$" for i in range(self.n_interacting, self.dim_x)]
            + [f"$W_{i + 1}$" for i in range(self.n_interacting, self.dim_y)]
        )

    @property
    def xy_labels(self) -> list[str]:
        return [f"$X_{i+1}$" for i in range(self.dim_x)] + [
            f"$Y_{j+1}$" for j in range(self.dim_y)
        ]


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

    @property
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

    @property
    def correlation_interacting(self) -> float:
        """Correlation between X_i and Y_i for i < `n_interacting`.
        (Other correlations Cor(X_i, Y_j)=0).
        """
        return self.lambd**2 / (self.beta_x**2 + self.epsilon_x**2 + self.lambd**2)
