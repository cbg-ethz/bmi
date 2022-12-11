from typing import Optional, Union

import numpy as np
from jax import random
from numpy.typing import ArrayLike

from bmi.interface import KeyArray
from bmi.samplers.base import BaseSampler


def _can_be_covariance(mat):
    """Checks if `mat` can be a covariance matrix (positive-definite and symmetric)."""
    if np.all(mat != mat.transpose()):
        raise ValueError("Covariance matrix is not symmetric.")

    if not np.all(np.linalg.eigvals(mat) > 0):
        raise ValueError("Covariance matrix is not positive-definite.")


def _cast_to_rng(seed: Union[KeyArray, int]) -> KeyArray:
    """Casts `int` to a KeyArray."""
    if isinstance(seed, int) or isinstance(seed, np.integer):
        seed = int(seed)
        return random.PRNGKey(seed)
    else:
        return seed


class _Multinormal:
    """Auxiliary object for representing multivariate normal distributions."""

    def __init__(self, mean: ArrayLike, covariance: ArrayLike) -> None:
        """
        Args:
            mean: mean vector of the distribution, shape (dim,)
            covariance: covariance matrix of the distribution, shape (dim, dim)
        """
        # Mean and the covariance
        self._mean = np.asarray(mean)
        self._covariance = np.asarray(covariance)

        # The determinant of the covariance, used to calculate entropy
        self._det_covariance: float = np.linalg.det(self._covariance)

        # Dimensionality of the space
        self._dim = self._mean.shape[0]

        # Validate the shape
        if self._covariance.shape != (self._dim, self._dim):
            raise ValueError(
                f"Covariance has shape {self._covariance.shape}, expected "
                f"{(self._dim, self._dim)}."
            )

        # Validate symmetry, positive-definiteness
        _can_be_covariance(self._covariance)

    def sample(self, n_samples: int, key: KeyArray) -> np.ndarray:
        """Sample from the distribution.
        Args:
            n_samples: number of samples to generate
            key: JAX key for the pseudorandom number generator
        Returns:
            samples, shape (n_samples, dim)
        """
        return np.array(
            random.multivariate_normal(
                key=key, mean=self._mean, cov=self._covariance, shape=(n_samples,)
            )
        )

    @property
    def dim(self) -> int:
        """The dimensionality."""
        return self._dim

    def entropy(self) -> float:
        """Entropy in nats."""
        return 0.5 * (np.log(self._det_covariance) + self.dim * (1 + np.log(2 * np.pi)))


class SplitMultinormal(BaseSampler):
    """Represents two multinormal variables.

    Covariance matrix should have the block form:

        Cov[XX] Cov[XY]
        Cov[YX] Cov[YY]

    where:
      - Cov[XX] is the covariance matrix of X variable (shape (dim_x, dim_x)),
      - Cov[YY] is the covariance of the Y variable (shape (dim_y, dim_y))
      - Cov[XY] and Cov[YX] (being transposes of each other, as the matrix is symmetric,
            of shapes (dim_x, dim_y) or transposed one)
        describe the interaction between X and Y.
    """

    def __init__(
        self, *, dim_x: int, dim_y: int, covariance: ArrayLike, mean: Optional[ArrayLike] = None
    ) -> None:
        """

        Args:
            dim_x: dimension of the X space
            dim_y: dimension of the Y space
            mean: mean vector, shape (n,) where n = dim_x + dim_y. Default: zero vector
            covariance: covariance matrix, shape (n, n)
        """
        super().__init__(dim_x=dim_x, dim_y=dim_y)

        # The default mean vector is zero
        if mean is None:
            mean = np.zeros(dim_x + dim_y)

        # Set mean and covariance
        self._mean = np.array(mean)
        self._covariance = np.array(covariance)
        self._validate_shapes()

        self._joint_distribution = _Multinormal(mean=self._mean, covariance=self._covariance)
        self._x_distribution = _Multinormal(
            mean=self._mean[:dim_x], covariance=self._covariance[:dim_x, :dim_x]
        )
        self._y_distribution = _Multinormal(
            mean=self._mean[dim_x:], covariance=self._covariance[dim_x:, dim_x:]
        )

    def _validate_shapes(self) -> None:
        n = self.dim_total

        if self._mean.shape != (n,):
            raise ValueError(f"Mean vector has shape {self._mean.shape}, expected ({n},).")
        if self._covariance.shape != (n, n):
            raise ValueError(
                f"Covariance matrix has shape {self._covariance.shape}, " f"expected ({n}, {n})."
            )

    def sample(self, n_points: int, rng: Union[KeyArray, int]) -> tuple[np.ndarray, np.ndarray]:
        rng = _cast_to_rng(rng)
        xy = self._joint_distribution.sample(n_samples=n_points, key=rng)
        return xy[..., : self._dim_x], xy[..., self.dim_x :]  # noqa: E203

    def mutual_information(self) -> float:
        """Calculates the mutual information I(X; Y) using an exact formula.
        Returns:
            mutual information, in nats
        Mutual information is given by
            0.5 * log( det(covariance_x) * det(covariance_y) / det(full covariance) )
        which follows from the formula
            I(X; Y) = H(X) + H(Y) - H(X, Y)
        and the entropy of the multinormal distribution.
        """
        h_x = self._x_distribution.entropy()
        h_y = self._y_distribution.entropy()
        h_xy = self._joint_distribution.entropy()
        mi = h_x + h_y - h_xy  # Mutual information estimate
        return max(0.0, mi)  # Mutual information is always non-negative


class BivariateNormalSampler(BaseSampler):
    """A special case of a general multivariate normal sampler,
    where both X and Y are one-dimensional Gaussian variables with a given correlation."""

    def __init__(
        self,
        correlation: float,
        std_x: float = 1.0,
        std_y: float = 1.0,
        mean_x: float = 0.0,
        mean_y: float = 0.0,
    ) -> None:
        super().__init__(dim_x=1, dim_y=1)

        if min(std_x, std_y) <= 0:
            raise ValueError(
                f"Standard deviations {std_x} and {std_y} " f"need to be strictly greater than 0."
            )
        if not (-1 < correlation < 1):
            raise ValueError(f"Correlation {correlation} must be in the interval (-1, 1).")
        self._correlation = correlation

        cov = correlation * std_x * std_y
        var_x = std_x**2
        var_y = std_y**2

        covariance = np.asarray(
            [
                [var_x, cov],
                [cov, var_y],
            ]
        )
        mean = np.asarray([mean_x, mean_y])

        self._sampler = SplitMultinormal(
            dim_x=1,
            dim_y=1,
            covariance=covariance,
            mean=mean,
        )

    def sample(self, n_points: int, rng: Union[KeyArray, int]) -> tuple[np.ndarray, np.ndarray]:
        return self._sampler.sample(n_points=n_points, rng=rng)

    def correlation(self) -> float:
        return self._correlation

    def mutual_information(self) -> float:
        # This is faster and probably more stable than the more general expression
        # involving the determinants
        return -0.5 * np.log(1 - self._correlation**2)
