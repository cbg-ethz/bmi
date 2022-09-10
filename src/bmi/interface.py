"""Most important interfaces of the package."""
from abc import abstractmethod
from typing import Any, Protocol, Union

import numpy as np
from numpy.typing import ArrayLike  # pytype: disable=import-error

# Because we want to use a type annotation from private JAX API,
# we will wrap it into the try-except statement
try:
    from jax._src.prng import PRNGKeyArray  # pytype: disable=import-error
except ImportError:

    class PRNGKeyArray:
        pass


# This type can be used to annotate if PRNGKeyArray (for JAX sampling) is needed.
KeyArray = Union[PRNGKeyArray, Any]  # pytype: disable=invalid-annotation


class IMutualInformationPointEstimator(Protocol):
    """Interface for the mutual information estimator."""

    @abstractmethod
    def estimate(self, x: ArrayLike, y: ArrayLike) -> float:
        """A point estimate of MI(X; Y) from an i.i.d sample from the P(X, Y) distribution.

        Args:
            x: shape (n_samples, dim_x)
            y: shape (n_samples, dim_y)

        Returns:
            mutual information estimate
        """
        raise NotImplementedError


class ISampler(Protocol):
    """Interface for a distribution P(X, Y)."""

    @abstractmethod
    def sample(self, n_points: int, rng: KeyArray) -> tuple[np.ndarray, np.ndarray]:
        """Returns a sample from the joint distribution P(X, Y).

        Args:
            n_points: sample size
            rng: pseudorandom number generator

        Returns:
            X samples, shape (n_points, dim_x)
            Y samples, shape (n_points, dim_y). Note that these samples are paired with X samples.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dim_x(self) -> int:
        """Dimension of the space in which the `X` variable is valued."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dim_y(self) -> int:
        """Dimension of the space in which the `Y` variable is valued."""
        raise NotImplementedError

    @property
    def dim_total(self) -> int:
        """Dimension of the space in which the `(X, Y)` variable is valued.
        Should be equal to the sum of `dim_x` and `dim_y`.
        """
        return self.dim_x + self.dim_y

    @abstractmethod
    def mutual_information(self) -> float:
        """Mutual information MI(X; Y)."""
        raise NotImplementedError
