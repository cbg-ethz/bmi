"""Most important interfaces of the package."""
from typing import Any, Protocol, Tuple

import numpy as np
from numpy.typing import ArrayLike  # pytype: disable=import-error


class IMutualInformationPointEstimator(Protocol):
    """Interface for the mutual information estimator."""

    def estimate(self, x_samples: ArrayLike, y_samples: ArrayLike) -> float:
        """A point estimate of MI(X; Y) from a sample from P(X, Y) distribution.

        Args:
            x_samples: shape (n_samples, dim_x)
            y_samples: shape (n_samples, dim_y)

        Returns:
            mutual information estimate
        """
        raise NotImplementedError


class IDistribution(Protocol):
    """Interface for a distribution P(X, Y)."""

    def sample(self, n_points: int, rng: Any) -> Tuple[np.ndarray, np.ndarray]:
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
    def dim_x(self) -> int:
        """Dimension of the space in which the `X` variable is valued."""
        raise NotImplementedError

    @property
    def dim_y(self) -> int:
        """Dimension of the space in which the `Y` variable is valued."""
        raise NotImplementedError

    @property
    def dim_total(self) -> int:
        """Dimension of the space in which the `(X, Y)` variable is valued.
        Should be equal to the sum of `dim_x` and `dim_y`.
        """
        raise NotImplementedError
