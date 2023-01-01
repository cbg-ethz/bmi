"""Most important interfaces of the package."""
import pathlib
from abc import abstractmethod
from typing import Any, Protocol, Union

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel

# This should be updated to the PRNGKeyArray (or possibly union with Any)
# when it becomes a part of public JAX API
KeyArray = Any
Pathlike = Union[str, pathlib.Path]


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

    @abstractmethod
    def parameters(self) -> BaseModel:
        """Returns the parameters of the estimator."""
        raise NotImplementedError


class ISampler(Protocol):
    """Interface for a distribution P(X, Y)."""

    @abstractmethod
    def sample(self, n_points: int, rng: Union[int, KeyArray]) -> tuple[np.ndarray, np.ndarray]:
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
