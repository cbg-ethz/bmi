"""Most important interfaces of the package.

Note:
    The `interface` module CANNOT import anything from the developed package.
    This restriction is to ensure that any subpackage can import from
    the `interface` module and that we do not run into the circular imports issue.
"""

import pathlib
from abc import abstractmethod
from typing import Any, Optional, Protocol, Union

import numpy as np
import pydantic
from numpy.typing import ArrayLike


class BaseModel(pydantic.BaseModel):  # pytype: disable=invalid-annotation
    """As pytype has a false-positive problem with BaseModel and our CI fails,
    we need to create this dummy class.

    We can remove it once the problem has been solved:
    https://github.com/google/pytype/issues/1105
    """

    pass


# This should be updated to the PRNGKeyArray (or possibly union with Any)
# when it becomes a part of public JAX API
KeyArray = Any
Pathlike = Union[str, pathlib.Path]
Seed = int


class EstimateResult(BaseModel):
    mi_estimate: float
    time_in_seconds: Optional[float] = None
    additional_information: dict = pydantic.Field(default_factory=dict)


class IMutualInformationPointEstimator(Protocol):
    """Interface for the mutual information estimator returning point estimates.
    All estimators should be implementations of this interface."""

    @abstractmethod
    def estimate(self, x: ArrayLike, y: ArrayLike) -> float:
        """A point estimate of MI(X; Y) from an i.i.d. sample from the $P(X, Y)$ distribution.

        Args:
            x: shape `(n_samples, dim_x)`
            y: shape `(n_samples, dim_y)`

        Returns:
            mutual information estimate
        """
        raise NotImplementedError

    def estimate_with_info(self, x: ArrayLike, y: ArrayLike) -> EstimateResult:
        """Allows for reporting additional information about the run."""
        return EstimateResult(mi_estimate=self.estimate(x, y))

    @abstractmethod
    def parameters(self) -> BaseModel:
        """Returns the parameters of the estimator."""
        raise NotImplementedError


class ISampler(Protocol):
    """Interface for a distribution $P(X, Y)$."""

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
