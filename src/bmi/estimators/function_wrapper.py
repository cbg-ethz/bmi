from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike

from bmi.interface import BaseModel, IMutualInformationPointEstimator


class _EmptyParams(BaseModel):
    pass


class FunctionalEstimator(IMutualInformationPointEstimator):
    """Wraps a function estimating mutual information to the estimator interface."""

    def __init__(
        self,
        /,
        func: Callable[[np.ndarray, np.ndarray], float],
        *,
        params: Optional[BaseModel] = None,
    ) -> None:
        """
        Args:
            func: takes X (shape (n_samples, dim_x) and Y samples (shape (n_samples, dim_y))
              and returns the mutual information estimate (float)
            params: parameters of the estimator
        """
        self._estimator_function = func
        self._params = params if params is not None else _EmptyParams()

    def estimate(self, x: ArrayLike, y: ArrayLike) -> float:
        x = np.asarray(x)
        y = np.asarray(y)

        return self._estimator_function(x, y)

    def parameters(self) -> BaseModel:
        return self._params
