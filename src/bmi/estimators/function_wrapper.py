from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

from bmi.interface import IMutualInformationPointEstimator


class FunctionalEstimator(IMutualInformationPointEstimator):
    """Wraps a function estimating mutual information to the estimator interface."""

    def __init__(self, /, func: Callable[[np.ndarray, np.ndarray], float]) -> None:
        """
        Args:
            func: takes X (shape (n_samples, dim_x) and Y samples (shape (n_samples, dim_y))
              and returns the mutual information estimate (float)
        """
        self._estimator_function = func

    def estimate(self, x: ArrayLike, y: ArrayLike) -> float:
        x = np.asarray(x)
        y = np.asarray(y)

        return self._estimator_function(x, y)
