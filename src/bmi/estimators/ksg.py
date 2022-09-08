"""Kraskov estimators."""
from typing import Sequence, cast

import numpy as np
from numpy.typing import ArrayLike  # pytype: disable=import-error
from scipy import special  # pytype: disable=import-error
from sklearn import metrics, preprocessing  # pytype: disable=import-error

from bmi.estimators.base import EstimatorNotFittedException
from bmi.interface import IMutualInformationPointEstimator

_DIGAMMA = special.digamma


class KSGEnsembleFirstEstimator(IMutualInformationPointEstimator):
    """Ensemble estimator built using the first approximation (equation (8) in the paper)."""

    def __init__(self, neighborhoods: Sequence[int] = (5, 10), standardize: bool = True) -> None:
        """

        Args:
            neighborhoods: sequence of positive integers,
              specifying the size of neighborhood for MI calculation
            standardize: whether to standardize the data before MI calculation, by default true
        """
        if min(neighborhoods) < 1:
            raise ValueError("Each neighborhood must be at least 1.")

        self._neighborhoods = list(neighborhoods)
        self._standardize = standardize

        self._fitted = False
        self._mi_dict = {k: None for k in neighborhoods}

    def fit(self, x: ArrayLike, y: ArrayLike) -> None:
        x, y = np.array(x), np.array(y)

        if len(x) != len(y):
            raise ValueError(f"Arrays have different length: {len(x)} != {len(y)}.")

        if self._standardize:
            x = preprocessing.StandardScaler(copy=False).fit_transform(x)
            y = preprocessing.StandardScaler(copy=False).fit_transform(y)

        digammas_dict = {k: [] for k in self._neighborhoods}

        n_points = len(x)
        for index in range(n_points):
            # Distances from x[index] to all the points:
            distances_x = metrics.pairwise_distances(x[None, index], x)[0, :]
            distances_y = metrics.pairwise_distances(y[None, index], y)[0, :]

            # In the product (XxY) space we use the maximum distance
            distances_z = np.maximum(distances_x, distances_y)
            # And we sort the point indices by being the closest to the considered one
            closest_points = sorted(range(len(distances_z)), key=lambda i: distances_z[i])

            for k in self._neighborhoods:
                # Note that the points are 0-indexed and that the "0th neighbor"
                # is the point itself (as distance(x, x) = 0 is the smallest possible)
                # Hence, the kth neighbour is at index k
                kth_neighbour = closest_points[k]
                distance = distances_z[kth_neighbour]

                # Don't include the `i`th point itself in n_x and n_y
                n_x = (distances_x < distance).sum() - 1
                n_y = (distances_y < distance).sum() - 1

                digammas_per_point = _DIGAMMA(n_x + 1) + _DIGAMMA(n_y + 1)
                digammas_dict[k].append(digammas_per_point)

        for k, digammas in digammas_dict.items():
            mi_estimate = _DIGAMMA(k) - np.mean(digammas) + _DIGAMMA(n_points)
            self._mi_dict[k] = max(0.0, mi_estimate)

        self._fitted = True

    def get_predictions(self) -> dict[int, float]:
        if not self._fitted:
            raise EstimatorNotFittedException
        return cast(dict[int, float], self._mi_dict.copy())

    def estimate(self, x: ArrayLike, y: ArrayLike) -> float:
        self.fit(x, y)
        predictions = self.get_predictions().values()
        return cast(float, np.mean(predictions))
