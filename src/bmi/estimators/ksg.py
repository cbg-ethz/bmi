"""Kraskov estimators."""
from typing import Literal, Optional, Sequence, cast

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import digamma as _DIGAMMA
from sklearn import metrics, preprocessing

from bmi.estimators.base import EstimatorNotFittedException
from bmi.interface import IMutualInformationPointEstimator

_AllowedContinuousMetric = Literal["euclidean", "manhattan", "chebyshev", "mahalonobis"]


class KSGEnsembleFirstEstimator(IMutualInformationPointEstimator):
    """Ensemble estimator built using the first approximation (equation (8) in the paper)."""

    def __init__(
        self,
        neighborhoods: Sequence[int] = (5, 10),
        standardize: bool = True,
        metric_x: _AllowedContinuousMetric = "euclidean",
        metric_y: Optional[_AllowedContinuousMetric] = None,
        n_jobs: int = 1,
    ) -> None:
        """

        Args:
            neighborhoods: sequence of positive integers,
              specifying the size of neighborhood for MI calculation
            standardize: whether to standardize the data before MI calculation, by default true
            metric_x: metric on the X space
            metric_y: metric on the Y space. If None, `metric_x` will be used
            n_jobs: number of jobs to be launched to compute distances.
              Use -1 to use all processors.

        Note:
            If you use Chebyshev (l-infinity) distance for both X and Y,
            `KSGChebyshevEstimator` may be faster.
        """
        if min(neighborhoods) < 1:
            raise ValueError("Each neighborhood must be at least 1.")

        self._neighborhoods = list(neighborhoods)
        self._standardize = standardize

        self._n_jobs: int = n_jobs

        self._metric_x: _AllowedContinuousMetric = metric_x
        self._metric_y: _AllowedContinuousMetric = (
            metric_y or metric_x
        )  # If `metric_y` is None, use `metric_x`

        self._fitted = False
        self._mi_dict = dict()  # set by fit()

    def fit(self, x: ArrayLike, y: ArrayLike) -> None:
        x, y = np.array(x), np.array(y)

        if len(x) != len(y):
            raise ValueError(f"Arrays have different length: {len(x)} != {len(y)}.")
        if len(x) <= max(self._neighborhoods):
            raise ValueError(
                f"Maximum neighborhood used is {max(self._neighborhoods)} "
                f"but the number of points provided is only {len(x)}."
            )

        if self._standardize:
            x: np.ndarray = preprocessing.StandardScaler(copy=False).fit_transform(x)
            y: np.ndarray = preprocessing.StandardScaler(copy=False).fit_transform(y)

        digammas_dict = {k: [] for k in self._neighborhoods}

        n_points = np.shape(x)[0]
        for index in range(n_points):
            # Distances from x[index] to all the points:
            distances_x = metrics.pairwise_distances(
                x[None, index], x, metric=self._metric_x, n_jobs=self._n_jobs
            )[0, :]
            distances_y = metrics.pairwise_distances(
                y[None, index], y, metric=self._metric_y, n_jobs=self._n_jobs
            )[0, :]

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
        predictions = np.asarray(list(self.get_predictions().values()))
        return cast(float, np.mean(predictions))


class KSGEnsembleChebyshevEstimator(IMutualInformationPointEstimator):
    """Mutual information estimator based on fast nearest neighbours,
    available when Chebyshev (l-infty) metric is used for both X and Y spaces.
    """

    raise NotImplementedError
