"""Kraskov estimators."""
from typing import Literal, Optional, Sequence, cast

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import digamma as _DIGAMMA
from sklearn import metrics

from bmi.estimators.base import EstimatorNotFittedException
from bmi.interface import IMutualInformationPointEstimator
from bmi.utils import ProductSpace, chunker

_AllowedContinuousMetric = Literal["euclidean", "manhattan", "chebyshev"]


def _cast_and_check_neighborhoods(neighborhoods: Sequence[int]) -> list[int]:
    """Auxiliary function used to make sure that the provided `neighborhoods` are right
    and standardize a sequence into a list.

    Raises:
        ValueError, if any of `neighborhoods` is not positive
    """
    if not len(neighborhoods):
        raise ValueError("Neighborhoods list must be non-empty.")
    if min(neighborhoods) < 1:
        raise ValueError("Each neighborhood must be at least 1.")
    return list(neighborhoods)


class KSGEnsembleFirstEstimator(IMutualInformationPointEstimator):
    """Ensemble estimator built using the first approximation (equation (8) in the paper)."""

    def __init__(
        self,
        neighborhoods: Sequence[int] = (5, 10),
        standardize: bool = True,
        metric_x: _AllowedContinuousMetric = "euclidean",
        metric_y: Optional[_AllowedContinuousMetric] = None,
        n_jobs: int = 1,
        chunk_size: int = 10,
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
            chunk_size: internal batch size, used to speed up the computations while fitting
              into the memory

        Note:
            If you use Chebyshev (l-infinity) distance for both X and Y,
            `KSGChebyshevEstimator` may be faster.
        """

        self._neighborhoods: list[int] = _cast_and_check_neighborhoods(neighborhoods)
        self._standardize = standardize

        self._n_jobs: int = n_jobs
        self._chunk_size: int = chunk_size

        self._metric_x: _AllowedContinuousMetric = metric_x
        self._metric_y: _AllowedContinuousMetric = (
            metric_y or metric_x
        )  # If `metric_y` is None, use `metric_x`

        self._fitted = False
        self._mi_dict = dict()  # set by fit()

    def fit(self, x: ArrayLike, y: ArrayLike) -> None:
        space = ProductSpace(x, y, standardize=self._standardize)

        if len(space) <= max(self._neighborhoods):
            raise ValueError(
                f"Maximum neighborhood used is {max(self._neighborhoods)} "
                f"but the number of points provided is only {len(space)}."
            )

        digammas_dict = {k: [] for k in self._neighborhoods}

        n_points = len(space)
        for batch_index in chunker(n_items=n_points, chunk_size=self._chunk_size):
            # All these arrays have shape (batch_size, n_points)
            distances_x = metrics.pairwise_distances(
                space.x[batch_index], space.x, metric=self._metric_x, n_jobs=self._n_jobs
            )
            distances_y = metrics.pairwise_distances(
                space.y[batch_index], space.y, metric=self._metric_y, n_jobs=self._n_jobs
            )
            # Distance in space `X x Y` is the maximum distance.
            # Shape is also (batch_size, n_points)
            distances_xy = np.maximum(distances_x, distances_y)

            # Indices to the closest points in the `X x Y` space.
            # Shape is also (batch_size, n_points)
            closest_points = np.argsort(distances_xy, axis=-1)

            for k in self._neighborhoods:
                # Note that the points are 0-indexed and that the "0th neighbor"
                # is the point itself (as distance(x, x) = 0 is the smallest possible)
                # Hence, the kth neighbour is at index k
                kth_neighbors = closest_points[:, k]  # Shape (batch_size,)
                # Distance to the k-th neighbor for each point
                distances_k = distances_xy[
                    np.arange(len(kth_neighbors)), kth_neighbors
                ]  # Shape (batch_size,)

                # Don't include the `i`th point itself in n_x and n_y
                n_x = (distances_x < distances_k[:, None]).sum(axis=1) - 1  # Shape (batch_size,)
                n_y = (distances_y < distances_k[:, None]).sum(axis=1) - 1  # Shape (batch_size,)

                digammas = _DIGAMMA(n_x + 1) + _DIGAMMA(n_y + 1)  # Shape (batch_size,)
                # We calculate mean(digammas) over all the points rather than the batch
                digammas_mean_contribution = np.sum(digammas / n_points)
                digammas_dict[k].append(digammas_mean_contribution)

        for k, digammas in digammas_dict.items():
            # As the mean over all the points was calculated for each chunk separately,
            # we should add the contributions from each chunk
            mi_estimate = _DIGAMMA(k) - np.sum(digammas) + _DIGAMMA(n_points)
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


class KSGEnsembleFirstEstimatorSlow(IMutualInformationPointEstimator):
    """Ensemble estimator built using the first approximation (equation (8) in the paper).

    Note:
        `KSGEnsembleFirstEstimator` is a faster, NumPy-based, version of this one.
        We decided to retain it for instructional purposes (as it's easier to read)
    """

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
        """
        self._neighborhoods = _cast_and_check_neighborhoods(neighborhoods)
        self._standardize = standardize

        self._n_jobs: int = n_jobs

        self._metric_x: _AllowedContinuousMetric = metric_x
        self._metric_y: _AllowedContinuousMetric = (
            metric_y or metric_x
        )  # If `metric_y` is None, use `metric_x`

        self._fitted = False
        self._mi_dict = dict()  # set by fit()

    def fit(self, x: ArrayLike, y: ArrayLike) -> None:
        space = ProductSpace(x=x, y=y, standardize=self._standardize)

        n_points = len(space)
        if n_points <= max(self._neighborhoods):
            raise ValueError(
                f"Maximum neighborhood used is {max(self._neighborhoods)} "
                f"but the number of points provided is only {n_points}."
            )

        digammas_dict = {k: [] for k in self._neighborhoods}

        for index in range(n_points):
            # Distances from x[index] to all the points:
            distances_x = metrics.pairwise_distances(
                space.x[None, index], space.x, metric=self._metric_x, n_jobs=self._n_jobs
            )[0, :]
            distances_y = metrics.pairwise_distances(
                space.y[None, index], space.y, metric=self._metric_y, n_jobs=self._n_jobs
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
