import numpy as np
from numpy.typing import ArrayLike
from sklearn.neighbors import KernelDensity

from bmi.interface import IMutualInformationPointEstimator
from bmi.utils import ProductSpace


def _entropy(estimator, samples):
    estimator.fit(samples)
    log_probs = estimator.score_samples(samples)
    return -np.mean(log_probs)


class KDEMutualInformationEstimator(IMutualInformationPointEstimator):
    def __init__(
        self,
        bandwidth_xy="scott",
        bandwidth_x="scott",
        bandwidth_y="scott",
        standardize: bool = True,
    ) -> None:
        self._kde_x = KernelDensity(bandwidth=bandwidth_x)
        self._kde_y = KernelDensity(bandwidth=bandwidth_y)
        self._kde_xy = KernelDensity(bandwidth=bandwidth_xy)

        self._standardize = standardize

    def estimate(self, x: ArrayLike, y: ArrayLike) -> float:
        space = ProductSpace(x=x, y=y)

        h_x = _entropy(self._kde_x, space.x)
        h_y = _entropy(self._kde_y, space.y)
        h_xy = _entropy(self._kde_xy, space.xy)

        return h_x + h_y - h_xy
