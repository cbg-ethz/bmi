from typing import Optional

import numpy as np
import pytest
from jax import random

from bmi.estimators.histogram import HistogramEstimator
from bmi.samplers.splitmultinormal import SplitMultinormal


@pytest.mark.parametrize("n_points", [2000])
@pytest.mark.parametrize("correlation", [0.0, 0.5, 0.8])
@pytest.mark.parametrize("n_bins_x", [8, 10, 15])
@pytest.mark.parametrize("n_bins_y", [None, 10])
def test_estimate_mi_histogram_2d(
    n_points: int, correlation: float, n_bins_x: int, n_bins_y: Optional[int]
) -> None:
    """Histogram-based approach for a 2D Gaussian with known correlation."""
    covariance = np.array(
        [
            [1.0, correlation],
            [correlation, 1.0],
        ]
    )
    distribution = SplitMultinormal(
        dim_x=1,
        dim_y=1,
        mean=np.zeros(2),
        covariance=covariance,
    )
    rng = random.PRNGKey(19)
    points_x, points_y = distribution.sample(n_points, rng=rng)

    estimator = HistogramEstimator(n_bins_x=n_bins_x, n_bins_y=n_bins_y)
    estimated_mi = estimator.estimate(points_x, points_y)

    true_mi = distribution.mutual_information()

    assert estimated_mi == pytest.approx(true_mi, rel=0.12, abs=0.1)
