import numpy as np
import pytest  # pytype: disable=import-error
from jax import random  # pytype: disable=import-error

import bmi.estimators.ksg as ksg
from bmi.samplers.splitmultinormal import SplitMultinormal


def random_covariance(size: int, jitter: float = 1e1, rng=0):
    a = np.random.default_rng(rng).normal(0, 1, size=(size, size))

    return np.dot(a, a.T) + np.diag([jitter] * size)


def test_digamma():
    """Tests whether the digamma function we use seems right."""
    assert ksg._DIGAMMA(1) == pytest.approx(-0.5772156, abs=1e-5)

    for k in range(2, 5):
        assert ksg._DIGAMMA(k + 1) == pytest.approx(ksg._DIGAMMA(k) + 1 / k, rel=0.01)


@pytest.mark.parametrize("n_points", [200])
@pytest.mark.parametrize("k", [10])
@pytest.mark.parametrize("correlation", [0.0, 0.5, 0.8])
def test_estimate_mi_ksg_2d(n_points: int, k: int, correlation: float) -> None:
    """Simple tests for the KSG estimator with 2D Gaussian with known correlation."""
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

    estimator = ksg.KSGEnsembleFirstEstimator(neighborhoods=(k,), standardize=True)
    estimated_mi = estimator.estimate(points_x, points_y)

    true_mi = distribution.mutual_information()

    assert estimated_mi == pytest.approx(true_mi, rel=0.15, abs=0.12)


@pytest.mark.parametrize("n_points", [250])
@pytest.mark.parametrize("k", [10])
@pytest.mark.parametrize("dims", [(1, 2), (2, 2)])
def test_estimate_mi_ksg(n_points: int, k: int, dims: tuple[int, int]) -> None:
    dim_x, dim_y = dims

    distribution = SplitMultinormal(
        dim_x=dim_x,
        dim_y=dim_y,
        mean=np.zeros(dim_x + dim_y),
        covariance=random_covariance(dim_x + dim_y),
    )

    rng = random.PRNGKey(20)
    points_x, points_y = distribution.sample(n_points, rng=rng)

    estimator = ksg.KSGEnsembleFirstEstimator(neighborhoods=(k,), standardize=True)
    estimated_mi = estimator.estimate(points_x, points_y)
    true_mi = distribution.mutual_information()

    # Approximate the MI to 10% and take correction for very small values
    assert estimated_mi == pytest.approx(true_mi, rel=0.1, abs=0.02)


@pytest.mark.parametrize("difference", [0, 1, 5])
def test_ksg_more_neighbors_than_points(
    difference: int, neighborhoods: tuple[int, ...] = (10,), dim: int = 3
) -> None:
    """Tests for a fitting error if the number of required neighbors
    is greater than the number of points provided."""
    n_points = max(neighborhoods) - difference  # We don't have enough data points
    x = np.zeros((n_points, dim))

    estimator = ksg.KSGEnsembleFirstEstimator(neighborhoods=neighborhoods)

    with pytest.raises(ValueError):
        estimator.fit(x, x)


@pytest.mark.parametrize("neighborhoods", [(2, 5), (1, 10)])
def test_ksg_fit_predict(neighborhoods: tuple[int, ...], n_samples: int = 15) -> None:
    """Tests whether the estimator can be fitted
    and if it allows to get the predictions afterwards."""

    distribution = SplitMultinormal(
        mean=np.zeros(2),
        covariance=np.eye(2),
        dim_x=1,
        dim_y=1,
    )

    rng = random.PRNGKey(10)
    x_sample, y_sample = distribution.sample(n_samples, rng=rng)

    estimator = ksg.KSGEnsembleFirstEstimator(neighborhoods=neighborhoods)
    estimator.fit(x_sample, y_sample)

    predictions = estimator.get_predictions()

    for k in neighborhoods:
        assert predictions.get(k) is not None
        assert predictions[k] >= 0

    assert list(predictions.keys()) == list(neighborhoods)


def test_ksg_predict_before_fit() -> None:
    estimator = ksg.KSGEnsembleFirstEstimator()

    with pytest.raises(ksg.EstimatorNotFittedException):
        estimator.get_predictions()
