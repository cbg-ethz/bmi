import numpy as np
import pytest

import bmi.estimators.function_wrapper as fw
import bmi.estimators.ksg as ksg
import bmi.samplers.api as samplers


@pytest.mark.parametrize("mi", (0.5, 0.2))
@pytest.mark.parametrize("n_samples", (10,))
@pytest.mark.parametrize("dim_x", (1, 2))
@pytest.mark.parametrize("dim_y", (2, 3))
def test_wrapped_dummy(mi: float, n_samples: int, dim_x: int, dim_y: int) -> None:
    """We wrap a dummy estimator always returning `mi`."""

    def dummy_mi(x, y) -> float:
        return mi

    estimator = fw.FunctionalEstimator(dummy_mi)

    x = np.zeros((n_samples, dim_x))
    y = np.zeros((n_samples, dim_y))

    assert estimator.estimate(x, y) == pytest.approx(mi)
    # Check if we cast to lists
    assert estimator.estimate(x.tolist(), y) == pytest.approx(mi)
    assert estimator.estimate(x, y.tolist()) == pytest.approx(mi)


@pytest.mark.parametrize("correlation", (0.5,))
@pytest.mark.parametrize("dim_x", (2,))
@pytest.mark.parametrize("dim_y", (1,))
@pytest.mark.parametrize("n_samples", (30,))
def test_wrapped_ksg(
    correlation: float, dim_x: int, dim_y: int, n_samples: int, seed: int = 19
) -> None:
    """Now we compare with a wrapped KSG estimator."""

    def get_ksg():
        return ksg.KSGEnsembleFirstEstimator(neighborhoods=(3,))

    ksg_estimator = get_ksg()

    def func(x, y):
        return ksg_estimator.estimate(x, y)

    wrapped_estimator = fw.FunctionalEstimator(func)

    covariance = samplers.one_and_one_correlation_matrix(
        dim_x=dim_x, dim_y=dim_y, correlation=correlation
    )
    sampler = samplers.SplitMultinormal(dim_x=dim_x, dim_y=dim_y, covariance=covariance)

    x, y = sampler.sample(n_points=n_samples, rng=seed)

    assert wrapped_estimator.estimate(x, y) == pytest.approx(get_ksg().estimate(x, y))
