"""Tests of the neural estimators on simple distributions."""

import equinox as eqx
import numpy as np
import pytest

import bmi.estimators.neural as neural
import bmi.samplers as samplers


@pytest.mark.parametrize(
    "estimator_class",
    [
        neural.DonskerVaradhanEstimator,
        neural.NWJEstimator,
        neural.InfoNCEEstimator,
    ],
)
def test_estimator_estimates(estimator_class, n_points: int = 5_000) -> None:
    estimator = estimator_class(batch_size=100, learning_rate=0.15, max_n_steps=1_000)

    cov = np.asarray(
        [
            [1.0, 0.0, 0.7],
            [0.0, 1.0, 0.3],
            [0.7, 0.3, 1.0],
        ]
    )

    sampler = samplers.SplitMultinormal(dim_x=2, dim_y=1, covariance=cov)
    xs, ys = sampler.sample(n_points=n_points, rng=0)

    estimated_mi = estimator.estimate(xs, ys)
    true_mi = sampler.mutual_information()

    # TODO(frdrc): pick good abs/rel
    assert estimated_mi == pytest.approx(true_mi, abs=0.1, rel=0.1)


@pytest.mark.parametrize(
    "estimator_class",
    [
        neural.DonskerVaradhanEstimator,
        neural.NWJEstimator,
        neural.InfoNCEEstimator,
    ],
)
def test_critic_saved(estimator_class, n_points: int = 100) -> None:
    """Tests whether the critic is saved after estimation."""
    estimator = estimator_class(batch_size=16, learning_rate=0.15, max_n_steps=100)
    sampler = samplers.SplitMultinormal(dim_x=2, dim_y=1, covariance=np.eye(3))
    xs, ys = sampler.sample(n_points=n_points, rng=0)

    estimator.estimate(xs, ys)
    assert isinstance(estimator.trained_critic, eqx.Module)


@pytest.mark.parametrize(
    "estimator_class",
    [
        neural.DonskerVaradhanEstimator,
        neural.NWJEstimator,
        neural.InfoNCEEstimator,
        neural.MINEEstimator,
    ],
)
def test_batch_size_error(estimator_class, n_points: int = 100) -> None:
    """Tests whether the critic is saved after estimation."""
    batch_size = n_points + 1
    estimator = estimator_class(batch_size=batch_size, learning_rate=0.15, max_n_steps=100)
    sampler = samplers.SplitMultinormal(dim_x=2, dim_y=1, covariance=np.eye(3))
    xs, ys = sampler.sample(n_points=n_points, rng=0)

    result = estimator.estimate_with_info(xs, ys)
    assert np.isnan(result.mi_estimate)
    assert "batch_size_larger_than_train" in result.additional_information
    assert result.additional_information["batch_size_larger_than_train"]
