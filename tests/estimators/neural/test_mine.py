import equinox as eqx
import jax.random
import numpy as np
import pytest

import bmi.estimators.neural._mine_estimator as mine
import bmi.samplers as samplers


def test_mine_estimator_3d(n_points: int = 8_000) -> None:
    rng = jax.random.PRNGKey(23)
    key_mlp, key_train, key_sampler = jax.random.split(rng, 3)

    cov = np.asarray(
        [
            [1.0, 0.0, 0.7],
            [0.0, 1.0, 0.3],
            [0.7, 0.3, 1.0],
        ]
    )

    sampler = samplers.SplitMultinormal(dim_x=2, dim_y=1, covariance=cov)
    xs, ys = sampler.sample(n_points=n_points, rng=key_sampler)

    estimator = mine.MINEEstimator(
        verbose=False,
    )

    estimated_mi = estimator.estimate(xs, ys)
    true_mi = sampler.mutual_information()

    assert estimated_mi == pytest.approx(true_mi, abs=0.05, rel=0.1)


def test_mine_estimator_2d(n_points: int = 8_000, correlation: float = 0.7) -> None:
    distribution = samplers.BivariateNormalSampler(correlation=correlation)
    points_x, points_y = distribution.sample(n_points, rng=19)

    estimator = mine.MINEEstimator(
        hidden_layers=(10, 5),
        verbose=False,
    )

    true_mi = distribution.mutual_information()
    estimate = estimator.estimate(points_x, points_y)
    assert estimate == pytest.approx(true_mi, abs=0.05, rel=0.1)


def test_mine_estimator_logs(n_points: int = 20, correlation: float = 0.5) -> None:
    """Checks if we have training history in logs."""
    distribution = samplers.BivariateNormalSampler(correlation=correlation)
    points_x, points_y = distribution.sample(n_points, rng=101)

    test_every_n_steps = 2
    estimator = mine.MINEEstimator(
        hidden_layers=(3, 2),
        verbose=False,
        max_n_steps=10,
        batch_size=max(5, n_points // 10),
        train_test_split=0.5,
        test_every_n_steps=test_every_n_steps,
    )
    estimate_result = estimator.estimate_with_info(points_x, points_y)
    n_steps = estimate_result.additional_information["n_training_steps"]
    assert n_steps > 1

    assert "test_history" in estimate_result.additional_information
    test_history = estimate_result.additional_information["test_history"]
    assert len(test_history) * test_every_n_steps == pytest.approx(n_steps, abs=1.01)
    assert isinstance(test_history[-1][0], int)


def test_critic_saved(n_points: int = 100) -> None:
    """Tests whether the critic is saved after estimation."""
    estimator = mine.MINEEstimator(batch_size=16, learning_rate=0.15, max_n_steps=100)
    sampler = samplers.SplitMultinormal(dim_x=2, dim_y=1, covariance=np.eye(3))
    xs, ys = sampler.sample(n_points=n_points, rng=0)

    estimator.estimate(xs, ys)
    assert isinstance(estimator.trained_critic, eqx.Module)
