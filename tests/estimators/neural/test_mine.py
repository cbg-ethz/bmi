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
        hidden_layers=(8, 5),
        verbose=False,
    )

    true_mi = distribution.mutual_information()
    estimate = estimator.estimate(points_x, points_y)
    assert estimate == pytest.approx(true_mi, abs=0.05, rel=0.1)
