"""Tests of the neural estimators on simple distributions."""
import numpy as np
import pytest

import bmi.estimators.neural.api as neural
import bmi.samplers.api as samplers


def test_estimator(n_points: int = 400) -> None:
    estimator = neural.DonskerVaradhanEstimator(batch_size=100, hidden_layers=(5, 5))

    cov = np.asarray(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.3],
            [0.5, 0.3, 1.0],
        ]
    )

    sampler = samplers.SplitMultinormal(dim_x=2, dim_y=1, covariance=cov)
    xs, ys = sampler.sample(n_points=n_points, rng=0)

    estimated_mi = estimator.estimate(xs, ys)
    true_mi = sampler.mutual_information()

    assert estimated_mi == pytest.approx(true_mi)
