import jax.random
import numpy as np
import pytest

import bmi.estimators.neural._mine as mine
import bmi.estimators.neural._nn as nn
import bmi.samplers.api as samplers


def test_estimator_estimates(n_points: int = 5_000) -> None:
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

    estimated_mi = mine.training_loop(
        key_train,
        critic=nn.MLP(dim_x=2, dim_y=1, hidden_layers=(5, 5), key=key_mlp),
        xs=xs,
        ys=ys,
        max_n_steps=3_000,
        batch_size=256,
        learning_rate=0.1,
    )

    true_mi = sampler.mutual_information()

    assert estimated_mi == pytest.approx(true_mi, abs=0.05, rel=0.01)
