import numpy as np
import pytest
from jax import random
from sklearn.feature_selection import mutual_info_regression

from bmi.samplers._additive_uniform import AdditiveUniformSampler


@pytest.mark.parametrize("epsilon", [0.5, 0.1, 2.0, 5.0])
@pytest.mark.parametrize("rng", [2, random.PRNGKey(32), np.int32(12)])
def test_uniform_additive(epsilon: float, rng, n_samples: int = 5_000) -> None:
    sampler = AdditiveUniformSampler(epsilon=epsilon)

    x, y = sampler.sample(n_points=n_samples, rng=rng)

    assert x.shape == (n_samples, 1)
    assert y.shape == (n_samples, 1)

    mi_estimate = mutual_info_regression(x, y.ravel(), random_state=0)[0]
    assert mi_estimate == pytest.approx(sampler.mutual_information(), rel=0.03, abs=0.05)
