import numpy as np
import pytest

import bmi.benchmark.tasks.bimodal_gaussians as bimodal_gaussians


@pytest.mark.parametrize("n_samples", [1_000])
@pytest.mark.parametrize("seed", [0])
def test_numerical_inversion(n_samples, seed):
    task = bimodal_gaussians.task_bimodal_gaussians()

    samples_x, samples_y = task.sample(n_samples=n_samples, seed=seed)

    print(samples_x.min())
    print(samples_x.max())

    # there should be almost no repeats
    assert np.unique(samples_x).shape[0] > 0.99 * n_samples
    assert np.unique(samples_y).shape[0] > 0.99 * n_samples
