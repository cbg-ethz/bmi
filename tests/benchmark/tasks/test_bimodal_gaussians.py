import pytest

import bmi.benchmark.tasks.bimodal_gaussians as bimodal_gaussians


@pytest.mark.parametrize("n_samples", [10_000])
@pytest.mark.parametrize("seed", [0])
def test_numerical_inversion(n_samples, seed):
    task = bimodal_gaussians.task_bimodal_gaussians()

    samples_x, samples_y = task.sample(n_samples=n_samples, seed=seed)

    print(samples_x.min())
    print(samples_x.max())

    # there should be almost no repeats
    assert len(set(map(float, samples_x))) > 0.999 * n_samples
    assert len(set(map(float, samples_y))) > 0.999 * n_samples
