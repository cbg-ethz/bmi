import numpy as np
import pytest

import bmi.benchmark.tasks.scaling as sc
from bmi.benchmark.task_list import EMBEDDINGS_TASKS


@pytest.fixture
def generic_task():
    return EMBEDDINGS_TASKS[0]


@pytest.mark.parametrize("n_points", (500,))
def test_transform_rescale(n_points: int, generic_task) -> None:
    new_task = sc.transform_rescale(generic_task)

    samples_x, samples_y = new_task.sample(n_points, seed=4)
    assert samples_x.shape == (n_points, generic_task.dim_x)
    assert samples_y.shape == (n_points, generic_task.dim_y)
    assert "rescale" in new_task.id
    assert "Rescale" in new_task.name

    samples_xy = np.hstack([samples_x, samples_y])

    for dim in range(samples_xy.shape[1]):
        assert np.mean(samples_xy[:, dim]) == pytest.approx(0.0, abs=0.03)
        assert np.std(samples_xy[:, dim]) == pytest.approx(1.0, abs=0.03)


@pytest.mark.parametrize("n_points", (1000,))
@pytest.mark.parametrize("n_quantiles", (100, 1500))
def test_uniformise(n_points: int, generic_task, n_quantiles: int) -> None:
    new_task = sc.transform_uniformise(generic_task, n_quantiles=n_quantiles)

    samples_x, samples_y = new_task.sample(n_points, seed=42)
    assert samples_x.shape == (n_points, generic_task.dim_x)
    assert samples_y.shape == (n_points, generic_task.dim_y)
    assert "uniformise" in new_task.id
    assert "Uniformise" in new_task.name
    assert new_task.params.get("n_quantiles") == n_quantiles

    samples_xy = np.hstack([samples_x, samples_y])

    for dim in range(samples_xy.shape[1]):
        hist, _ = np.histogram(samples_xy[:, dim], bins=10, density=True)
        for bin_value in hist:
            assert bin_value == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("n_points", (1000,))
def test_gaussianise(n_points: int, generic_task, n_quantiles: int = 100) -> None:
    new_task = sc.transform_gaussianise(generic_task, n_quantiles=n_quantiles)

    samples_x, samples_y = new_task.sample(n_points, seed=42)
    assert samples_x.shape == (n_points, generic_task.dim_x)
    assert samples_y.shape == (n_points, generic_task.dim_y)
    assert "gaussianise" in new_task.id
    assert "Gaussianise" in new_task.name

    assert new_task.params.get("n_quantiles") == n_quantiles

    samples_xy = np.hstack([samples_x, samples_y])
    for dim in range(samples_xy.shape[1]):
        assert np.mean(samples_xy[:, dim]) == pytest.approx(0.0, abs=0.05)
        assert np.std(samples_xy[:, dim]) == pytest.approx(1.0, abs=0.05)
        # TODO(Pawel): Think about more elaborate checks for Gaussianity.
