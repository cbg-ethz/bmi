import numpy as np

import bmi.benchmark.core as core
from bmi.samplers.api import SplitMultinormal


def test_save_load(tmp_path) -> None:
    target_path = tmp_path / "my-test-task"

    sampler = SplitMultinormal(dim_x=1, dim_y=1, covariance=np.asarray([[1, 0.7], [0.7, 1]]))

    task = core.generate_task(sampler=sampler, n_samples=5, seeds=[1, 2], task_id="some-task-id")
    task.save(target_path, exist_ok=False)

    new_task = core.Task.load(target_path)

    assert new_task.metadata == task.metadata
    assert new_task.keys() == {1, 2}

    assert task == new_task
