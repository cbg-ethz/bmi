import yaml

import bmi.benchmark.task as task
from bmi.benchmark.utils.dict_dumper import DictDumper


def test_save_load_task_metadata(tmp_path) -> None:
    target_path = tmp_path / "task_metadata.yaml"

    task_metadata = task.TaskMetadata(
        task_id="test-task",
        task_name="Test Task",
        dim_x=2,
        dim_y=3,
        mi_true=1.5,
        task_params={
            "covariance": 1.8,
            "test": True,
        },
    )

    with open(target_path, "w") as f:
        yaml.dump(task_metadata.dict(), f, Dumper=DictDumper)

    with open(target_path) as f:
        task_metadata_load = task.TaskMetadata(**yaml.load(f, Loader=yaml.SafeLoader))

    assert task_metadata == task_metadata_load
