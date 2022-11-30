import numpy as np
import pytest

import bmi.benchmark.core as core
from bmi.samplers.api import SplitMultinormal


def test_task_save_load(tmp_path) -> None:
    """Tests if saving and loading the data to the disk doesn't change anything."""
    target_path = tmp_path / "my-test-task"

    sampler = SplitMultinormal(dim_x=1, dim_y=1, covariance=np.asarray([[1, 0.7], [0.7, 1]]))

    task = core.generate_task(sampler=sampler, n_samples=5, seeds=[1, 2], task_id="some-task-id")
    task.save(target_path, exist_ok=False)

    new_task = core.Task.load(target_path)

    assert new_task.metadata == task.metadata
    assert new_task.keys() == {1, 2}

    # TODO(Pawel): This assumes that __eq__ works. Check manually.
    assert task == new_task


class LoadTaskInWrongFormat:
    """Tests for loading a corrupted Task."""

    @pytest.mark.skip("Not implemented!")
    def test_dataframe_in_wrong_format(self) -> None:
        """Checks if an informative error is raised when
        a data frame in wrong format is passed."""
        raise NotImplementedError

    @pytest.mark.skip("Not implemented!")
    def test_metadata_and_samples_match(self) -> None:
        """Checks if an informative error is raised when
        there is a mismatch between the metadata and the samples."""
        raise NotImplementedError
