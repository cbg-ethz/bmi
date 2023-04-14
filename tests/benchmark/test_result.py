import pytest
import yaml

import bmi.benchmark.result as result
from bmi.benchmark.utils.dict_dumper import DictDumper


def test_save_load_run_result(tmp_path) -> None:
    target_path = tmp_path / "run_result.yaml"

    run_result = result.RunResult(
        mi_estimate=1.5241,
        time_in_seconds=70.3412,
        success=True,
        estimator_id="test-estimator",
        task_id="test-task",
        n_samples=5000,
        seed=42,
        additional_information={
            "covariance": 1.8,
            "test": True,
        },
    )

    with open(target_path, "w") as f:
        yaml.dump(run_result.dict(), f, Dumper=DictDumper)

    with open(target_path) as f:
        run_result_load = result.RunResult(**yaml.load(f, Loader=yaml.SafeLoader))

    assert run_result == run_result_load


@pytest.mark.skip("Not implemented!")
def test_run_estimator(self) -> None:
    """Checks if an run_estimator generates RunResult succesfully."""
    raise NotImplementedError


@pytest.mark.skip("Not implemented!")
def test_run_estimator_exceptions(self) -> None:
    """Checks if an run_estimator handles exceptions correctly."""
    raise NotImplementedError
