import bmi.benchmark.traverse as tr
from bmi.interface import RunResult


def test_save_load_run_result(tmp_path) -> None:
    directory = tmp_path / "something"
    directory.mkdir()
    path = directory / "test-results.yaml"

    result = RunResult(
        task_id="test-task",
        seed=4,
        estimator_id="test-estimator",
        mi_estimate=0.5,
    )
    tr.SaveLoadRunResults.dump(result=result, path=path)

    result_ = tr.SaveLoadRunResults.from_path(path)

    assert result == result_
