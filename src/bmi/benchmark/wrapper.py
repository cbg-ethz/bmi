import subprocess
from typing import Sequence

from bmi.benchmark._serialize import TaskDirectory
from bmi.benchmark.core import RunResult, TaskMetadata
from bmi.benchmark.timer import Timer


def _run_command_and_read_mi(args: list[str]) -> float:
    output = subprocess.check_output(args)

    try:
        mi_estimate = float(output.decode().strip())
    except Exception as e:
        raise ValueError(
            f"Failed to cast output to float, got output '{output.decode()}' and error '{e}'."
        )

    return mi_estimate


def run_external_estimator(
    task_path, seed: int, command_args: Sequence[str], estimator_id: str
) -> RunResult:
    """TODO(Pawel): Add the docstring.

    Args:
        task_path: the directory
    """
    task_directory = TaskDirectory(task_path)
    metadata = TaskMetadata(**task_directory.load_metadata())

    our_args = [
        str(task_directory.samples),
        str(seed),
        str(metadata.dim_x),
        str(metadata.dim_y),
    ]

    timer = Timer()
    mi_estimate = _run_command_and_read_mi(list(command_args) + our_args)

    # TODO(Pawel): Add handling params and measure time
    return RunResult(
        task_id=metadata.task_id,
        estimator_id=estimator_id,
        mi_estimate=mi_estimate,
        seed=seed,
        time_in_seconds=timer.check(),
    )


"""
$ julia my_estimator.jl samples.csv 3 [dim_x] [dim_y]
0.12
"""
