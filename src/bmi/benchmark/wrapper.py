"""Allows running an external estimator."""
import subprocess
from typing import Optional, Sequence

from bmi.benchmark._serialize import TaskDirectory
from bmi.benchmark.core import RunResult, TaskMetadata
from bmi.benchmark.timer import Timer
from bmi.interface import Pathlike


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
    task_path: Pathlike,
    seed: int,
    command_args: Sequence[str],
    estimator_id: str,
    estimator_params: Optional[dict] = None,
    additional_args: Sequence[str] = (),
) -> RunResult:
    """Runs an external estimator via a command line.

    The command is run as:

    Args:
        task_path: path to the directory with a given task
        seed: random seed to select the right samples
        command_args: arguments necessary to run the estimator
          (without the parameters dependent on the task,
          i.e., the samples CSV, seed, and dimensions)
        estimator_id: the unique identifier of the estimator (and its parameters)
        estimator_params: dictionary with parameters of the estimator, to be added to the RunResult
        additional_args: method hyperparameters added after the task-dependent parameters

    Note:
        The estimator should take the following CLI arguments:

        COMMAND ARG_1 ... ARG_N  SAMPLES_CSV SEED DIM_X DIM_Y  ADDITIONAL_ARG1 ADDITIONAL_ARG2
        |______________________| |__________________________|  |______________________________|
              command_args          task-dependent params               additional_args
                                    added by this command

        and *print a single float* to the standard output.

        For example, an estimator `some_estimator.sh` which takes
        one hyperparameter (in this case 1524)
        can be run on task1/samples.csv with seed 42
        and with X dimension 5 and Y dimension 2 by:
        some_estimator.sh 1524 task1/samples.csv 42 5 2
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
    mi_estimate = _run_command_and_read_mi(list(command_args) + our_args + list(additional_args))

    return RunResult(
        task_id=metadata.task_id,
        seed=seed,
        estimator_id=estimator_id,
        mi_estimate=mi_estimate,
        time_in_seconds=timer.check(),
        estimator_params=estimator_params,
    )
