"""This script is used to generate the experimental design from a YAML file of the format:

estimator_id: "--argument 1 --argument2 ..."
"""
import argparse
from pathlib import Path

import yaml

import bmi.api as bmi


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--TASKS", type=Path, help="Directory with the tasks.")
    parser.add_argument(
        "--ESTIMATORS", type=Path, help="YAML file with a description of the estimators to run."
    )
    parser.add_argument(
        "--RESULTS", type=Path, help="Directory where the results should be dumped.", default=None
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Instead of printing experimental design, print a summary.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Add the `--dry-run` flag to all the commands. "
        "It it useful for catching errors within the experimental design.",
    )
    return parser


def get_estimators(args) -> dict[str, str]:
    """Parses the YAML with estimator specification.

    Example output:
    {
        "KSG": "--estimator KSG --neighbors 5",
        "LNN": "--estimator R-LNN --neighbors 5 --truncation 15",
        "Histogram-3": "--estimator HISTOGRAM --bins-x 3",
        "Histogram-5": "--estimator HISTOGRAM --bins-x 5",
        "CCA": "--estimator CCA",
    }
    """
    path = args.ESTIMATORS
    with open(path) as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def summary(args) -> None:
    """Prints a basic summary of the experimental design."""
    task_directory = args.TASKS
    estimators = get_estimators(args)

    n_runs: int = 0
    task_ids: list[str] = []

    for task_path in task_directory.iterdir():
        task = bmi.benchmark.Task.load(task_path)
        task_ids.append(task.task_id)
        n_runs += len(task.keys()) * len(estimators.keys())

    print(f"Total number of runs: {n_runs}")
    print(f"Total number of tasks: {len(task_ids)}")
    for task_id in task_ids:
        print(f"  {task_id}")
    print("Estimators (ID and additional arguments):")
    for estimator_id, estimator_args in estimators.items():
        print(f"  {estimator_id}: {estimator_args}")


def print_experimental_design(args) -> None:
    """Prints the experimental design,
    so it can be piped to GNU Parallel or run sequentially."""
    task_directory = args.TASKS
    estimators = get_estimators(args)

    if args.RESULTS is None:
        raise ValueError(
            "RESULTS directory must be specified unless the `--summary` flag is used."
        )
    results_directory = args.RESULTS
    results_directory.mkdir(exist_ok=True)

    for task_path in task_directory.iterdir():
        task = bmi.benchmark.Task.load(task_path)
        for seed in task.keys():
            for estimator_id, estimator_args in estimators.items():
                # We hash task ID as it may contain spaces or other special characters, which
                # are not bash-friendly
                task_id_hash = str(hash(task.task_id))
                output_path = results_directory / f"{estimator_id}-{task_id_hash}-{seed}.yaml"

                # Note the space after the flag, to have correct separation of flags in the command
                dry_run_flag = "--dry-run " if args.dry_run else ""

                command = (
                    f"python scripts/run_estimator.py {dry_run_flag}"
                    f"--TASK {task_path} "
                    f"--SEED {seed} "
                    f"--OUTPUT {output_path} "
                    f"--estimator-id {estimator_id} "
                    f"{estimator_args}"
                )
                print(command)


def main() -> None:
    args = create_parser().parse_args()
    # We want to take a look at the design summary
    if args.summary:
        summary(args)
    # We want to print the design, so it can be run
    else:
        print_experimental_design(args)


if __name__ == "__main__":
    main()
