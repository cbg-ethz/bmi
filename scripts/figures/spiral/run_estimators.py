import argparse
from pathlib import Path

import bmi.api as bmi


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("TASKS", type=Path, help="Directory with the tasks.")
    parser.add_argument("RESULTS", type=Path, help="Directory where the results should be dumped.")
    return parser


ESTIMATORS = [
    "KSG-10",
    "KSG-5",
    "R-KSG-10",
    "R-KSG-5",
    "R-LNN-10",
    "R-LNN-5",
    # "MINE",
    "Histogram-3",
    "Histogram-5",
    "CCA",
]


def main() -> None:
    args = create_parser().parse_args()
    task_directory = args.TASKS
    results_directory = args.RESULTS

    results_directory.mkdir(exist_ok=True)

    for task_path in task_directory.iterdir():
        task = bmi.benchmark.Task.load(task_path)
        for seed in task.keys():
            for estimator in ESTIMATORS:
                # We hash task ID as it may contain spaces or other special characters, which
                # are not bash-friendly
                task_id_hash = str(hash(task.task_id))

                output_path = results_directory / f"{estimator}-{task_id_hash}-{seed}.yaml"
                command = (
                    f"python scripts/run_estimator.py "
                    f"--task {str(task_path)} "
                    f"--estimator {estimator} "
                    f"--seed {seed} "
                    f"--output {output_path}"
                )
                print(command)


if __name__ == "__main__":
    main()
