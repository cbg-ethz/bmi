import argparse
from pathlib import Path

import bmi.api as bmi


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("TASKS", type=Path, help="Directory with the tasks.")
    parser.add_argument("RESULTS", type=Path, help="Directory where the results should be dumped.")
    return parser


ESTIMATORS: dict[str, str] = {
    "KSG": "--estimator KSG --neighbors 5",
    "LNN": "--estimator R-LNN --neighbors 5 --truncation 15",
    "Histogram-3": "--estimator HISTOGRAM --bins-x 3",
    "Histogram-5": "--estimator HISTOGRAM --bins-x 5",
    "CCA": "--estimator CCA",
}


def main() -> None:
    args = create_parser().parse_args()
    task_directory = args.TASKS
    results_directory = args.RESULTS

    results_directory.mkdir(exist_ok=True)

    for task_path in task_directory.iterdir():
        task = bmi.benchmark.Task.load(task_path)
        for seed in task.keys():
            for estimator_id, estimator_args in ESTIMATORS.items():
                # We hash task ID as it may contain spaces or other special characters, which
                # are not bash-friendly
                task_id_hash = str(hash(task.task_id))

                output_path = results_directory / f"{estimator_id}-{task_id_hash}-{seed}.yaml"
                command = (
                    f"python scripts/run_estimator.py "
                    f"--TASK {str(task_path)} "
                    f"--SEED {seed} "
                    f"--OUTPUT {output_path} "
                    f"--estimator-id {estimator_id} "
                    f"{estimator_args}"
                )
                print(command)


if __name__ == "__main__":
    main()
