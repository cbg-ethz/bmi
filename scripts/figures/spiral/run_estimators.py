import argparse
import time
import warnings
from pathlib import Path

import bmi.api as bmi


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("TASKS", type=Path, help="Directory with the tasks.")
    parser.add_argument("RESULTS", type=Path, help="Directory where the results should be dumped.")
    return parser


def run_estimator_on_task_seed(
    estimator_id: str,
    estimator: bmi.IMutualInformationPointEstimator,
    task: bmi.benchmark.Task,
    seed: int,
) -> bmi.RunResult:
    x, y = task[seed]
    t0 = time.time()
    mi_estimate = estimator.estimate(x, y)
    t1 = time.time()

    return bmi.RunResult(
        task_id=task.task_id,
        estimator_id=estimator_id,
        seed=seed,
        mi_estimate=mi_estimate,
        time_in_seconds=t1 - t0,
    )


ESTIMATORS = {
    "KSG: 10": bmi.estimators.KSGEnsembleFirstEstimator(neighborhoods=(10,)),
    "KSG: 5": bmi.estimators.KSGEnsembleFirstEstimator(neighborhoods=(5,)),
    "Histograms: 5, 7": bmi.estimators.HistogramEstimator(n_bins_x=5, n_bins_y=7),
    "Histograms: 10, 10": bmi.estimators.HistogramEstimator(n_bins_x=10, n_bins_y=10),
}


def main() -> None:
    args = create_parser().parse_args()
    task_directory = args.TASKS
    results_directory = args.RESULTS

    results_directory.mkdir(exist_ok=True)

    for task_path in task_directory.iterdir():
        task = bmi.benchmark.Task.load(task_path)

        print(f"Running estimators on {task.task_id}...")
        for seed in task.keys():
            print(f"    Seed {seed}...")
            for estimator_id, estimator in ESTIMATORS.items():
                print(f"        Running estimator {estimator_id}...")
                try:
                    result = run_estimator_on_task_seed(
                        estimator_id=estimator_id,
                        estimator=estimator,
                        task=task,
                        seed=seed,
                    )

                    path = results_directory / f"{estimator_id}-{task.task_id}-{seed}.json"

                    with open(path, "w") as f:
                        f.write(result.json())

                except Exception as e:
                    warnings.warn(
                        f"Running estimator {estimator_id} on task {task.task_id} "
                        f"with seed {seed} raised error {e}."
                    )

    print("Run finished.")


if __name__ == "__main__":
    main()
