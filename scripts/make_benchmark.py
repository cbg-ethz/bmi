import argparse
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd

import bmi.api as bmi


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("TASKS", type=Path, help="Directory where the tasks will be dumped to.")
    return parser


# List of tuples (estimator_id, estimator) without internal estimators.
# These will be converted to ITaskEstimators.
# Leave estimator_id = None to use the automated generation capabilities.
ESTIMATORS_INTERNAL: list[tuple[Optional[str], bmi.IMutualInformationPointEstimator]] = [
    ("default-KSG", bmi.estimators.KSGEnsembleFirstEstimator()),
    ("hist-naive-3-3", bmi.estimators.HistogramEstimator(n_bins_x=3)),
    ("hist-naive-5-5", bmi.estimators.HistogramEstimator(n_bins_x=5)),
    ("cca", bmi.estimators.CCAMutualInformationEstimator()),
]

# Other ITaskEstimators, e.g., these using R or Julia.
ESTIMATORS_EXTERNAL: list[bmi.ITaskEstimator] = []

# All the estimators we will use
ESTIMATORS: list[bmi.ITaskEstimator] = [
    bmi.WrappedEstimator(estimator=estimator, estimator_id=estimator_id)
    for estimator_id, estimator in ESTIMATORS_INTERNAL
] + ESTIMATORS_EXTERNAL


def main() -> None:
    args = create_parser().parse_args()
    tasks_dir = args.TASKS

    # Generating benchmark tasks
    print("Generating benchmark tasks...")
    bmi.benchmark.save_benchmark_tasks(
        tasks_dir=tasks_dir,
        exist_ok=True,
    )

    # Running the estimators
    results = []
    for task_path in tasks_dir.iterdir():
        seeds = bmi.TaskDirectory(task_path).seeds()
        print(f"Running estimators on task loaded from {task_path}.")
        print(f"Available seeds: {seeds}.")
        for seed in seeds:
            for estimator in ESTIMATORS:
                print(f"    Running estimator {estimator.estimator_id()}...")
                try:
                    result = estimator.estimate(task_path=task_path, seed=seed)
                    results.append(result)
                except Exception as e:
                    warnings.warn(
                        f"When running estimator {estimator.estimator_id()} on "
                        f"task {task_path} with seed {seed} an exception was raised: {e}."
                    )

    # Now do Pandas magic to have a nice table
    # TODO(Frederic): add column true MI!!!
    # TODO(Frederic, Pawel): idea we can have results.csv
    #  (detailed, not reduced) and results.html (pretty plots and means)
    results = pd.DataFrame(map(dict, results))
    results.to_csv("results.csv", index=False)
    interesting_cols = ["mi_estimate", "time_in_seconds"]

    means = (
        results.groupby(["task_id", "estimator_id"])[interesting_cols]
        .mean(numeric_only=True)
        .rename(columns=lambda x: x + "_mean")
    )

    stds = (
        results.groupby(["task_id", "estimator_id"])[interesting_cols]
        .std(numeric_only=True)
        .rename(columns=lambda x: x + "_std")
    )
    stats = means.join(stds)
    stats.to_csv("stats.csv")
    print(stats)


if __name__ == "__main__":
    main()
