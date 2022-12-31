from pathlib import Path
from time import time

import pandas as pd

import bmi.api as bmi

# TODO(frdrc): refractor into argparse
TASKS_DIR = Path(__file__).parent.parent / "tasks"


# generating benchmark tasks
print("generating benchmark tasks!")
bmi.benchmark.save_benchmark_tasks(
    tasks_dir=TASKS_DIR,
    exist_ok=True,
)


# try loading tasks
tasks = []
for task_path in TASKS_DIR.iterdir():
    task = bmi.benchmark.Task.load(task_path)
    tasks.append(task)


# select estimators
estimators = {
    "ksg-5-10": bmi.estimators.KSGEnsembleFirstEstimator(),
    "hist-naive-3-3": bmi.estimators.HistogramEstimator(n_bins_x=3),
    "hist-naive-5-5": bmi.estimators.HistogramEstimator(),  # TODO(frdrc): get better estimators
}


# TODO(frdrc): save and load results! run only when not found
def run_estimator_on_task(
    estimator_id: str,
    estimator: bmi.IMutualInformationPointEstimator,
    task: bmi.benchmark.Task,
):
    results = []
    for seed, (samples_x, samples_y) in task:
        t0 = time()
        mi_estimate = estimator.estimate(samples_x, samples_y)
        t1 = time()

        results.append(
            bmi.benchmark.RunResult(
                task_id=task.task_id,
                estimator_id=estimator_id,
                seed=seed,
                mi_estimate=mi_estimate,
                time_in_seconds=t1 - t0,
            )
        )
    return results


print("running benchmark!")
results = []
for task in tasks:
    print(f"running task '{task.task_id}'")
    for estimator_id, estimator in estimators.items():
        print(f"running estimator '{estimator_id}'")
        results += run_estimator_on_task(estimator_id, estimator, task)


# quick stats
# idea: we can have results.csv (detailed, not reduced) and results.html (pretty plots and means)

# TODO(frdrc): add column true MI!!!
results = pd.DataFrame(map(dict, results))
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
print(stats)
