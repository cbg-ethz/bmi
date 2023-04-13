# SnakeMake workflow used to generate benchmark results
import bmi.estimators.api as estimators
from bmi.benchmark import BENCHMARK_TASKS, run_estimator, RunResult

class FailingEstimator:
    def estimate(self, x, y):
        raise ValueError

    def parameters(self):
        raise ValueError

# TODO: Add more estimators.
ESTIMATORS = {
    # 'MINE-test': estimators.MINEEstimator(max_n_steps=100, batch_size=32),
    'KSG-5': estimators.KSGEnsembleFirstEstimator(neighborhoods=(5,)),
    # 'KSG-10': estimators.KSGEnsembleFirstEstimator(neighborhoods=(10,)),
    "Failing": FailingEstimator(),
}

# TODO: Remove this restriction.
benchmark_tasks = dict(list(BENCHMARK_TASKS.items())[:2])

rule all:
    input:
        expand('benchmark/results/{estimator_id}/{task_id}/{n_samples}-{seed}.yaml', task_id=benchmark_tasks, n_samples=[100], seed=[0], estimator_id=ESTIMATORS)


rule sample_task:
    # This rule samples a task for a given seed and number of samples
    output:
        file = 'benchmark/tasks/{task_id}/{n_samples}-{seed}.csv'
    run:
        task_id = output.file.split("/")[2]
        n_samples_seed = output.file.split('/')[3]
        n_samples, seed = n_samples_seed.split('-')
        n_samples = int(n_samples)
        seed = int(seed.split(".")[0])
        BENCHMARK_TASKS[task_id].save_sample(output.file, n_samples, seed)


rule apply_estimator:
    # This rule applies an estimator to a given task with number of samples and seed provided
    output:
        yaml = 'benchmark/results/{estimator_id}/{task_id}/{n_samples}-{seed}.yaml'
    input:
        csv = 'benchmark/tasks/{task_id}/{n_samples}-{seed}.csv'
    run:
        estimator_id = output.yaml.split('/')[2]
        task_id = output.yaml.split('/')[3]

        n_samples_seed = input.csv.split('/')[3]
        n_samples, seed = n_samples_seed.split('-')
        n_samples = int(n_samples)
        seed = int(seed.split(".")[0])

        print(f"{seed} {n_samples} {estimator_id} {task_id}")

        # Save dummy file
        dummy_result = RunResult(mi_estimate=float("nan"), time_in_seconds=float("nan"), estimator_id=estimator_id, task_id=task_id, seed=seed, n_samples=n_samples)
        dummy_result.dump(output.yaml)

        # Try to overwrite with a true estimator
        try:
            estimator = ESTIMATORS[estimator_id]
            result = run_estimator(estimator=estimator, estimator_id=estimator_id, sample_path=input.csv, task_id=task_id, seed=seed)
            result.dump(output.yaml)
        except Exception as e:
            print(f"CAUGHT EXCEPTION: Estimator {estimator_id} on task {task_id} with samples {n_samples} and seed {seed} raised exception {e}")
