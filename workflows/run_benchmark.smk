# SnakeMake workflow used to generate benchmark results
import resource

import bmi.estimators.api as estimators
from bmi.benchmark import BENCHMARK_TASKS, run_estimator


# === CONFIG ===

# TODO: Add more estimators.
ESTIMATORS = {
#    'MINE': estimators.MINEEstimator(max_n_steps=100, batch_size=32),
    'KSG-5': estimators.KSGEnsembleFirstEstimator(neighborhoods=(5,)),
    'Hist': estimators.HistogramEstimator(),
#    'KSG-10': estimators.KSGEnsembleFirstEstimator(neighborhoods=(10,)),
}

# TODO: Remove this restriction.
TASKS = dict(list(BENCHMARK_TASKS.items())[-3:-2])

N_SAMPLES = [10000] #[100, 200]

SEEDS = [0] #, 1]


# === RULES ===

rule all:
    input:
        expand(
            'benchmark/results/{estimator_id}/{task_id}/{n_samples}-{seed}.yaml',
            estimator_id=ESTIMATORS,
            task_id=TASKS,
            n_samples=N_SAMPLES,
            seed=SEEDS,
        )


# Sample task given a seed and number of samples.
rule sample_task:
    output: 'benchmark/tasks/{task_id}/{n_samples}-{seed}.csv'
    run:
        task_id = wildcards.task_id
        n_samples = int(wildcards.n_samples)
        seed = int(wildcards.seed)

        BENCHMARK_TASKS[task_id].save_sample(str(output), n_samples, seed)


# Apply estimator to sample
rule apply_estimator:
    output: 'benchmark/results/{estimator_id}/{task_id}/{n_samples}-{seed}.yaml'
    input: 'benchmark/tasks/{task_id}/{n_samples}-{seed}.csv'
    run:
        estimator_id = wildcards.estimator_id
        estimator = ESTIMATORS[estimator_id]
        task_id = wildcards.task_id
        seed = int(wildcards.seed)
        
        # this should be about ~4GiB
        resource.setrlimit(resource.RLIMIT_AS, (1<<33, 1<<33))

        result = run_estimator(estimator=estimator, estimator_id=estimator_id, sample_path=str(input), task_id=task_id, seed=seed)
        result.dump(str(output))

