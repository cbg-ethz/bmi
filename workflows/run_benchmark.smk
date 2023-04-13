# SnakeMake workflow used to generate benchmark results
import bmi.estimators.api as estimators
from bmi.benchmark import BENCHMARK_TASKS, run_estimator


# TODO: Add more estimators.
ESTIMATORS = {
    'MINE': estimators.MINEEstimator(max_n_steps=100, batch_size=32),
    'KSG-5': estimators.KSGEnsembleFirstEstimator(neighborhoods=(5,)),
    'KSG-10': estimators.KSGEnsembleFirstEstimator(neighborhoods=(10,)),
}

# TODO: Remove this restriction.
benchmark_tasks = dict(list(BENCHMARK_TASKS.items())[:3])

rule all:
    input:
        expand('benchmark/results/{estimator_id}/{task_id}/{n_samples}-{seed}.yaml', task_id=benchmark_tasks, n_samples=[100, 200], seed=[0, 1], estimator_id=ESTIMATORS)


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
        seed = int(output.yaml.split('.')[-2].split('-')[-1])

        estimator = ESTIMATORS[estimator_id]
        
        result = run_estimator(estimator=estimator, estimator_id=estimator_id, sample_path=input.csv, task_id=task_id, seed=seed)
        result.dump(output.yaml)

