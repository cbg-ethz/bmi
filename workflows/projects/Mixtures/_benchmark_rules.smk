# SnakeMake workflow used to generate results
# ASSUMES THAT THE FOLLOWING ARE DECLARED:
#   - ESTIMATORS: dict estimator_id -> estimator
#   - UNSCALED_TASKS: dict task_id -> task
#   - N_SAMPLES: list of ints
#   - SEEDS: list of ints

import resource
import yaml

import pandas as pd

import bmi
from bmi.benchmark import run_estimator
from bmi.benchmark.tasks import transform_rescale


def scale_tasks(tasks: dict[str, bmi.Task]) -> dict[str, bmi.Task]:
    """Auxiliary method used to rescale (whiten) each task in the list,
    without changing its name nor id."""
    return {
        key: transform_rescale(
            base_task=base_task,
            task_name=base_task.name,
            task_id=base_task.id,
        )
        for key, base_task in tasks.items()
    }

TASKS = scale_tasks(UNSCALED_TASKS)

rule results:
    output: 'results.csv'
    input:
        expand(
            'results/{estimator_id}/{task_id}/{n_samples}-{seed}.yaml',
            estimator_id=ESTIMATORS,
            task_id=TASKS,
            n_samples=N_SAMPLES,
            seed=SEEDS,
        )
    run:
        results = []
        for result_path in input:
            with open(result_path) as f:
                result = yaml.load(f, yaml.SafeLoader)
                task = TASKS[result['task_id']]
                result['mi_true'] = task.mutual_information
                result['task_params'] = task.params
                results.append(result)
        pd.DataFrame(results).to_csv(str(output), index=False)


# Sample task given a seed and number of samples.
rule sample_task:
    output: 'tasks/{task_id}/{n_samples}-{seed}.csv'
    run:
        task_id = wildcards.task_id
        n_samples = int(wildcards.n_samples)
        seed = int(wildcards.seed)
        task = TASKS[task_id]
        task.save_sample(str(output), n_samples, seed)


# Apply estimator to sample
rule apply_estimator:
    output: 'results/{estimator_id}/{task_id}/{n_samples}-{seed}.yaml'
    input: 'tasks/{task_id}/{n_samples}-{seed}.csv'
    run:
        estimator_id = wildcards.estimator_id
        estimator = ESTIMATORS[estimator_id]
        task_id = wildcards.task_id
        seed = int(wildcards.seed)
        
        # this should be about ~4GiB
        resource.setrlimit(resource.RLIMIT_DATA, (1<<33, 1<<33))

        result = run_estimator(
            estimator=estimator,
            estimator_id=estimator_id,
            sample_path=str(input),
            task_id=task_id,
            seed=seed,
        )
        result.dump(str(output))

