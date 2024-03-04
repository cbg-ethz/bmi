# Main workflow for running the benchmark.

import resource
import yaml
import pandas as pd

from bmi.benchmark import run_estimator
import _utils as utils


TASKS_DICT = {
    task.id: task for task in TASKS
}


rule benchmark_table:
    input: 'results.csv'
    output: 'benchmark.html'
    run:
        results = utils.read_results(str(input))
        table = utils.create_benchmark_table(results)
        table.to_html(str(output))


# Gather all results into one CSV file
rule results:
    output: 'results.csv'
    input:
        expand(
            'results/{estimator_id}/{task_id}/{n_samples}-{seed}.yaml',
            estimator_id=ESTIMATORS_DICT,
            task_id=TASKS_DICT,
            n_samples=N_SAMPLES,
            seed=SEEDS,
        )
    run:
        results = []
        for result_path in input:
            with open(result_path) as f:
                result = yaml.load(f, yaml.SafeLoader)
                task = TASKS_DICT[result['task_id']]
                result['mi_true'] = task.mutual_information
                result['task_name'] = task.name
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
        task = TASKS_DICT[task_id]
        task.save_sample(str(output), n_samples, seed)


# Apply estimator to sample
rule apply_estimator:
    output: 'results/{estimator_id}/{task_id}/{n_samples}-{seed}.yaml'
    input: 'tasks/{task_id}/{n_samples}-{seed}.csv'
    run:
        estimator_id = wildcards.estimator_id
        estimator = ESTIMATORS_DICT[estimator_id]
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
