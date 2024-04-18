import joblib
import resource
import pandas as pd

import numpy as np
import jax.numpy as jnp

from bmi.estimators.external.gmm import GMMEstimator
import bmi.benchmark.tasks as tasks
import bmi.benchmark.tasks.mixtures as mixtures

from bmi.utils import read_sample


TASKS = [
    tasks.task_additive_noise(epsilon=0.75),
    mixtures.task_x(),
    mixtures.task_ai(),
    mixtures.task_galaxy(),
    mixtures.task_concentric_multinormal(dim_x=5, n_components=5),
    mixtures.task_multinormal_sparse_w_inliers(dim_x=5, dim_y=5, inlier_fraction=0.2),
]

TASKS_DICT = {
    task.id: task for task in TASKS
}
SEEDS = [0]

N_SAMPLES: int = 5_000

# We only allow GMMEstimator
ESTIMATORS_DICT = {
    "GMM": GMMEstimator(),
}

workdir: "generated/projects/Mixtures/gmm_benchmark"

rule all:
    input: "results.csv"


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
rule apply_gmm_estimator:
    output: 'results/{estimator_id}/{task_id}/{n_samples}-{seed}.joblib'
    input: 'tasks/{task_id}/{n_samples}-{seed}.csv'
    run:
        estimator_id = wildcards.estimator_id
        estimator = ESTIMATORS_DICT[estimator_id]
        task_id = wildcards.task_id
        seed = int(wildcards.seed)
        
        # this should be about ~4GiB
        resource.setrlimit(resource.RLIMIT_DATA, (1<<33, 1<<33))

        x, y = read_sample(str(input))

        posterior_mi = estimator.get_posterior_mi(x, y)
        mcmc_samples = estimator._mcmc.get_samples()

        result = {
            "posterior_mi": posterior_mi,
            "mcmc_samples": mcmc_samples,
            "mi": posterior_mi.mean(),
            "task_id": task_id,
            "estimator_id": estimator_id,
            "seed": seed,
        }

        joblib.dump(result, str(output))


# Gather all results into one CSV file
rule results:
    output: 'results.csv'
    input:
        expand(
            'results/{estimator_id}/{task_id}/{n_samples}-{seed}.joblib',
            estimator_id=ESTIMATORS_DICT,
            task_id=TASKS_DICT,
            n_samples=N_SAMPLES,
            seed=SEEDS,
        )
    run:
        results = []
        for result_path in input:
            run_info = joblib.load(result_path)
            mi_samples = np.asarray(run_info["posterior_mi"])

            original_length = mi_samples.shape[0]
            mi_samples = mi_samples[np.isfinite(mi_samples)]
            new_length = mi_samples.shape[0]

            if new_length > 0:
                q_low = np.quantile(mi_samples, q=0.1)
                mean = np.mean(mi_samples)
                q_high = np.quantile(mi_samples, q=0.9)
            else:
                q_low = -1
                mean = -1
                q_high = -1


            task = TASKS_DICT[run_info['task_id']]
            results.append({
                "mi_true": task.mutual_information,
                "task_id": task.id,
                "estimator_id": run_info["estimator_id"],
                "seed": run_info["seed"],
                "task_name": task.name,
                "task_params": task.params,
                "entries_filtered_out": original_length - new_length,
                "used_length": new_length,
                "mi_q_low": float(q_low),
                "mi_mean": float(mean),
                "mi_q_high": float(q_high),
            })
    
        pd.DataFrame(results).to_csv(str(output), index=False)
