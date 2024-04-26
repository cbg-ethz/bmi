import joblib
import resource
import pandas as pd

import numpy as np
import jax.numpy as jnp

from collections import defaultdict

from bmi.estimators.external.gmm import GMMEstimator
import bmi.benchmark.tasks as tasks
import bmi.benchmark.tasks.mixtures as mixtures
import bmi.benchmark.tasks.multinormal as multinormal
import bmi.benchmark.tasks.student as student
from bmi.benchmark.tasks.asinh import transform_asinh_task as asinh


from bmi.utils import read_sample


TASKS = [
    tasks.task_additive_noise(epsilon=0.75),
    mixtures.task_x(),
    mixtures.task_ai(),
    mixtures.task_galaxy(),
    # Concentric
    mixtures.task_concentric_multinormal(dim_x=3, n_components=5),
    mixtures.task_concentric_multinormal(dim_x=5, n_components=5),
    mixtures.task_concentric_multinormal(dim_x=3, n_components=10),
    mixtures.task_concentric_multinormal(dim_x=5, n_components=10),
    # Inliers
    mixtures.task_multinormal_sparse_w_inliers(dim_x=5, dim_y=5, inlier_fraction=0.2),
    mixtures.task_multinormal_sparse_w_inliers(dim_x=5, dim_y=5, inlier_fraction=0.5),
    # Multivariate normal
    multinormal.task_multinormal_dense(5, 5),
    multinormal.task_multinormal_2pair(5, 5),
    # Student
    asinh(student.task_student_identity(dim_x=1, dim_y=1, df=1)),
    asinh(student.task_student_identity(dim_x=2, dim_y=2, df=1)),
    asinh(student.task_student_identity(dim_x=3, dim_y=3, df=2)),
    asinh(student.task_student_identity(dim_x=5, dim_y=5, df=2)),
]


NAMES = {
    # one-dimensional
    '1v1-additive-0.75': "Additive",
    '1v1-AI': "AI",
    '1v1-X-0.9': "X",
    '2v1-galaxy-0.5-3.0': "Galaxy",
    # Concentric
    '3v1-concentric_gaussians-10': "Concentric (3-dim, 10)",
    '3v1-concentric_gaussians-5': "Concentric (3-dim, 5)",
    '5v1-concentric_gaussians-10': "Concentric (5-dim, 10)",
    '5v1-concentric_gaussians-5': "Concentric (5-dim, 5)",
    # Inliers
    'mult-sparse-w-inliers-5-5-2-2.0-0.2': "Inliers (5-dim, 0.2)",
    'mult-sparse-w-inliers-5-5-2-2.0-0.5': "Inliers (5-dim, 0.5)",
    # Multivariate normal
    'multinormal-dense-5-5-0.5': "Normal (5-dim, dense)",
    'multinormal-sparse-5-5-2-2.0': "Normal (5-dim, sparse)",
    # Student
    'asinh-student-identity-1-1-1': "Student (1-dim)",
    'asinh-student-identity-2-2-1': "Student (2-dim)",
    'asinh-student-identity-3-3-2': "Student (3-dim)",
    'asinh-student-identity-5-5-2': "Student (5-dim)",
}


TASKS_DICT = {
    task.id: task for task in TASKS
}
SEEDS = [0, 1, 2]

N_SAMPLES: int = 5_000

# We only allow GMMEstimator
ESTIMATORS_DICT = {
    "GMM": GMMEstimator(),
}

workdir: "generated/projects/Mixtures/gmm_benchmark"

rule all:
    input: ["results.csv", "bmm_table.txt"]


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


rule create_table:
    output: 'bmm_table.txt'
    input: 'results.csv'
    run:
        df = pd.read_csv(str(input), index_col=None)
        print(df)

        def create_result_entry(mean, qlow, qhigh):
            return f"{mean:.2f} ({qlow:.2f}–{qhigh:.2f})"

        def create_row():
            return {
                "Task name": "",
                "True MI": -10,
                "Run 1": "mean (qlow – qhigh)",
                "Run 2": "mean (qlow – qhigh)",
                "Run 3": "mean (qlow – qhigh)",
            }

        # task_id -> row in the table
        table_map = defaultdict(create_row)

        for _, record in df.iterrows():
            task_id = record["task_id"]
            result_entry = create_result_entry(record["mi_mean"], record["mi_q_low"], record["mi_q_high"])
            mi_true = record["mi_true"]
            seed = record["seed"]

            row = table_map[task_id]
            row["Task name"] = NAMES[task_id]
            row["True MI"] = f"{mi_true:.2f}"
            row[f"Run {seed + 1}"] = result_entry

        table = pd.DataFrame(table_map.values())
        table = table.set_index("Task name")

        with open(str(output), "w") as fh:
            fh.write(table.to_latex(escape=False))
