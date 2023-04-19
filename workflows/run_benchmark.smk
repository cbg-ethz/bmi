# SnakeMake workflow used to generate benchmark results
import resource
import yaml

# keep jax from using multiple threads on CPU
import os
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false "
    "intra_op_parallelism_threads=1 "
    "inter_op_parallelism_threads=1"
)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREAD"] = "1"

import pandas as pd

import bmi.estimators as estimators
import bmi.estimators.external.r_estimators as r_estimators
import bmi.estimators.external.julia_estimators as julia_estimators
from bmi.benchmark import BENCHMARK_TASKS, run_estimator


# === CONFIG ===

ESTIMATORS = {
    'MINE': estimators.MINEEstimator(verbose=False),
    'InfoNCE': estimators.InfoNCEEstimator(verbose=False),
    'NWJ': estimators.NWJEstimator(verbose=False),
    'Donsker-Varadhan': estimators.DonskerVaradhanEstimator(verbose=False),

    'KSG-5': estimators.KSGEnsembleFirstEstimator(neighborhoods=(5,)),
    'KSG-10': estimators.KSGEnsembleFirstEstimator(neighborhoods=(10,)),
    'Hist-10': estimators.HistogramEstimator(n_bins_x=10),

    'R-KSG-I-5': r_estimators.RKSGEstimator(variant=1, neighbors=5),
    'R-KSG-I-10': r_estimators.RKSGEstimator(variant=1, neighbors=10),
    'R-KSG-II-5': r_estimators.RKSGEstimator(variant=2, neighbors=5),
    'R-KSG-II-10': r_estimators.RKSGEstimator(variant=2, neighbors=10),
    'R-BNSL': r_estimators.RBNSLEstimator(),
    'R-LNN': r_estimators.RLNNEstimator(),

    'Julia-Hist-10': julia_estimators.JuliaHistogramEstimator(bins=10),
    'Julia-Kernel': julia_estimators.JuliaKernelEstimator(),
    'Julia-Transfer-30': julia_estimators.JuliaTransferEstimator(bins=30),
    #'Julia-KSG-I-5': julia_estimators.JuliaKSGEstimator(variant=1, neighbors=5),
}

TASKS = {
    task_id: BENCHMARK_TASKS[task_id]
    for task_id in BENCHMARK_TASKS.keys() & {
        '1v1-bimodal-0.75',
        'student-dense-1-1-5-0.75',
        'swissroll_x-1v1-normal-0.75',
        'multinormal-sparse-3-3-2-0.8-0.1',
        'multinormal-sparse-5-5-2-0.8-0.1',
    }
}

N_SAMPLES = [10000]

SEEDS = [0]


# === RULES ===

rule all:
    output: 'benchmark/results.csv'
    input:
        expand(
            'benchmark/results/{estimator_id}/{task_id}/{n_samples}-{seed}.yaml',
            estimator_id=ESTIMATORS,
            task_id=TASKS,
            n_samples=N_SAMPLES,
            seed=SEEDS,
        )
    run:
        results = []
        for result_path in input:
            with open(result_path) as f:
                results.append(yaml.load(f, yaml.SafeLoader))
        pd.DataFrame(results).to_csv(str(output), index=False)


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

