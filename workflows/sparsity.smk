# Snakemake workflow for experiments with estimating mutual information
# for the transition from sparse to dense interactions between Gaussian variables
import resource

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

import bmi
from bmi.benchmark.tasks.multinormal import task_multinormal_sparse

# Number of seeds
N_SEED: int = 3


ESTIMATORS = {
    # TODO(Pawel): Add more estimators
    'KSG-5': bmi.estimators.KSGEnsembleFirstEstimator(neighborhoods=(5,)),
    'CCA': bmi.estimators.CCAMutualInformationEstimator(),
}
# Estimator names on the generated figure
ESTIMATOR_NAMES = {
    'KSG-5': "KSG",
    'CCA': "CCA",
}

assert tuple(sorted(ESTIMATOR_NAMES.keys())) == tuple(sorted(ESTIMATORS.keys())), "Estimator IDs don't agree."

def get_ks(dim_x: int, dim_y: int) -> list[int]:
    return list(range(1, min(dim_x, dim_y) + 1))


rule all:
    input:
        expand(
            'generated/sparsity/results_csv/{dim_x}-{dim_y}-{signal}-{noise}-{n_samples}.csv',
            dim_x=[25],
            dim_y=[25],
            signal=[0.8],
            noise=[0.1],
            n_samples=[1_000],
        )

rule plot:
    input: 'generated/sparsity/results_csv/{dim_x}-{dim_y}-{signal}-{noise}-{n_samples}.csv',
    output: 'generated/sparsity/figures/{dim_x}-{dim_y}-{signal}-{noise}-{n_samples}.pdf'
    run:
        pass

def get_inputs_to_assemble(wildcards) -> list[str]:
    """Constructs a list of files to be pulled to the data frame.

    Note: We want to have inputs of the format:
    'generated/sparsity/results/{estimator_id}/{dim_x}-{dim_y}-{signal}-{noise}-{n_samples}-{seed}-sparsity-{sparsity}.yaml'
    """
    lst = []
    dim_x = int(wildcards.dim_x)
    dim_y = int(wildcards.dim_y)
    signal = wildcards.signal
    noise = wildcards.noise
    n_samples = wildcards.n_samples

    for estimator_id in ESTIMATORS.keys():
        for seed in range(N_SEED):
            for sparsity in get_ks(dim_x=dim_x, dim_y=dim_y):
                name = f'generated/sparsity/results/{estimator_id}/{dim_x}-{dim_y}-{signal}-{noise}-{n_samples}-{seed}-sparsity-{sparsity}.yaml'
                lst.append(name)

    return lst


rule assemble_results:
    # Assembles results from YAML files into one CSV
    output: 'generated/sparsity/results_csv/{dim_x}-{dim_y}-{signal}-{noise}-{n_samples}.csv',
    input:
        get_inputs_to_assemble
    run:
        def flatten_dict(d, sep='_'):
            items = []
            for key, v in d.items():
                if isinstance(v,dict):
                    items.extend(flatten_dict(v, sep=sep).items())
                else:
                    items.append((key, v))
            return dict(items)

        entries = []
        for input_name in input:
            with open(input_name) as handler:
                result = yaml.safe_load(handler)
                result = flatten_dict(result)
                entries.append(result)

        pd.DataFrame(entries).to_csv(str(output), index=False)


rule apply_estimator:
    # Apply estimator to a given sample and save result
    output: 'generated/sparsity/results/{estimator_id}/{dim_x}-{dim_y}-{signal}-{noise}-{n_samples}-{seed}-sparsity-{sparsity}.yaml'
    input: 'generated/sparsity/samples/{dim_x}-{dim_y}-{signal}-{noise}-{sparsity}-{n_samples}-{seed}.csv'
    run:
        estimator_id = wildcards.estimator_id
        estimator = ESTIMATORS[estimator_id]

        # Construct the task to read off the metadata
        task = task_multinormal_sparse(
            dim_x=int(wildcards.dim_x),
            dim_y=int(wildcards.dim_y),
            n_interacting=int(wildcards.sparsity),
            correlation_signal=float(wildcards.signal),
            correlation_noise=float(wildcards.noise),
        )

        # this should be about ~4GiB
        resource.setrlimit(resource.RLIMIT_AS, (1 << 33, 1 << 33))

        task_info = task.params | {
            "dim_x": task.dim_x,
            "dim_y": task.dim_y,
            "task_seed": int(wildcards.seed),
            "mi_true": task.mutual_information,
        }

        result = bmi.benchmark.run_estimator(
            estimator=estimator,
            estimator_id=estimator_id,
            sample_path=str(input),
            task_id=str(input).split("/")[-1].split(".")[-2],
            seed=int(wildcards.seed),  # TODO(Pawel): Consider using different seed
            additional_information=task_info,
        )
        result.dump(str(output))


rule sample_task:
    # Sample a specified task and save to CSV
    output: 'generated/sparsity/samples/{dim_x}-{dim_y}-{signal}-{noise}-{sparsity}-{n_samples}-{seed}.csv'
    run:
        task = task_multinormal_sparse(
            dim_x=int(wildcards.dim_x),
            dim_y=int(wildcards.dim_y),
            n_interacting=int(wildcards.sparsity),
            correlation_signal=float(wildcards.signal),
            correlation_noise=float(wildcards.noise),
        )

        task.save_sample(
            str(output),
            n_samples=int(wildcards.n_samples),
            seed=int(wildcards.seed),
        )
