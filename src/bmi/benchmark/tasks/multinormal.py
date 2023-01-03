from typing import Optional

import numpy as np

import bmi.samplers.api as samplers
from bmi.benchmark.core import generate_task

SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


def task_mn_uniform(
    dim_x: int,
    dim_y: int,
    n_samples: int,
    seeds=SEEDS,
):
    sampler = samplers.SplitMultinormal(
        dim_x=dim_x,
        dim_y=dim_y,
        covariance=np.eye(dim_x + dim_y) * 0.5 + 0.5,
    )

    task = generate_task(
        sampler=sampler,
        n_samples=n_samples,
        seeds=seeds,
        task_id=f"mn-uniform-{dim_x}-{dim_y}-{n_samples}",
    )

    return task


def task_mn_sparse(
    dim_x: int,
    dim_y: int,
    n_samples: int,
    correlation_signal: float = 0.8,
    correlation_noise: float = 0.1,
    seeds=SEEDS,
    task_id: Optional[str] = None,
):
    covariance = samplers.parametrised_correlation_matrix(
        dim_x=dim_x,
        dim_y=dim_y,
        k=2,
        correlation=correlation_signal,
        correlation_x=correlation_noise,
        correlation_y=correlation_noise,
    )

    sampler = samplers.SplitMultinormal(
        dim_x=dim_x,
        dim_y=dim_y,
        covariance=covariance,
    )

    task_id = task_id if task_id is not None else f"mn-sparse-{dim_x}-{dim_y}-{n_samples}"
    task = generate_task(
        sampler=sampler,
        n_samples=n_samples,
        seeds=seeds,
        task_id=task_id,
        task_params=dict(
            correlation_signal=correlation_signal, correlation_noise=correlation_noise
        ),
    )

    return task


# Uniform correlations

task_mn_uniform_2_2 = task_mn_uniform(dim_x=2, dim_y=2, n_samples=5000)
task_mn_uniform_2_5 = task_mn_uniform(dim_x=2, dim_y=5, n_samples=5000)
task_mn_uniform_5_5 = task_mn_uniform(dim_x=5, dim_y=5, n_samples=5000)
task_mn_uniform_25_25 = task_mn_uniform(dim_x=25, dim_y=25, n_samples=5000)
task_mn_uniform_50_50 = task_mn_uniform(dim_x=50, dim_y=50, n_samples=5000)


# Sparse correlations

task_mn_sparse_3_3 = task_mn_sparse(dim_x=3, dim_y=3, n_samples=5000)
task_mn_sparse_2_5 = task_mn_sparse(dim_x=2, dim_y=5, n_samples=5000)
task_mn_sparse_5_5 = task_mn_sparse(dim_x=5, dim_y=5, n_samples=5000)
task_mn_sparse_25_25 = task_mn_sparse(dim_x=25, dim_y=25, n_samples=5000)

task_mn_sparse_5_5_no_noise = task_mn_sparse(
    dim_x=5,
    dim_y=5,
    n_samples=5000,
    correlation_noise=0.0,
    task_id="mn-sparse-5-5-5000-no-noise",
)


MULTINORMAL_TASKS = (
    task_mn_uniform_2_2,
    task_mn_uniform_2_5,
    task_mn_uniform_5_5,
    task_mn_uniform_25_25,
    task_mn_uniform_50_50,
    task_mn_sparse_3_3,
    task_mn_sparse_2_5,
    task_mn_sparse_5_5,
    task_mn_sparse_25_25,
    task_mn_sparse_5_5_no_noise,
)
