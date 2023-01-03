from typing import Optional

import numpy as np

import bmi.samplers.api as samplers
from bmi.benchmark.core import generate_task

SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


def task_st_uniform(
    dim_x: int,
    dim_y: int,
    df: float,
    n_samples: int,
    seeds=SEEDS,
):
    sampler = samplers.SplitStudentT(
        dim_x=dim_x,
        dim_y=dim_y,
        dispersion=np.eye(dim_x + dim_y) * 0.5 + 0.5,
        df=df,
    )

    task = generate_task(
        sampler=sampler,
        n_samples=n_samples,
        seeds=seeds,
        task_id=f"st-uniform-{dim_x}-{dim_y}-{df}-{n_samples}",
    )

    return task


def task_st_sparse(
    dim_x: int,
    dim_y: int,
    df: float,
    n_samples: int,
    dispersion_signal: float = 0.8,
    dispersion_noise: float = 0.1,
    seeds=SEEDS,
    task_id: Optional[str] = None,
):
    dispersion = samplers.parametrised_correlation_matrix(
        dim_x=dim_x,
        dim_y=dim_y,
        k=2,
        correlation=dispersion_signal,
        correlation_x=dispersion_noise,
        correlation_y=dispersion_noise,
    )

    sampler = samplers.SplitStudentT(
        dim_x=dim_x,
        dim_y=dim_y,
        dispersion=dispersion,
        df=df,
    )

    task_id = task_id if task_id is not None else f"st-sparse-{dim_x}-{dim_y}-{df}-{n_samples}"
    task = generate_task(
        sampler=sampler,
        n_samples=n_samples,
        seeds=seeds,
        task_id=task_id,
        task_params=dict(dispersion_signal=dispersion_signal, dispersion_noise=dispersion_noise),
    )

    return task


# Uniform correlations

# no covariance here! let's see how this goes...
task_st_uniform_5_5_2 = task_st_uniform(dim_x=5, dim_y=5, df=2, n_samples=5000)

task_st_uniform_5_5_3 = task_st_uniform(dim_x=5, dim_y=5, df=3, n_samples=5000)
task_st_uniform_5_5_5 = task_st_uniform(dim_x=5, dim_y=5, df=5, n_samples=5000)
task_st_uniform_5_5_10 = task_st_uniform(dim_x=5, dim_y=5, df=10, n_samples=5000)
task_st_uniform_5_5_30 = task_st_uniform(dim_x=5, dim_y=5, df=30, n_samples=5000)

task_st_uniform_2_2_5 = task_st_uniform(dim_x=2, dim_y=2, df=5, n_samples=5000)
task_st_uniform_25_25_5 = task_st_uniform(dim_x=25, dim_y=25, df=5, n_samples=5000)


# Sparse correlations

task_st_sparse_3_3_5 = task_st_sparse(dim_x=3, dim_y=3, df=5, n_samples=5000)
task_st_sparse_2_5_5 = task_st_sparse(dim_x=2, dim_y=5, df=5, n_samples=5000)
task_st_sparse_5_5_5 = task_st_sparse(dim_x=5, dim_y=5, df=5, n_samples=5000)


STUDENT_T_TASKS = (
    task_st_uniform_5_5_2,
    task_st_uniform_5_5_3,
    task_st_uniform_5_5_5,
    task_st_uniform_5_5_10,
    task_st_uniform_5_5_30,
    task_st_uniform_2_2_5,
    task_st_uniform_25_25_5,
    task_st_sparse_3_3_5,
    task_st_sparse_2_5_5,
    task_st_sparse_5_5_5,
)
