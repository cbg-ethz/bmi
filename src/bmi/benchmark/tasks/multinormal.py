from typing import Iterable, Optional

import numpy as np

import bmi.samplers.api as samplers
from bmi.benchmark.core import Task, generate_task

SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


def task_mn_uniform(
    dim_x: int,
    dim_y: int,
    n_samples: int,
    seeds=SEEDS,
) -> Task:
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
) -> Task:
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


def _generate_uniform_tasks() -> Iterable[Task]:
    """Uniform correlations between all variables."""
    yield task_mn_uniform(dim_x=2, dim_y=2, n_samples=5000)
    yield task_mn_uniform(dim_x=2, dim_y=5, n_samples=5000)
    yield task_mn_uniform(dim_x=5, dim_y=5, n_samples=5000)
    yield task_mn_uniform(dim_x=25, dim_y=25, n_samples=5000)
    yield task_mn_uniform(dim_x=50, dim_y=50, n_samples=5000)


def _generate_sparse_tasks() -> Iterable[Task]:
    """Sparse correlations."""
    yield task_mn_sparse(dim_x=3, dim_y=3, n_samples=5000)
    yield task_mn_sparse(dim_x=2, dim_y=5, n_samples=5000)
    yield task_mn_sparse(dim_x=5, dim_y=5, n_samples=5000)
    yield task_mn_sparse(dim_x=25, dim_y=25, n_samples=5000)
    yield task_mn_sparse(
        dim_x=5,
        dim_y=5,
        n_samples=5000,
        correlation_noise=0.0,
        task_id="mn-sparse-5-5-5000-no-noise",
    )


def generate_tasks() -> Iterable[Task]:
    yield from _generate_uniform_tasks()
    yield from _generate_sparse_tasks()
