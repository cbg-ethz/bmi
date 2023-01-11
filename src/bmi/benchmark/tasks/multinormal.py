from typing import Iterable, Optional

import numpy as np

import bmi.samplers.api as samplers
from bmi.benchmark.core import Task, generate_task


def task_multinormal_uniform(
    dim_x: int,
    dim_y: int,
    n_samples: int,
    n_seeds: int,
) -> Task:
    sampler = samplers.SplitMultinormal(
        dim_x=dim_x,
        dim_y=dim_y,
        covariance=np.eye(dim_x + dim_y) * 0.5 + 0.5,
    )

    task = generate_task(
        sampler=sampler,
        n_samples=n_samples,
        seeds=range(n_seeds),
        task_id=f"multinormal-uniform-{dim_x}-{dim_y}-{n_samples}",
    )

    return task


def task_multinormal_sparse(
    dim_x: int,
    dim_y: int,
    n_samples: int,
    n_seeds: int,
    correlation_signal: float = 0.8,
    correlation_noise: float = 0.1,
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

    default_task_id = f"multinormal-sparse-{dim_x}-{dim_y}-{n_samples}"
    task_id = task_id if task_id is not None else default_task_id
    task = generate_task(
        sampler=sampler,
        n_samples=n_samples,
        seeds=range(n_seeds),
        task_id=task_id,
        task_params=dict(
            correlation_signal=correlation_signal, correlation_noise=correlation_noise
        ),
    )

    return task


def _generate_uniform_tasks(n_seeds: int, n_samples: int = 5000) -> Iterable[Task]:
    """Uniform correlations between all variables."""
    yield task_multinormal_uniform(dim_x=2, dim_y=2, n_samples=n_samples, n_seeds=n_seeds)
    yield task_multinormal_uniform(dim_x=2, dim_y=5, n_samples=n_samples, n_seeds=n_seeds)
    yield task_multinormal_uniform(dim_x=5, dim_y=5, n_samples=n_samples, n_seeds=n_seeds)
    yield task_multinormal_uniform(dim_x=25, dim_y=25, n_samples=n_samples, n_seeds=n_seeds)
    yield task_multinormal_uniform(dim_x=50, dim_y=50, n_samples=n_samples, n_seeds=n_seeds)


def _generate_sparse_tasks(n_seeds: int, n_samples: int = 5000) -> Iterable[Task]:
    """Sparse correlations."""
    yield task_multinormal_sparse(dim_x=3, dim_y=3, n_samples=n_samples, n_seeds=n_seeds)
    yield task_multinormal_sparse(dim_x=2, dim_y=5, n_samples=n_samples, n_seeds=n_seeds)
    yield task_multinormal_sparse(dim_x=5, dim_y=5, n_samples=n_samples, n_seeds=n_seeds)
    yield task_multinormal_sparse(dim_x=25, dim_y=25, n_samples=n_samples, n_seeds=n_seeds)
    yield task_multinormal_sparse(
        dim_x=5,
        dim_y=5,
        n_samples=n_samples,
        correlation_noise=0.0,
        task_id="multinormal-sparse-5-5-5000-no-noise",
        n_seeds=n_seeds,
    )


def generate_tasks(n_seeds: int) -> Iterable[Task]:
    yield from _generate_uniform_tasks(n_seeds=n_seeds)
    yield from _generate_sparse_tasks(n_seeds=n_seeds)
