from typing import Iterable, Optional

import numpy as np

import bmi.samplers.api as samplers
from bmi.benchmark.core import Task, generate_task

SEEDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


def task_student_uniform(
    dim_x: int,
    dim_y: int,
    df: int,
    n_samples: int,
    seeds=SEEDS,
) -> Task:
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
        task_id=f"student-uniform-{dim_x}-{dim_y}-{df}-{n_samples}",
        task_params=dict(
            degrees_of_freedom=df,
        ),
    )

    return task


def task_student_sparse(
    dim_x: int,
    dim_y: int,
    df: int,
    n_samples: int,
    dispersion_signal: float = 0.8,
    dispersion_noise: float = 0.1,
    seeds=SEEDS,
    task_id: Optional[str] = None,
) -> Task:
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

    default_task_id = f"student-sparse-{dim_x}-{dim_y}-{df}-{n_samples}"
    task_id = task_id if task_id is not None else default_task_id
    task = generate_task(
        sampler=sampler,
        n_samples=n_samples,
        seeds=seeds,
        task_id=task_id,
        task_params=dict(
            dispersion_signal=dispersion_signal,
            dispersion_noise=dispersion_noise,
            degrees_of_freedom=df,
        ),
    )

    return task


def _generate_uniform() -> Iterable[Task]:
    """Uniform dispersion matrix."""

    # Covariance for df=2 doesn't exist! Let's see how this goes...
    yield task_student_uniform(dim_x=5, dim_y=5, df=2, n_samples=5000)

    # Larger df have covariance
    yield task_student_uniform(dim_x=5, dim_y=5, df=3, n_samples=5000)
    yield task_student_uniform(dim_x=5, dim_y=5, df=5, n_samples=5000)
    yield task_student_uniform(dim_x=5, dim_y=5, df=10, n_samples=5000)
    yield task_student_uniform(dim_x=5, dim_y=5, df=30, n_samples=5000)

    # Different dimensions with df = 5
    yield task_student_uniform(dim_x=2, dim_y=2, df=5, n_samples=5000)
    yield task_student_uniform(dim_x=25, dim_y=25, df=5, n_samples=5000)


def _generate_sparse() -> Iterable[Task]:
    """Sparse dispersion matrix."""
    yield task_student_sparse(dim_x=3, dim_y=3, df=5, n_samples=5000)
    yield task_student_sparse(dim_x=2, dim_y=5, df=5, n_samples=5000)
    yield task_student_sparse(dim_x=5, dim_y=5, df=5, n_samples=5000)


def generate_tasks() -> Iterable[Task]:
    yield from _generate_uniform()
    yield from _generate_sparse()
