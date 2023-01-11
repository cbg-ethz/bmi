from typing import Iterable

import numpy as np

import bmi.samplers.api as samplers
from bmi.benchmark.core import Task, generate_task
from bmi.interface import ISampler

N_SAMPLES: int = 5000


def _generate_uniform_task(epsilon: float, n_seeds: int, n_samples: int = N_SAMPLES) -> Task:
    return generate_task(
        sampler=samplers.AdditiveUniformSampler(epsilon=epsilon),
        task_id=f"one-dimensional-uniform-additive-{epsilon}",
        n_samples=n_samples,
        task_params={"epsilon": epsilon},
        seeds=range(n_seeds),
    )


def _generate_uniform_tasks(n_seeds: int, n_samples: int) -> Iterable[Task]:
    yield _generate_uniform_task(epsilon=2.0, n_seeds=n_seeds, n_samples=n_samples)
    yield _generate_uniform_task(epsilon=0.1, n_seeds=n_seeds, n_samples=n_samples)


def get_student_sampler(df: int = 5, strength: float = 0.7) -> ISampler:
    return samplers.SplitStudentT(
        dim_x=1,
        dim_y=1,
        df=df,
        dispersion=np.asarray(
            [
                [1.0, strength],
                [strength, 1.0],
            ]
        ),
    )


def generate_student_task(
    n_seeds: int, n_samples: int, df: int = 5, strength: float = 0.7
) -> Task:
    if strength >= 1 or strength <= -1:
        raise ValueError(f"Strength must be between (-1, 1), was {strength}.")
    sampler = get_student_sampler(df=df, strength=strength)
    task_id = f"one-dimensional-student-{df}_df-{strength}_strength"
    task_params = dict(degrees_of_freedom=df, strength=strength)
    return generate_task(
        sampler=sampler,
        n_samples=n_samples,
        seeds=range(n_seeds),
        task_id=task_id,
        task_params=task_params,
    )


def generate_tasks(n_seeds: int = 10, n_samples: int = N_SAMPLES) -> Iterable[Task]:
    yield from _generate_uniform_tasks(n_seeds=n_seeds, n_samples=n_samples)
    yield generate_student_task(n_seeds=n_samples, n_samples=n_samples)
