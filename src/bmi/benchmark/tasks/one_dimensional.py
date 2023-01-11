from typing import Iterable

from bmi.benchmark.core import Task, generate_task
from bmi.samplers.api import AdditiveUniformSampler

N_SAMPLES: int = 5000


def _generate_uniform_task(epsilon: float, n_seeds: int, n_samples: int = N_SAMPLES) -> Task:
    return generate_task(
        sampler=AdditiveUniformSampler(epsilon=epsilon),
        task_id=f"one-dimensional-uniform-additive-{epsilon}",
        n_samples=n_samples,
        task_params={"epsilon": epsilon},
        seeds=range(n_seeds),
    )


def _generate_uniform_tasks(n_seeds: int, n_samples: int) -> Iterable[Task]:
    yield _generate_uniform_task(epsilon=2.0, n_seeds=n_seeds, n_samples=n_samples)
    yield _generate_uniform_task(epsilon=0.1, n_seeds=n_seeds, n_samples=n_samples)


def generate_tasks(n_seeds: int = 10, n_samples: int = N_SAMPLES) -> Iterable[Task]:
    yield from _generate_uniform_tasks(n_seeds=n_seeds, n_samples=n_samples)
