from typing import Iterable

from bmi.benchmark.core import Task, generate_task


def generate_tasks(n_seeds: int) -> Iterable[Task]:
    raise NotImplementedError
