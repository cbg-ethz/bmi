from pathlib import Path
from typing import Iterable

import bmi.benchmark.tasks.multinormal as mn
import bmi.benchmark.tasks.spiral as spiral
from bmi.benchmark.core import Task
from bmi.interface import Pathlike

BENCHMARK_TASKS = (
    mn.task_mn_1_2_eye_100,
    mn.task_mn_1_2_eye_1000,
)


def save_benchmark_tasks(
    tasks_dir: Pathlike,
    tasks: Iterable[Task] = BENCHMARK_TASKS,
    exist_ok: bool = False,
):
    for task in tasks:
        task_dir = Path(tasks_dir) / task.task_id
        task.save(task_dir, exist_ok=exist_ok)


__all__ = [
    "BENCHMARK_TASKS",
    "save_benchmark_tasks",
    "spiral",
]
