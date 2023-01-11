from pathlib import Path
from typing import Callable, Iterable, Literal, Optional

import bmi.benchmark.tasks.multinormal as mn
import bmi.benchmark.tasks.spiral as spiral
import bmi.benchmark.tasks.student as st
from bmi.benchmark.core import Task
from bmi.interface import Pathlike


def _generate_benchmark_v1() -> Iterable[Task]:
    # Multinormal tasks
    yield from mn.generate_tasks()
    # Student tasks
    yield from st.generate_tasks()


def generate_benchmark(version: Literal[1] = 1) -> Iterable[Task]:
    _allowed_versions = [1]
    _error = ValueError(
        f"Benchmark version {version} not recognized. " f"Allowed versions: {_allowed_versions}."
    )
    if version not in _allowed_versions:
        raise _error
    if version == 1:
        return _generate_benchmark_v1()
    else:
        raise _error


def _default_naming_function(task: Task) -> str:
    return task.task_id


def save_benchmark_tasks(
    tasks_dir: Pathlike,
    tasks: Optional[Iterable[Task]] = None,
    exist_ok: bool = False,
    naming_function: Optional[Callable[[Task], str]] = None,
) -> None:
    """Saves given tasks.

    Args:
        tasks_dir: directory where the tasks will be saved
        tasks: tasks to be saved. By default, version 1 of the benchmark.
        exist_ok: if False and `tasks_dir` already exists,
          an error will be raised
        naming_function: function which takes a task and produces
          the subdirectory name where the task should be saved.
          The default is `lambda task: task.task_id`
    """
    if naming_function is None:
        naming_function = _default_naming_function
    if tasks is None:
        tasks = generate_benchmark(version=1)

    for task in tasks:
        task_dir = Path(tasks_dir) / naming_function(task)
        task.save(task_dir, exist_ok=exist_ok)


__all__ = [
    "generate_benchmark",
    "save_benchmark_tasks",
    "spiral",
]
