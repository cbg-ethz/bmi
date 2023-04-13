from bmi.benchmark.result import run_estimator
from bmi.benchmark.task_list import BENCHMARK_TASKS_LIST as _BENCHMARK_LIST

BENCHMARK_TASKS = {task.id: task for task in _BENCHMARK_LIST}


__all__ = [
    "BENCHMARK_TASKS",
    "run_estimator",
]
