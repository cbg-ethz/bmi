from bmi.benchmark.result import RunResult, run_estimator
from bmi.benchmark.task import Task, TaskMetadata
from bmi.benchmark.task_list import BENCHMARK_TASKS_LIST as _BENCHMARK_LIST

BENCHMARK_TASKS = {task.id: task for task in _BENCHMARK_LIST}


__all__ = [
    "BENCHMARK_TASKS",
    "run_estimator",
    "RunResult",
    "Task",
    "TaskMetadata",
]
