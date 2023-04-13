from bmi.benchmark.task_list import BENCHMARK_TASKS as _BENCHMARK_LIST


BENCHMARK_TASKS = {
    task.id: task
    for task in _BENCHMARK_LIST
}


__all__ = (
    'BENCHMARK_TASKS',
)
