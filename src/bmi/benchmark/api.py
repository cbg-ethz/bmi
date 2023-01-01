from bmi.benchmark._serialize import TaskDirectory
from bmi.benchmark.core import Task, TaskMetadata, generate_task
from bmi.benchmark.tasks.api import BENCHMARK_TASKS, save_benchmark_tasks
from bmi.benchmark.tasks.spiral import generate_spiral_invariance_task

# ISort doesn't want to split these into several lines, conflicting with Black
# isort: off
from bmi.benchmark.wrapper import (
    ExternalEstimator,
    REstimatorKSG,
    REstimatorLNN,
    run_external_estimator,
    WrappedEstimator,
)

# isort: on

__all__ = [
    "generate_task",
    "generate_spiral_invariance_task",
    "run_external_estimator",
    "save_benchmark_tasks",
    "ExternalEstimator",
    "REstimatorKSG",
    "REstimatorLNN",
    "Task",
    "TaskDirectory",
    "TaskMetadata",
    "BENCHMARK_TASKS",
    "WrappedEstimator",
]
