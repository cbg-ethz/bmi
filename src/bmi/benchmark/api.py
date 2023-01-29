import bmi.benchmark.tasks.api as tasks
from bmi.benchmark.core import Task, TaskMetadata, generate_task
from bmi.benchmark.filesys.api import TaskDirectory
from bmi.benchmark.tasks.api import generate_benchmark, save_benchmark_tasks
from bmi.benchmark.tasks.spiral import generate_spiral_invariance_task
from bmi.benchmark.traverse import LoadTaskMetadata, SaveLoadRunResults

# ISort doesn't want to split these into several lines, conflicting with Black
# isort: off
from bmi.benchmark.wrapper import (
    ExternalEstimator,
    REstimatorBNSL,
    REstimatorKSG,
    REstimatorLNN,
    run_external_estimator,
    WrappedEstimator,
    JuliaEstimatorKSG,
    JuliaEstimatorKernel,
    JuliaEstimatorHistogram,
    JuliaEstimatorTransfer,
)

# isort: on

__all__ = [
    "generate_task",
    "generate_spiral_invariance_task",
    "run_external_estimator",
    "save_benchmark_tasks",
    "tasks",
    "ExternalEstimator",
    "JuliaEstimatorKSG",
    "JuliaEstimatorHistogram",
    "JuliaEstimatorTransfer",
    "JuliaEstimatorKernel",
    "REstimatorBNSL",
    "REstimatorKSG",
    "REstimatorLNN",
    "SaveLoadRunResults",
    "LoadTaskMetadata",
    "Task",
    "TaskDirectory",
    "TaskMetadata",
    "generate_benchmark",
    "WrappedEstimator",
]
