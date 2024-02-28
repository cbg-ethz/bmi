"""Benchmarking mutual information package."""

import bmi.benchmark as benchmark
import bmi.estimators as estimators
import bmi.samplers as samplers
import bmi.transforms as transforms
import bmi.utils as utils

# ISort doesn't want to split these into several lines, conflicting with Black
# isort: off
from bmi.benchmark import (
    BENCHMARK_TASKS,
    run_estimator,
    Task,
    TaskMetadata,
    RunResult,
)

# isort: on

# ISort doesn't want to split these into several lines, conflicting with Black
# isort: off
from bmi.interface import (
    IMutualInformationPointEstimator,
    ISampler,
    Pathlike,
)

# isort: on

__all__ = [
    "benchmark",
    "estimators",
    "samplers",
    "transforms",
    "utils",
    "IMutualInformationPointEstimator",
    "ISampler",
    "Pathlike",
    "RunResult",
    "Task",
    "TaskMetadata",
    "BENCHMARK_TASKS",
    "run_estimator",
]
