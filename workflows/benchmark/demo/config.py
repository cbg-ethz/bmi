# ====================================
# == Configuration of the benchmark ==
# ====================================
import bmi.benchmark.tasks.bivariate_normal as binormal
import bmi.benchmark.tasks.embeddings as embeddings
import bmi.benchmark.tasks.multinormal as multinormal
import bmi.estimators as estimators
from bmi.benchmark.tasks import transform_rescale
from bmi.benchmark.tasks.normal_cdf import transform_normal_cdf_task as normal_cdfise

# === ESTIMATORS ===
# Defines the estimators to be run in the benchmark
# Note that each estimator implements `IMutualInformationPointEstimator` interface

ESTIMATORS_DICT = {
    "KSG-5": estimators.KSGEnsembleFirstEstimator(neighborhoods=(5,)),
    "CCA": estimators.CCAMutualInformationEstimator(),
}


# === TASKS ===
# Defines the benchmark tasks.
# All tasks will be rescaled to the same range for numerical stability.

TASK_MULTINORMAL_BIVAR_1x1 = binormal.task_bivariate_normal(gaussian_correlation=0.75)
TASK_UNIFORM_FROM_BIVAR_1x1 = normal_cdfise(TASK_MULTINORMAL_BIVAR_1x1)

UNSCALED_TASKS = (
    embeddings.transform_swissroll_task(TASK_UNIFORM_FROM_BIVAR_1x1, task_name="Swiss roll 2 × 1"),
    multinormal.task_multinormal_dense(3, 3),
)

# Rescaled all tasks in case some estimator does not do it on its own
TASKS = (
    transform_rescale(
        base_task=base_task,
        task_name=base_task.name,
        task_id=base_task.id,
    )
    for base_task in UNSCALED_TASKS
)


# === SAMPLES ===
# Number of samples drawn from each task distribution

N_SAMPLES: list[int] = [500]


# === SEEDS ===
# Seeds used for task sampling

SEEDS: list[int] = [1, 2]
