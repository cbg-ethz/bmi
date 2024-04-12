# ====================================
# == Configuration of the benchmark ==
# ====================================

import bmi.benchmark.tasks.additive_noise as additive_noise
import bmi.benchmark.tasks.bivariate_normal as binormal
import bmi.benchmark.tasks.embeddings as embeddings
import bmi.benchmark.tasks.mixtures as mixtures
import bmi.benchmark.tasks.multinormal as multinormal
import bmi.benchmark.tasks.student as student
import bmi.estimators as estimators
from bmi.benchmark.tasks import transform_rescale
from bmi.benchmark.tasks.asinh import transform_asinh_task as asinh
from bmi.benchmark.tasks.normal_cdf import transform_normal_cdf_task as normal_cdfise
from bmi.benchmark.tasks.spiral import transform_spiral_task as spiralise

# === ESTIMATORS ===
# Defines the estimators to be run in the benchmark
# Note that each estimator implements `IMutualInformationPointEstimator` interface

ESTIMATORS_DICT = {
    "NWJ": estimators.NWJEstimator(verbose=False),
    "MINE": estimators.MINEEstimator(verbose=False),
    "InfoNCE": estimators.InfoNCEEstimator(verbose=False),
    "Donsker-Varadhan": estimators.DonskerVaradhanEstimator(verbose=False),
    "KSG-10": estimators.KSGEnsembleFirstEstimator(neighborhoods=(10,)),
    "CCA": estimators.CCAMutualInformationEstimator(),
}


# === TASKS ===
# Defines the benchmark tasks.
# All tasks will be rescaled to the same range for numerical stability.

TASK_MULTINORMAL_BIVAR_1x1 = binormal.task_bivariate_normal(gaussian_correlation=0.75)
TASK_UNIFORM_FROM_BIVAR_1x1 = normal_cdfise(TASK_MULTINORMAL_BIVAR_1x1)

TASK_MULTINORMAL_2PAIR_3x3 = multinormal.task_multinormal_2pair(3, 3)
TASK_MULTINORMAL_2PAIR_5x5 = multinormal.task_multinormal_2pair(5, 5)


UNSCALED_TASKS = (
    # 1v1
    additive_noise.task_additive_noise(epsilon=0.75),
    asinh(student.task_student_identity(dim_x=1, dim_y=1, df=1)),
    mixtures.task_x(),
    mixtures.task_ai(),
    # embedding
    embeddings.transform_swissroll_task(TASK_UNIFORM_FROM_BIVAR_1x1, task_name="Swiss roll 2 × 1"),
    # nv1
    mixtures.task_waves(),
    mixtures.task_galaxy(),
    mixtures.task_concentric_multinormal(dim_x=3, n_components=5),
    mixtures.task_concentric_multinormal(dim_x=5, n_components=5),
    mixtures.task_concentric_multinormal(dim_x=3, n_components=10),  # *
    mixtures.task_concentric_multinormal(dim_x=5, n_components=10),  # *
    mixtures.task_concentric_multinormal(dim_x=25, n_components=5),  # *
    # multinormal
    multinormal.task_multinormal_dense(5, 5),
    multinormal.task_multinormal_dense(25, 25),
    multinormal.task_multinormal_dense(50, 50),
    multinormal.task_multinormal_2pair(5, 5),
    multinormal.task_multinormal_2pair(25, 25),
    # inliers
    mixtures.task_multinormal_sparse_w_inliers(dim_x=5, dim_y=5, inlier_fraction=0.2),
    mixtures.task_multinormal_sparse_w_inliers(dim_x=25, dim_y=25, inlier_fraction=0.2),
    mixtures.task_multinormal_sparse_w_inliers(dim_x=5, dim_y=5, inlier_fraction=0.5),  # *
    mixtures.task_multinormal_sparse_w_inliers(dim_x=25, dim_y=25, inlier_fraction=0.5),  # *
    # student
    asinh(student.task_student_identity(dim_x=2, dim_y=2, df=1)),
    asinh(student.task_student_identity(dim_x=3, dim_y=3, df=2)),
    asinh(student.task_student_identity(dim_x=5, dim_y=5, df=2)),
    # transformed
    spiralise(TASK_MULTINORMAL_2PAIR_3x3),
    spiralise(TASK_MULTINORMAL_2PAIR_5x5),
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

N_SAMPLES: list[int] = [5_000]


# === SEEDS ===
# Seeds used for task sampling

SEEDS: list[int] = list(range(10))
