# ====================================
# == Configuration of the benchmark ==
# ====================================
import bmi.benchmark.tasks.additive_noise as additive_noise
import bmi.benchmark.tasks.bimodal_gaussians as bimodal_gaussians
import bmi.benchmark.tasks.bivariate_normal as binormal
import bmi.benchmark.tasks.embeddings as embeddings
import bmi.benchmark.tasks.multinormal as multinormal
import bmi.benchmark.tasks.student as student
import bmi.estimators as estimators
import bmi.estimators.external.julia_estimators as julia_estimators
import bmi.estimators.external.r_estimators as r_estimators
from bmi.benchmark.tasks import transform_rescale
from bmi.benchmark.tasks.asinh import transform_asinh_task as asinh
from bmi.benchmark.tasks.half_cube import transform_half_cube_task as half_cube
from bmi.benchmark.tasks.normal_cdf import transform_normal_cdf_task as normal_cdfise
from bmi.benchmark.tasks.spiral import transform_spiral_task as spiralise
from bmi.benchmark.tasks.wiggly import transform_wiggly_task as wigglify

# === ESTIMATORS ===
# Defines the estimators to be run in the benchmark
# Note that each estimator implements `IMutualInformationPointEstimator` interface

ESTIMATORS_DICT = {
    "MINE-10_5": estimators.MINEEstimator(verbose=False, hidden_layers=(10, 5)),
    "InfoNCE-10_5": estimators.InfoNCEEstimator(verbose=False, hidden_layers=(10, 5)),
    "NWJ-10_5": estimators.NWJEstimator(verbose=False, hidden_layers=(10, 5)),
    "Donsker-Varadhan-10_5": estimators.DonskerVaradhanEstimator(
        verbose=False, hidden_layers=(10, 5)
    ),
    # 'KSG-5': estimators.KSGEnsembleFirstEstimator(neighborhoods=(5,)),
    "R-KSG-I-10": r_estimators.RKSGEstimator(variant=1, neighbors=5),
    # 'R-BNSL': r_estimators.RBNSLEstimator(),
    "R-LNN": r_estimators.RLNNEstimator(),
    "Julia-Hist-10": julia_estimators.JuliaHistogramEstimator(bins=10),
    # 'Julia-Kernel': julia_estimators.JuliaKernelEstimator(),
    "Julia-Transfer-30": julia_estimators.JuliaTransferEstimator(bins=30),
    # 'Julia-KSG-I-5': julia_estimators.JuliaKSGEstimator(variant=1, neighbors=5),
    "CCA": estimators.CCAMutualInformationEstimator(),
    # TODO(frdrc): sth like:
    # "Your Estimator": estimators.ExternalScript('/path/to/your/estimator')
}


# === TASKS ===
# Defines the benchmark tasks.
# All tasks will be rescaled to the same range for numerical stability.

TASK_MULTINORMAL_BIVAR_1x1 = binormal.task_bivariate_normal(gaussian_correlation=0.75)
TASK_MULTINORMAL_2PAIR_3x3 = multinormal.task_multinormal_2pair(3, 3)
TASK_MULTINORMAL_2PAIR_5x5 = multinormal.task_multinormal_2pair(5, 5)
TASK_MULTINORMAL_2PAIR_25x25 = multinormal.task_multinormal_2pair(25, 25)

TASK_UNIFORM_FROM_BIVAR_1x1 = normal_cdfise(TASK_MULTINORMAL_BIVAR_1x1)


UNSCALED_TASKS = (
    # 1v1
    TASK_MULTINORMAL_BIVAR_1x1,
    TASK_UNIFORM_FROM_BIVAR_1x1,
    additive_noise.task_additive_noise(epsilon=0.75),
    bimodal_gaussians.task_bimodal_gaussians(gaussian_correlation=0.75),
    wigglify(TASK_MULTINORMAL_BIVAR_1x1),
    half_cube(TASK_MULTINORMAL_BIVAR_1x1),
    student.task_student_identity(dim_x=1, dim_y=1, df=1),
    asinh(student.task_student_identity(dim_x=1, dim_y=1, df=1)),
    # embedding
    embeddings.transform_swissroll_task(TASK_UNIFORM_FROM_BIVAR_1x1, task_name="Swiss roll 2 × 1"),
    # multinormal
    multinormal.task_multinormal_dense(2, 2),
    multinormal.task_multinormal_dense(3, 3),
    multinormal.task_multinormal_dense(5, 5),
    multinormal.task_multinormal_dense(25, 25),
    multinormal.task_multinormal_dense(50, 50),
    multinormal.task_multinormal_2pair(2, 2),
    multinormal.task_multinormal_2pair(3, 3),
    multinormal.task_multinormal_2pair(5, 5),
    multinormal.task_multinormal_2pair(25, 25),
    # student
    student.task_student_identity(dim_x=2, dim_y=2, df=1),
    student.task_student_identity(dim_x=2, dim_y=2, df=2),
    student.task_student_identity(dim_x=3, dim_y=3, df=2),
    student.task_student_identity(dim_x=3, dim_y=3, df=3),
    student.task_student_identity(dim_x=5, dim_y=5, df=2),
    student.task_student_identity(dim_x=5, dim_y=5, df=3),
    # transformed
    normal_cdfise(TASK_MULTINORMAL_2PAIR_3x3),
    normal_cdfise(TASK_MULTINORMAL_2PAIR_5x5),
    normal_cdfise(TASK_MULTINORMAL_2PAIR_25x25),
    half_cube(TASK_MULTINORMAL_2PAIR_3x3),
    half_cube(TASK_MULTINORMAL_2PAIR_5x5),
    half_cube(TASK_MULTINORMAL_2PAIR_25x25),
    spiralise(TASK_MULTINORMAL_2PAIR_3x3),
    spiralise(TASK_MULTINORMAL_2PAIR_5x5),
    spiralise(TASK_MULTINORMAL_2PAIR_25x25),
    spiralise(normal_cdfise(TASK_MULTINORMAL_2PAIR_3x3)),
    spiralise(normal_cdfise(TASK_MULTINORMAL_2PAIR_5x5)),
    spiralise(normal_cdfise(TASK_MULTINORMAL_2PAIR_25x25)),
    asinh(student.task_student_identity(dim_x=2, dim_y=2, df=1)),
    asinh(student.task_student_identity(dim_x=3, dim_y=3, df=2)),
    asinh(student.task_student_identity(dim_x=5, dim_y=5, df=2)),
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
