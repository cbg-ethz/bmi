import bmi.benchmark.tasks.additive_noise as additive_noise
import bmi.benchmark.tasks.bimodal_gaussians as bimodal_gaussians
import bmi.benchmark.tasks.bivariate_normal as binormal
import bmi.benchmark.tasks.embeddings as embeddings
import bmi.benchmark.tasks.multinormal as multinormal
import bmi.benchmark.tasks.student as student
from bmi.benchmark.tasks.asinh import transform_asinh_task as asinh
from bmi.benchmark.tasks.half_cube import transform_half_cube_task as half_cube
from bmi.benchmark.tasks.normal_cdf import transform_normal_cdf_task as normal_cdfise
from bmi.benchmark.tasks.spiral import transform_spiral_task as spiralise
from bmi.benchmark.tasks.wiggly import transform_wiggly_task as wigglify

BINORMAL_BASE = binormal.task_bivariate_normal(gaussian_correlation=0.75)
UNIFORM_BASE = normal_cdfise(BINORMAL_BASE)
BISTUDENT_BASE = student.task_student_dense(dim_x=1, dim_y=1, df=5, off_diag=0.75)

ONE_DIM_TASKS = [
    BINORMAL_BASE,
    UNIFORM_BASE,
    additive_noise.task_additive_noise(epsilon=0.10),
    additive_noise.task_additive_noise(epsilon=0.75),
    bimodal_gaussians.task_bimodal_gaussians(),
    wigglify(BINORMAL_BASE),
    half_cube(BINORMAL_BASE),
    BISTUDENT_BASE,
]

EMBEDDINGS_TASKS = [
    embeddings.transform_swissroll_task(UNIFORM_BASE, task_name="Swiss roll 2 × 1"),
]

MULTINORMAL_TASKS = [
    multinormal.task_multinormal_dense(2, 2),
    multinormal.task_multinormal_dense(3, 3),
    multinormal.task_multinormal_dense(5, 5),
    multinormal.task_multinormal_dense(25, 25),
    multinormal.task_multinormal_dense(50, 50),
    multinormal.task_multinormal_sparse(2, 2),
    multinormal.task_multinormal_sparse(3, 3),
    multinormal.task_multinormal_sparse(5, 5),
    multinormal.task_multinormal_sparse(25, 25),
]

STUDENT_TASKS = [
    student.task_student_identity(dim_x=2, dim_y=2, df=3),
    student.task_student_identity(dim_x=2, dim_y=2, df=5),
    student.task_student_identity(dim_x=5, dim_y=5, df=3),
    student.task_student_identity(dim_x=5, dim_y=5, df=5),
    student.task_student_identity(dim_x=10, dim_y=10, df=3),
    student.task_student_identity(dim_x=10, dim_y=10, df=5),
]

TRANS_MULTINORMAL_BASE_3x3 = multinormal.task_multinormal_sparse(3, 3)
TRANS_MULTINORMAL_BASE_5x5 = multinormal.task_multinormal_sparse(5, 5)
TRANS_MULTINORMAL_BASE_25x25 = multinormal.task_multinormal_sparse(25, 25)

# TODO(Pawel, Frederic): Think about replacing these with identity.
#   In particular:
#     - Run again these tasks in `benchmark.smk` workflow
#     - Update `figure_visualise_tasks.smk` workflow
TRANS_STUDENT_BASE_3x3 = student.task_student_sparse(3, 3, 5)
TRANS_STUDENT_BASE_5x5 = student.task_student_sparse(5, 5, 5)
TRANS_STUDENT_BASE_25x25 = student.task_student_sparse(25, 25, 5)

TRANSFORMED_TASKS = [
    normal_cdfise(TRANS_MULTINORMAL_BASE_3x3),
    normal_cdfise(TRANS_MULTINORMAL_BASE_5x5),
    normal_cdfise(TRANS_MULTINORMAL_BASE_25x25),
    half_cube(TRANS_MULTINORMAL_BASE_3x3),
    half_cube(TRANS_MULTINORMAL_BASE_5x5),
    half_cube(TRANS_MULTINORMAL_BASE_25x25),
    spiralise(TRANS_MULTINORMAL_BASE_3x3),
    spiralise(TRANS_MULTINORMAL_BASE_5x5),
    spiralise(TRANS_MULTINORMAL_BASE_25x25),
    spiralise(normal_cdfise(TRANS_MULTINORMAL_BASE_3x3)),
    spiralise(normal_cdfise(TRANS_MULTINORMAL_BASE_5x5)),
    spiralise(normal_cdfise(TRANS_MULTINORMAL_BASE_25x25)),
    asinh(TRANS_STUDENT_BASE_3x3),
    asinh(TRANS_STUDENT_BASE_5x5),
    asinh(TRANS_STUDENT_BASE_25x25),
]


BENCHMARK_TASKS_LIST = (
    ONE_DIM_TASKS + EMBEDDINGS_TASKS + MULTINORMAL_TASKS + STUDENT_TASKS + TRANSFORMED_TASKS
)


__all__ = [
    "BENCHMARK_TASKS_LIST",
]
