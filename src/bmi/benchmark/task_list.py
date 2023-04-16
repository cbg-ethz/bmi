import bmi.benchmark.tasks.bimodal_gaussians as bimodal_gaussians
import bmi.benchmark.tasks.bivariate_normal as binormal
import bmi.benchmark.tasks.embeddings as embeddings
import bmi.benchmark.tasks.multinormal as multinormal
import bmi.benchmark.tasks.student as student
from bmi.benchmark.tasks.normal_cdf import transform_normal_cdf_task as normal_cdfise
from bmi.benchmark.tasks.spiral import transform_spiral_task as spiralise

ONE_DIM_TASKS = [
    normal_cdfise(binormal.task_bivariate_normal(), task_name="Uniform 1 × 1"),
    bimodal_gaussians.task_bimodal_gaussians(),
]

EMBEDDINGS_TASKS = [
    embeddings.generate_swissroll_task(gaussian_correlation=0.9, task_name="Swiss roll"),
]

MULTINORMAL_TASKS = [
    multinormal.task_multinormal_dense(2, 2),
    multinormal.task_multinormal_dense(2, 5),
    multinormal.task_multinormal_dense(5, 5),
    multinormal.task_multinormal_dense(25, 25),
    multinormal.task_multinormal_dense(50, 50),
    multinormal.task_multinormal_sparse(2, 2),
    multinormal.task_multinormal_sparse(3, 3),
    multinormal.task_multinormal_sparse(5, 5),
    multinormal.task_multinormal_sparse(25, 25),
    multinormal.task_multinormal_sparse(
        5, 5, correlation_noise=0.0, task_name="Multinormal 5 × 5 (sparse, no noise)"
    ),
]

STUDENT_TASKS = [
    student.task_student_dense(dim_x=2, dim_y=2, df=5),
    student.task_student_dense(dim_x=2, dim_y=5, df=5),
    student.task_student_dense(dim_x=5, dim_y=5, df=5),
    student.task_student_dense(dim_x=25, dim_y=25, df=5),
    student.task_student_sparse(dim_x=2, dim_y=2, df=5),
    student.task_student_sparse(dim_x=3, dim_y=3, df=5),
    student.task_student_sparse(dim_x=5, dim_y=5, df=5),
    student.task_student_dense(dim_x=5, dim_y=5, df=2),
    student.task_student_dense(dim_x=5, dim_y=5, df=3),
    student.task_student_dense(dim_x=5, dim_y=5, df=10),
]

TRANSFORMED_TASKS = [
    normal_cdfise(multinormal.task_multinormal_sparse(3, 3), task_name="Uniform 3 × 3"),
    normal_cdfise(multinormal.task_multinormal_sparse(5, 5), task_name="Uniform 5 × 5"),
    normal_cdfise(multinormal.task_multinormal_sparse(25, 25), task_name="Uniform 25 × 25"),
    spiralise(multinormal.task_multinormal_sparse(3, 3)),
    spiralise(multinormal.task_multinormal_sparse(5, 5)),
    spiralise(multinormal.task_multinormal_sparse(25, 25)),
]


BENCHMARK_TASKS_LIST = (
    ONE_DIM_TASKS + EMBEDDINGS_TASKS + MULTINORMAL_TASKS + STUDENT_TASKS + TRANSFORMED_TASKS
)


__all__ = [
    "BENCHMARK_TASKS_LIST",
]
