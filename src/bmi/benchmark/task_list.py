import bmi.benchmark.tasks.embeddings as embeddings
import bmi.benchmark.tasks.multinormal as multinormal
import bmi.benchmark.tasks.student as student

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
    multinormal.task_multinormal_sparse(
        5, 5, correlation_noise=0.0, task_name="Multinormal 5 × 5 (sparse, no noise)"
    ),
    multinormal.task_multinormal_sparse(25, 25),
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


BENCHMARK_TASKS_LIST = EMBEDDINGS_TASKS + MULTINORMAL_TASKS + STUDENT_TASKS


__all__ = [
    "BENCHMARK_TASKS_LIST",
]
