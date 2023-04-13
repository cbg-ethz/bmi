import bmi.benchmark.tasks.multinormal as multinormal

BENCHMARK_TASKS = [
    multinormal.task_multinormal_dense(2, 2),
    multinormal.task_multinormal_dense(2, 5),
    multinormal.task_multinormal_dense(5, 5),
    multinormal.task_multinormal_dense(25, 25),
    multinormal.task_multinormal_dense(50, 50),

    multinormal.task_multinormal_sparse(2, 2),
    multinormal.task_multinormal_sparse(3, 3),
    multinormal.task_multinormal_sparse(5, 5),
    multinormal.task_multinormal_sparse(5, 5, correlation_noise=0.0, task_name='Multinormal 5 × 5 (sparse, no noise)'),
    multinormal.task_multinormal_sparse(25, 25),
]
