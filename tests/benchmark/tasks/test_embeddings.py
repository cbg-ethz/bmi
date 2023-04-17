from bmi.benchmark.tasks.bivariate_normal import task_bivariate_normal
from bmi.benchmark.tasks.embeddings import transform_swissroll_task


def test_swissroll(n_samples: int = 20, correlation: float = 0.9) -> None:
    task = transform_swissroll_task(task_bivariate_normal(), "test-task")

    xs, ys = task.sample(n_samples=n_samples, seed=42)

    assert xs.shape == (n_samples, 2)
    assert ys.shape == (n_samples, 1)
