import bmi.samplers.transformed as tr
from bmi.benchmark.task import Task


def generate_swissroll_task(gaussian_correlation: float, task_name: str) -> Task:
    uniform_sampler = tr.BivariateUniformMarginsSampler(gaussian_correlation=gaussian_correlation)
    swissroll_sampler = tr.SwissRollSampler(sampler=uniform_sampler)

    return Task(
        sampler=swissroll_sampler,
        task_id=f"swissroll-gaussian_correlation{gaussian_correlation:.4f}",
        task_name=task_name,
        task_params=dict(gaussian_correlation=gaussian_correlation),
    )
