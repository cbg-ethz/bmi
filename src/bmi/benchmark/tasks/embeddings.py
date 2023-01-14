import bmi.samplers.transformed as tr
from bmi.benchmark.core import Task, generate_task


def generate_swissroll_task(correlation: float, n_seeds: int, n_samples: int) -> Task:
    uniform_sampler = tr.BivariateUniformMarginsSampler(gaussian_correlation=correlation)
    swissroll_sampler = tr.SwissRollSampler(sampler=uniform_sampler)

    return generate_task(
        sampler=swissroll_sampler,
        seeds=range(n_seeds),
        n_samples=n_samples,
        task_id=f"swissroll-gaussian_correlation{correlation:.4f}",
        task_params=dict(gaussian_correlation=correlation),
    )
