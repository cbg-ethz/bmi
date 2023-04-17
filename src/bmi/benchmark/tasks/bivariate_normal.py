import bmi.samplers.api as samplers
from bmi.benchmark.task import Task


def task_bivariate_normal(
    gaussian_correlation=0.75,
) -> Task:
    sampler = samplers.BivariateNormalSampler(correlation=gaussian_correlation)

    return Task(
        sampler=sampler,
        task_id=f"1v1-normal-{gaussian_correlation}",
        task_name="Binormal 1 × 1",
        task_params={
            "gaussian_correlation": gaussian_correlation,
        },
    )
