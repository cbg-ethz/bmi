import bmi.samplers as samplers
from bmi.benchmark.task import Task
from bmi.benchmark.tasks.bivariate_normal import task_bivariate_normal
from bmi.benchmark.tasks.normal_cdf import transform_normal_cdf_task as normal_cdfise
from bmi.transforms import invert_cdf, normal_cdf


def task_bimodal_gaussians(
    gaussian_correlation=0.75,
) -> Task:
    task_binormal = task_bivariate_normal(gaussian_correlation=gaussian_correlation)
    task_uniform = normal_cdfise(task_binormal)
    base_sampler = task_uniform.sampler

    icdf_x = invert_cdf(lambda x: 0.3 * normal_cdf(x + 0) + 0.7 * normal_cdf(x - 5))
    icdf_y = invert_cdf(lambda x: 0.5 * normal_cdf(x + 1) + 0.5 * normal_cdf(x - 3))

    bimodal_sampler = samplers.TransformedSampler(
        base_sampler=base_sampler,
        transform_x=icdf_x,
        transform_y=icdf_y,
    )

    return Task(
        sampler=bimodal_sampler,
        task_id=f"1v1-bimodal-{gaussian_correlation}",
        task_name="Bimodal 1 × 1",
        task_params={
            "gaussian_correlation": gaussian_correlation,
        },
    )
