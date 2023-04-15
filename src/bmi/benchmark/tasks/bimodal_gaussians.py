from jax.scipy.special import erf

import bmi.samplers.api as samplers
from bmi.benchmark.task import Task
from bmi.transforms.invert_cdf import invert_cdf


def normal_cdf(x):
    return 0.5 * (1 + erf(x / 2**0.5))


def task_bimodal_gaussians(
    gaussian_correlation=0.75,
) -> Task:
    gaussian_sampler = samplers.BivariateNormalSampler(correlation=gaussian_correlation)

    icdf_x = invert_cdf(lambda x: 0.3 * normal_cdf(x + 0) + 0.7 * normal_cdf(x - 5))
    icdf_y = invert_cdf(lambda x: 0.5 * normal_cdf(x + 1) + 0.5 * normal_cdf(x - 3))

    bimodal_sampler = samplers.TransformedSampler(
        base_sampler=gaussian_sampler,
        transform_x=lambda x: icdf_x(normal_cdf(x)),
        transform_y=lambda y: icdf_y(normal_cdf(y)),
    )

    return Task(
        sampler=bimodal_sampler,
        task_id=f"1v1-bimodal-{gaussian_correlation}",
        task_name="Bimodal gaussians 1 × 1",
        task_params={
            "gaussian_correlation": gaussian_correlation,
        },
    )
