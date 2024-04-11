import jax.numpy as jnp

import bmi.samplers as samplers
from bmi.benchmark.task import Task
from bmi.samplers import fine

_MC_MI_ESTIMATE_SAMPLE = 100_000


def task_x(
    gaussian_correlation=0.9,
    mi_estimate_sample=_MC_MI_ESTIMATE_SAMPLE,
) -> Task:

    x_dist = fine.mixture(
        proportions=jnp.array([0.5, 0.5]),
        components=[
            fine.MultivariateNormalDistribution(
                covariance=samplers.canonical_correlation([x * gaussian_correlation]),
                mean=jnp.zeros(2),
                dim_x=1,
                dim_y=1,
            )
            for x in [-1, 1]
        ],
    )
    x_sampler = fine.FineSampler(x_dist, mi_estimate_sample=mi_estimate_sample)

    return Task(
        sampler=x_sampler,
        task_id=f"1v1-X-{gaussian_correlation}",
        task_name="X 1 × 1",
        task_params={
            "gaussian_correlation": gaussian_correlation,
        },
    )
