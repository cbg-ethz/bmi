import jax.numpy as jnp
import numpy as np

import bmi.samplers as samplers
import bmi.transforms as transforms
from bmi.benchmark.task import Task
from bmi.samplers import fine

_MC_MI_ESTIMATE_SAMPLE = 100_000


def task_x(
    gaussian_correlation=0.9,
    mi_estimate_sample=_MC_MI_ESTIMATE_SAMPLE,
) -> Task:
    """The X distribution."""

    dist = fine.mixture(
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
    sampler = fine.FineSampler(dist, mi_estimate_sample=mi_estimate_sample)

    return Task(
        sampler=sampler,
        task_id=f"1v1-X-{gaussian_correlation}",
        task_name="X 1 × 1",
        task_params={
            "gaussian_correlation": gaussian_correlation,
        },
    )


def task_ai(
    mi_estimate_sample=_MC_MI_ESTIMATE_SAMPLE,
) -> Task:
    """The AI distribution."""

    corr = 0.95
    var_x = 0.04

    dist = fine.mixture(
        proportions=jnp.full(6, fill_value=1 / 6),
        components=[
            # I components
            fine.MultivariateNormalDistribution(
                dim_x=1,
                dim_y=1,
                mean=jnp.array([1.0, 0.0]),
                covariance=np.diag([0.01, 0.2]),
            ),
            fine.MultivariateNormalDistribution(
                dim_x=1,
                dim_y=1,
                mean=jnp.array([1.0, 1]),
                covariance=np.diag([0.05, 0.001]),
            ),
            fine.MultivariateNormalDistribution(
                dim_x=1,
                dim_y=1,
                mean=jnp.array([1.0, -1]),
                covariance=np.diag([0.05, 0.001]),
            ),
            # A components
            fine.MultivariateNormalDistribution(
                dim_x=1,
                dim_y=1,
                mean=jnp.array([-0.8, -0.2]),
                covariance=np.diag([0.03, 0.001]),
            ),
            fine.MultivariateNormalDistribution(
                dim_x=1,
                dim_y=1,
                mean=jnp.array([-1.2, 0.0]),
                covariance=jnp.array(
                    [[var_x, jnp.sqrt(var_x * 0.2) * corr], [jnp.sqrt(var_x * 0.2) * corr, 0.2]]
                ),
            ),
            fine.MultivariateNormalDistribution(
                dim_x=1,
                dim_y=1,
                mean=jnp.array([-0.4, 0.0]),
                covariance=jnp.array(
                    [[var_x, -jnp.sqrt(var_x * 0.2) * corr], [-jnp.sqrt(var_x * 0.2) * corr, 0.2]]
                ),
            ),
        ],
    )
    sampler = fine.FineSampler(dist, mi_estimate_sample=mi_estimate_sample)

    return Task(
        sampler=sampler,
        task_id="1v1-AI",
        task_name="AI 1 × 1",
    )


def task_galaxy(
    speed=0.5,
    distance=3.0,
    mi_estimate_sample=_MC_MI_ESTIMATE_SAMPLE,
) -> Task:
    """The Galaxy distribution."""

    balls_mixt = fine.mixture(
        proportions=jnp.array([0.5, 0.5]),
        components=[
            fine.MultivariateNormalDistribution(
                covariance=samplers.canonical_correlation([0.0], additional_y=1),
                mean=jnp.array([x, x, x]) * distance / 2,
                dim_x=2,
                dim_y=1,
            )
            for x in [-1, 1]
        ],
    )

    base_sampler = fine.FineSampler(balls_mixt, mi_estimate_sample=mi_estimate_sample)
    a = jnp.array([[0, -1], [1, 0]])
    spiral = transforms.Spiral(a, speed=speed)

    sampler = samplers.TransformedSampler(base_sampler, transform_x=spiral)

    return Task(
        sampler=sampler,
        task_id=f"2v1-galaxy-{speed}-{distance}",
        task_name="Galaxy 2 × 1",
        task_params={
            "speed": speed,
            "distance": distance,
        },
    )


def task_waves(
    n_components=12,
    wave_amplitude=5.0,
    wave_frequency=3.0,
    mi_estimate_sample=_MC_MI_ESTIMATE_SAMPLE,
) -> Task:
    """The Waves distribution."""

    assert n_components > 0

    base_dist = fine.mixture(
        proportions=jnp.ones(n_components) / n_components,
        components=[
            fine.MultivariateNormalDistribution(
                covariance=jnp.diag(jnp.array([0.1, 1.0, 0.1])),
                mean=jnp.array([x, 0, x % 4]) * 1.5,
                dim_x=2,
                dim_y=1,
            )
            for x in range(n_components)
        ],
    )
    base_sampler = fine.FineSampler(base_dist, mi_estimate_sample=mi_estimate_sample)
    aux_sampler = samplers.TransformedSampler(
        base_sampler,
        transform_x=lambda x: x
        + jnp.array([wave_amplitude, 0.0]) * jnp.sin(wave_frequency * x[1]),
    )
    sampler = samplers.TransformedSampler(
        aux_sampler, transform_x=lambda x: jnp.array([0.1 * x[0] - 0.8, 0.5 * x[1]])
    )

    return Task(
        sampler=sampler,
        task_id=f"2v1-waves-{n_components}-{wave_amplitude}-{wave_frequency}",
        task_name="Waves 2 × 1",
        task_params={
            "n_components": n_components,
            "wave_amplitude": wave_amplitude,
            "wave_frequency": wave_frequency,
        },
    )
