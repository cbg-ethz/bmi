import jax.numpy as jnp
import numpy as np

import bmi.samplers as samplers
import bmi.transforms as transforms
from bmi.benchmark.task import Task
from bmi.samplers import bmm

_MC_MI_ESTIMATE_SAMPLE = 100_000


def task_x(
    gaussian_correlation=0.9,
    mi_estimate_sample=_MC_MI_ESTIMATE_SAMPLE,
) -> Task:
    """The X distribution."""

    dist = bmm.mixture(
        proportions=jnp.array([0.5, 0.5]),
        components=[
            bmm.MultivariateNormalDistribution(
                covariance=samplers.canonical_correlation([x * gaussian_correlation]),
                mean=jnp.zeros(2),
                dim_x=1,
                dim_y=1,
            )
            for x in [-1, 1]
        ],
    )
    sampler = bmm.FineSampler(dist, mi_estimate_sample=mi_estimate_sample)

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

    dist = bmm.mixture(
        proportions=jnp.full(6, fill_value=1 / 6),
        components=[
            # I components
            bmm.MultivariateNormalDistribution(
                dim_x=1,
                dim_y=1,
                mean=jnp.array([1.0, 0.0]),
                covariance=np.diag([0.01, 0.2]),
            ),
            bmm.MultivariateNormalDistribution(
                dim_x=1,
                dim_y=1,
                mean=jnp.array([1.0, 1]),
                covariance=np.diag([0.05, 0.001]),
            ),
            bmm.MultivariateNormalDistribution(
                dim_x=1,
                dim_y=1,
                mean=jnp.array([1.0, -1]),
                covariance=np.diag([0.05, 0.001]),
            ),
            # A components
            bmm.MultivariateNormalDistribution(
                dim_x=1,
                dim_y=1,
                mean=jnp.array([-0.8, -0.2]),
                covariance=np.diag([0.03, 0.001]),
            ),
            bmm.MultivariateNormalDistribution(
                dim_x=1,
                dim_y=1,
                mean=jnp.array([-1.2, 0.0]),
                covariance=jnp.array(
                    [[var_x, jnp.sqrt(var_x * 0.2) * corr], [jnp.sqrt(var_x * 0.2) * corr, 0.2]]
                ),
            ),
            bmm.MultivariateNormalDistribution(
                dim_x=1,
                dim_y=1,
                mean=jnp.array([-0.4, 0.0]),
                covariance=jnp.array(
                    [[var_x, -jnp.sqrt(var_x * 0.2) * corr], [-jnp.sqrt(var_x * 0.2) * corr, 0.2]]
                ),
            ),
        ],
    )
    sampler = bmm.FineSampler(dist, mi_estimate_sample=mi_estimate_sample)

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

    balls_mixt = bmm.mixture(
        proportions=jnp.array([0.5, 0.5]),
        components=[
            bmm.MultivariateNormalDistribution(
                covariance=samplers.canonical_correlation([0.0], additional_y=1),
                mean=jnp.array([x, x, x]) * distance / 2,
                dim_x=2,
                dim_y=1,
            )
            for x in [-1, 1]
        ],
    )

    base_sampler = bmm.FineSampler(balls_mixt, mi_estimate_sample=mi_estimate_sample)
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

    base_dist = bmm.mixture(
        proportions=jnp.ones(n_components) / n_components,
        components=[
            bmm.MultivariateNormalDistribution(
                covariance=jnp.diag(jnp.array([0.1, 1.0, 0.1])),
                mean=jnp.array([x, 0, x % 4]) * 1.5,
                dim_x=2,
                dim_y=1,
            )
            for x in range(n_components)
        ],
    )
    base_sampler = bmm.FineSampler(base_dist, mi_estimate_sample=mi_estimate_sample)
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


def task_concentric_multinormal(
    dim_x,
    n_components=3,
    mi_estimate_sample=_MC_MI_ESTIMATE_SAMPLE,
) -> Task:
    """Isotropic Gaussians with varying standard deviation."""

    assert n_components > 0

    dist = bmm.mixture(
        proportions=jnp.ones(n_components) / n_components,
        components=[
            bmm.MultivariateNormalDistribution(
                covariance=jnp.diag(jnp.array(dim_x * [i**2] + [0.0001])),
                mean=jnp.array(dim_x * [0.0] + [1.0 * i]),
                dim_x=dim_x,
                dim_y=1,
            )
            for i in range(1, 1 + n_components)
        ],
    )
    sampler = bmm.FineSampler(dist, mi_estimate_sample=mi_estimate_sample)

    return Task(
        sampler=sampler,
        task_id=f"{dim_x}v1-concentric_gaussians-{n_components}",
        task_name=f"Concentric {dim_x} × 1",
        task_params={
            "n_components": n_components,
        },
    )


def task_multinormal_sparse_w_inliers(
    dim_x,
    dim_y,
    n_interacting: int = 2,
    strength: float = 2.0,
    inlier_fraction: float = 0.2,
    mi_estimate_sample=_MC_MI_ESTIMATE_SAMPLE,
) -> Task:

    assert 0.0 <= inlier_fraction <= 1.0

    params = samplers.GaussianLVMParametrization(
        dim_x=dim_x,
        dim_y=dim_y,
        n_interacting=n_interacting,
        alpha=0.0,
        lambd=strength,
        beta_x=0.0,
        eta_x=strength,
    )

    signal_dist = bmm.MultivariateNormalDistribution(
        dim_x=dim_x,
        dim_y=dim_y,
        covariance=params.correlation,
    )

    noise_dist = bmm.ProductDistribution(
        dist_x=signal_dist.dist_x,
        dist_y=signal_dist.dist_y,
    )

    dist = bmm.mixture(
        proportions=jnp.array([1 - inlier_fraction, inlier_fraction]),
        components=[signal_dist, noise_dist],
    )

    sampler = bmm.FineSampler(dist, mi_estimate_sample=mi_estimate_sample)

    task_id = f"mult-sparse-w-inliers-{dim_x}-{dim_y}-{n_interacting}-{strength}-{inlier_fraction}"
    return Task(
        sampler=sampler,
        task_id=task_id,
        task_name=f"Multinormal {dim_x} × {dim_y} (sparse, {inlier_fraction:.0%} inliers)",
        task_params={
            "n_interacting": n_interacting,
            "strength": strength,
            "inlier_fraction": inlier_fraction,
        },
    )
