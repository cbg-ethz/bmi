from typing import Callable, Iterable

import jax.numpy as jnp
from jax.scipy.linalg import expm

import bmi.benchmark.tasks.one_dimensional as od
import bmi.samplers.api as samplers
import bmi.transforms.rotate as rt
from bmi.benchmark.core import Task, generate_task
from bmi.interface import ISampler

DIMS: list[int] = [3, 5, 25]

# *** The base sampler ***


def create_base_sparse_sampler(dim: int) -> ISampler:
    covariance = samplers.parametrised_correlation_matrix(
        dim_x=dim,
        dim_y=dim,
        k=2,
        correlation=0.8,
        correlation_x=0.1,
        correlation_y=0.1,
    )
    return samplers.SplitMultinormal(
        dim_x=dim,
        dim_y=dim,
        covariance=covariance,
    )


# *** Uniform margins ***
def create_uniform_margins_sampler(dim: int) -> ISampler:
    return samplers.TransformedSampler(
        base_sampler=create_base_sparse_sampler(dim=dim),
        transform_x=od.normal_cdf,
        transform_y=od.normal_cdf,
        vectorise=True,
    )


def generate_uniform_margins_tasks(n_seeds: int, n_samples: int) -> Iterable[Task]:
    for dim in DIMS:
        yield generate_task(
            sampler=create_uniform_margins_sampler(dim=dim),
            seeds=range(n_seeds),
            n_samples=n_samples,
            task_id=f"uniform-margins-sparse-interactions-{dim}-vs-{dim}",
        )


# *** Half-cube sampler ***


def create_half_cube_sampler(dim: int) -> ISampler:
    return samplers.TransformedSampler(
        base_sampler=create_base_sparse_sampler(dim=dim),
        transform_x=od.half_cube,
        transform_y=od.half_cube,
        vectorise=True,
    )


def generate_half_cube_tasks(n_seeds: int, n_samples: int) -> Iterable[Task]:
    for dim in DIMS:
        yield generate_task(
            sampler=create_half_cube_sampler(dim),
            seeds=range(n_seeds),
            n_samples=n_samples,
            task_id=f"half_cube-sparse-interactions-{dim}-vs-{dim}",
        )


# *** Spiral ***


def create_spiral_sampler(dim: int, base_sampler: ISampler, speed: float) -> ISampler:
    if dim < 3:
        raise ValueError(f"dim must be at least 3, was {dim}")
    x_generator = rt.so_generator(dim, 0, 1)
    y_generator = rt.so_generator(dim, 1, 2)

    x_transform = rt.Spiral(generator=x_generator, speed=speed)
    y_transform = rt.Spiral(generator=y_generator, speed=speed)

    return samplers.TransformedSampler(
        base_sampler=base_sampler,
        transform_x=x_transform,
        transform_y=y_transform,
    )


def generate_spiral_tasks(*, n_samples: int, n_seeds: int, speed: float = 1 / 3) -> Iterable[Task]:
    for dim in DIMS:
        # Transform Gaussian sampler
        yield generate_task(
            sampler=create_spiral_sampler(
                dim=dim, base_sampler=create_base_sparse_sampler(dim=dim), speed=speed
            ),
            n_samples=n_samples,
            seeds=range(n_seeds),
            task_params=dict(speed=speed),
            task_id=f"spiral-gaussian-{dim}-{dim}",
        )
        # Transform uniform sampler
        yield generate_task(
            sampler=create_spiral_sampler(
                dim=dim, base_sampler=create_uniform_margins_sampler(dim=dim), speed=speed
            ),
            n_samples=n_samples,
            seeds=range(n_seeds),
            task_params=dict(speed=speed),
            task_id=f"spiral-uniform-margins-{dim}-{dim}",
        )


# *** Rotated "cube" (uniform margins)

_RotationMapping = Callable[[jnp.ndarray], jnp.ndarray]


def generate_rotations(dim: int) -> tuple[_RotationMapping, _RotationMapping]:
    def transform_x(x):
        # x.shape = (n_samples, dim_x)
        so_gen = (
            2 * rt.so_generator(dim, 0, 1)
            - 1 * rt.so_generator(dim, 1, 2)
            - 1 * rt.so_generator(dim, 0, 2)
        )
        rot_m = expm(so_gen)
        return x @ rot_m

    def transform_y(x):
        # x.shape = (n_samples, dim_y)
        so_gen = 1 * rt.so_generator(dim, 0, 1) + 2 * rt.so_generator(dim, 1, 2)
        rot_m = expm(so_gen)
        return x @ rot_m

    return transform_x, transform_y


def create_rotated_uniform_margins_sampler(dim: int) -> ISampler:
    transform_x, transform_y = generate_rotations(dim)
    return samplers.TransformedSampler(
        base_sampler=create_uniform_margins_sampler(dim=dim),
        transform_x=transform_x,
        transform_y=transform_y,
        vectorise=False,
    )


def generate_rotated_uniform_margins_tasks(*, n_seeds: int, n_samples: int) -> Iterable[Task]:
    for dim in DIMS:
        yield generate_task(
            sampler=create_rotated_uniform_margins_sampler(dim),
            seeds=range(n_seeds),
            n_samples=n_samples,
            task_id=f"rotated-uniform-margins-{dim}-{dim}",
        )


# *** "Wiggly" diffeomorphisms ***


def wiggle(x, freqs, scales=None, phases=None):
    if scales is None:
        scales = 1 / jnp.abs(freqs) / len(freqs) / 1.1

    if phases is None:
        phases = jnp.full_like(freqs, 1 + 2 * freqs.max())

    assert (freqs * scales).sum() < 1
    assert len(freqs.shape) == 1
    assert freqs.shape == scales.shape
    assert freqs.shape == phases.shape
    n = freqs.shape[0]
    shape = (n, *(len(x.shape) * [1]))

    xb = x.reshape(1, *x.shape)
    fb = freqs.reshape(*shape)
    sb = scales.reshape(*shape)
    pb = phases.reshape(*shape)

    return x + (sb * jnp.sin(xb * fb + pb)).sum(axis=0)


def wiggle_x(x):
    # x.shape = (n_samples, 3)

    x0_new = wiggle(x[..., 0], freqs=jnp.array([1.4, 2.3, 4.3, 5.1]))
    x1_new = od.normal_cdf(x[..., 1])
    x2_new = wiggle(x[..., 2], freqs=jnp.array([0.5, 3.7, 4.8, 7.9]), phases=jnp.arange(4))

    x_new = jnp.stack([x0_new, x1_new, x2_new], axis=-1)

    so_gen = (
        2 * rt.so_generator(3, 0, 1) - 1 * rt.so_generator(3, 1, 2) - 1 * rt.so_generator(3, 0, 2)
    )
    rot_m = expm(so_gen)

    return x_new @ rot_m


def wiggle_y(x):
    # x.shape = (n_samples, 3)

    x0_new = wiggle(x[..., 0], freqs=jnp.array([0.9, 2.1, 2.5, 5.7]))
    x1_new = wiggle(x[..., 1], freqs=jnp.array([1.8, 3.7, 5.2, 6.6]), phases=jnp.arange(4))
    x2_new = od.normal_cdf(x[..., 2])

    x_new = jnp.stack([x0_new, x1_new, x2_new], axis=-1)

    so_gen = 1 * rt.so_generator(3, 0, 1) + 2 * rt.so_generator(3, 1, 2)
    rot_m = expm(so_gen)

    return x_new @ rot_m


def create_wiggly_sampler() -> ISampler:
    return samplers.TransformedSampler(
        base_sampler=create_base_sparse_sampler(dim=3),
        transform_x=wiggle_x,
        transform_y=wiggle_y,
        vectorise=False,
    )


def generate_wiggly_task(*, n_seeds: int, n_samples: int) -> Task:
    return generate_task(
        sampler=create_wiggly_sampler(),
        n_samples=n_samples,
        seeds=range(n_seeds),
        task_id="wiggly-sparse-3-3",
    )


# *** The main function generating all the tasks ***


def generate_tasks(n_seeds: int, n_samples: int) -> Iterable[Task]:
    yield from generate_uniform_margins_tasks(n_seeds=n_seeds, n_samples=n_samples)
    yield from generate_half_cube_tasks(n_seeds=n_seeds, n_samples=n_samples)
    yield from generate_spiral_tasks(n_seeds=n_seeds, n_samples=n_samples)
    yield from generate_rotated_uniform_margins_tasks(n_seeds=n_seeds, n_samples=n_samples)
    yield generate_wiggly_task(n_seeds=n_seeds, n_samples=n_samples)
