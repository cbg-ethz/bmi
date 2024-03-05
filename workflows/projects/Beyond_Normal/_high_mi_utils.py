"""Utilities for generating tasks
for high mutual information plot."""

from typing import Callable, Optional

import numpy as np

import bmi

# *************************************************
# *************** Utility functions ***************
# *************************************************


def binsearch(
    func: Callable[[float], float],
    target: float,
    min_value: float,
    max_value: float,
    delta: float = 1e-4,
    max_iter: int = 5_000,
    seed: int = 0,
) -> Optional[float]:
    """Solves func(x) = target for x assuming that func is strictly increasing
    and using a mixture of binary search and random sampling.

    The returned type is Optional[float] --- we will propagate None
    whenever we can't generate a value.
    This pattern will appear also in the later examples
    (essentially Maybe monad).
    """
    rng = np.random.default_rng(seed)

    min_val = min_value
    max_val = max_value
    mid = 0.5 * (min_val + max_val)
    value = -1e9

    for _ in range(max_iter):
        try:
            value = func(mid)

            if abs(value - target) < delta:
                break
            elif value > target:
                max_val = mid
            else:
                min_val = mid

            mid = 0.5 * (max_val + min_val)
        except ValueError:
            mid = rng.uniform(min_value, max_value)

    if abs(value - target) < delta:
        return mid
    else:
        return None


# ******************************************************
# *************** Basic Gaussian sampler ***************
# ******************************************************


def get_sparse_gaussian_sampler(correlation: float, dim: int) -> bmi.ISampler:
    """Generates the sparse Gaussian sampler we will use."""
    covariance = bmi.samplers.parametrised_correlation_matrix(
        dim_x=dim,
        dim_y=dim,
        k=2,
        correlation=correlation,
        correlation_x=0.0,
        correlation_y=0.0,
    )
    return bmi.samplers.SplitMultinormal(
        dim_x=dim,
        dim_y=dim,
        covariance=covariance,
    )


def mi_sparse_gaussian(correlation: float, dim: int) -> float:
    """The mutual information of the sparse Gaussian sampler."""
    sampler = get_sparse_gaussian_sampler(correlation=correlation, dim=dim)
    return sampler.mutual_information()


def generate_sparse_gaussian_task(
    mi: float,
    family_name: str,
    dim: int,
    noise: float = 0.1,
) -> Optional[bmi.Task]:
    corr = binsearch(
        lambda c: mi_sparse_gaussian(c, dim=dim),
        target=mi,
        min_value=0,
        max_value=1,
    )
    # Maybe monad behaviour without actual monads
    if corr is None:
        return None

    sampler = get_sparse_gaussian_sampler(correlation=corr, dim=dim)

    return bmi.benchmark.Task(
        sampler=sampler,
        task_params={
            "correlation": corr,
            "desired_mi": mi,
            "true_mi": sampler.mutual_information(),
            "dim": dim,
            "family_name": family_name,
        },
        task_id=f"sparse-gaussian-{mi:.3f}",
        task_name=f"Gaussian with {mi:.2f} MI",
    )


# **********************************************************
# *************** Spiralled Gaussian sampler ***************
# **********************************************************


def generate_spiral_task(
    mi: float,
    family_name: str,
    dim: int,
    noise: float = 0.1,
    speed: float = 1 / 3,
) -> Optional[bmi.Task]:
    base_gaussian_task = generate_sparse_gaussian_task(
        mi=mi,
        dim=dim,
        noise=noise,
        family_name=family_name,
    )
    # Maybe monad behaviour without monads
    if base_gaussian_task is None:
        return None

    task = bmi.benchmark.tasks.transform_spiral_task(
        base_task=base_gaussian_task,
        speed=speed,
        task_name=f"Spiral with {mi:.2f} MI",
    )
    return task


# ***********************************************************
# *************** Half-cubed Gaussian sampler ***************
# ***********************************************************


def generate_half_cube_task(
    mi: float, family_name: str, dim: int, noise: float = 0.1
) -> Optional[bmi.Task]:
    base_gaussian_task = generate_sparse_gaussian_task(
        mi=mi,
        dim=dim,
        noise=noise,
        family_name=family_name,
    )
    # Maybe monad behaviour without monads
    if base_gaussian_task is None:
        return None

    task = bmi.benchmark.tasks.transform_half_cube_task(
        base_task=base_gaussian_task,
        task_name=f"Half-cube with {mi:.2f} MI",
    )
    return task
