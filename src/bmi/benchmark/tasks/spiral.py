from typing import Optional

import numpy as np

import bmi.benchmark.core as core
import bmi.transforms.rotate as rt
from bmi.samplers.api import SplitMultinormal, TransformedSampler


def generate_spiral_invariance_task(
    correlation: float,
    n_points: int,
    n_seeds: int,
    speed: float,
    dim_x: int = 2,
    dim_y: int = 1,
    task_id: Optional[str] = None,
) -> core.Task:
    task_id = (
        task_id
        or f"spiral-cor_{correlation:.2f}-N_{n_points}-dim_{dim_x}_{dim_y}-speed_{speed:.2f}"
    )

    # First, we create a Gaussian sampler
    # We introduce the non-zero correlation
    # only between the first dimension of X and first dimension of Y
    cov = np.eye(dim_x + dim_y)
    cov[0, dim_x] = correlation
    cov[dim_x, 0] = correlation
    base_sampler = SplitMultinormal(dim_x=dim_x, dim_y=dim_y, covariance=cov)

    # Then, we apply the spiral diffeomorphism to the X variable
    generator = rt.so_generator(dim_x)
    spiral = rt.Spiral(
        generator=generator,
        speed=speed,
    )
    sampler = TransformedSampler(base_sampler=base_sampler, transform_x=spiral, vectorise=True)

    return core.generate_task(
        sampler=sampler,
        n_samples=n_points,
        seeds=list(range(n_seeds)),
        task_id=task_id,
    )
