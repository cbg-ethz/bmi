from typing import Optional

from jax.scipy.linalg import expm

import bmi.samplers as samplers
import bmi.transforms._rotate as rt
from bmi.benchmark.task import Task


def transform_rotate_task(
    base_task: Task,
    task_name: Optional[str] = None,
) -> Task:
    """Note: we always use the same rotation."""
    dim_x = base_task.dim_x
    dim_y = base_task.dim_y

    assert dim_x > 2, "dim_x has to be at least 3 for rotation to exist"
    assert dim_y > 2, "dim_y has to be at least 3 for rotation to exist"

    def transform_x(x):
        # x.shape = (n_samples, dim_x)
        so_gen = (
            2 * rt.so_generator(dim_x, 0, 1)
            - 1 * rt.so_generator(dim_x, 1, 2)
            - 1 * rt.so_generator(dim_x, 0, 2)
        )
        rot_m = expm(so_gen)
        return x @ rot_m

    def transform_y(x):
        # x.shape = (n_samples, dim_y)
        so_gen = 1 * rt.so_generator(dim_y, 0, 1) + 2 * rt.so_generator(dim_y, 1, 2)
        rot_m = expm(so_gen)
        return x @ rot_m

    base_sampler = base_task.sampler

    spiral_sampler = samplers.TransformedSampler(
        base_sampler=base_sampler,
        transform_x=transform_x,
        transform_y=transform_y,
    )

    return Task(
        sampler=spiral_sampler,
        task_id=f"rotate-{base_task.id}",
        task_name=task_name or f"Rotate @ {base_task.name}",
        task_params=base_task.params,
    )
