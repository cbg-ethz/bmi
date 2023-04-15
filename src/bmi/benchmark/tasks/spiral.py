from typing import Optional

import bmi.samplers.api as samplers
import bmi.transforms.rotate as rt
from bmi.benchmark.task import Task
from bmi.benchmark.tasks import multinormal


def transform_spiral_task(
    base_task,
    speed: float = 1 / 3,
    task_name: Optional[str] = None,
) -> Task:
    assert base_task.dim_x > 1, "dim_x has to be at least 2 for rotation to exist"
    assert base_task.dim_x > 2, "dim_y has to be at least 3 for rotation to exist"

    base_sampler = base_task.sampler

    x_generator = rt.so_generator(base_task.dim_x, 0, 1)
    y_generator = rt.so_generator(base_task.dim_y, 1, 2)

    x_transform = rt.Spiral(generator=x_generator, speed=speed)
    y_transform = rt.Spiral(generator=y_generator, speed=speed)

    spiral_sampler = samplers.TransformedSampler(
        base_sampler=base_sampler,
        transform_x=x_transform,
        transform_y=y_transform,
    )

    return Task(
        sampler=spiral_sampler,
        task_id=f"spiral-{base_task.id}",
        task_name=task_name or f"Spiral @ {base_task.name}",
        task_params=base_task.params | {"speed": speed},
    )


def task_spiral_multinormal_sparse(
    dim_x: int,
    dim_y: int,
    speed: float = 1 / 3,
    task_name: Optional[str] = None,
    **kwargs,
) -> Task:
    base_task = multinormal.task_multinormal_sparse(
        dim_x=dim_x,
        dim_y=dim_y,
        **kwargs,
    )

    return transform_spiral_task(
        base_task=base_task,
        speed=speed,
        task_name=task_name,
    )
