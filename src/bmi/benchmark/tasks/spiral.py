from typing import Optional

import bmi.samplers as samplers
import bmi.transforms._rotate as rt
from bmi.benchmark.task import Task


def transform_spiral_task(
    base_task: Task,
    speed: float = 1 / 3,
    task_name: Optional[str] = None,
) -> Task:
    assert base_task.dim_x > 1, "dim_x has to be at least 2 for rotation to exist"
    assert base_task.dim_y > 2, "dim_y has to be at least 3 for rotation to exist"

    base_sampler = base_task.sampler

    generator_x = rt.so_generator(base_task.dim_x, 0, 1)
    generator_y = rt.so_generator(base_task.dim_y, 1, 2)

    transform_x = rt.Spiral(generator=generator_x, speed=speed)
    transform_y = rt.Spiral(generator=generator_y, speed=speed)

    spiral_sampler = samplers.TransformedSampler(
        base_sampler=base_sampler,
        transform_x=transform_x,
        transform_y=transform_y,
    )

    return Task(
        sampler=spiral_sampler,
        task_id=f"spiral-{base_task.id}",
        task_name=task_name or f"Spiral @ {base_task.name}",
        task_params=base_task.params | {"speed": speed},
    )
