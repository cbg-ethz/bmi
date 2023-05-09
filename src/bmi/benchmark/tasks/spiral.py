from typing import Optional

import bmi.samplers as samplers
import bmi.transforms._rotate as rt
from bmi.benchmark.task import Task


def transform_spiral_task(
    base_task: Task,
    speed: float = 1.0,
    task_name: Optional[str] = None,
    normalize_speed: bool = True,
) -> Task:
    assert base_task.dim_x > 1, "dim_x has to be at least 2 for rotation to exist"
    assert base_task.dim_y > 2, "dim_y has to be at least 3 for rotation to exist"

    base_sampler = base_task.sampler

    generator_x = rt.so_generator(base_task.dim_x, 0, 1)
    generator_y = rt.so_generator(base_task.dim_y, 1, 2)

    if normalize_speed:
        speed_x = speed / base_task.dim_x
        speed_y = speed / base_task.dim_y
    else:
        speed_x = speed
        speed_y = speed

    transform_x = rt.Spiral(generator=generator_x, speed=speed_x)
    transform_y = rt.Spiral(generator=generator_y, speed=speed_y)

    spiral_sampler = samplers.TransformedSampler(
        base_sampler=base_sampler,
        transform_x=transform_x,
        transform_y=transform_y,
    )

    return Task(
        sampler=spiral_sampler,
        task_id=f"spiral-{base_task.id}",
        task_name=task_name or f"Spiral @ {base_task.name}",
        task_params=base_task.params | {"speed": speed, "normalize_speed": normalize_speed},
    )
