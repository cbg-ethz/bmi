from typing import Optional

import bmi.samplers as samplers
from bmi.benchmark.task import Task
from bmi.transforms import normal_cdf


def transform_normal_cdf_task(
    base_task: Task,
    task_name: Optional[str] = None,
) -> Task:
    base_sampler = base_task.sampler

    sampler = samplers.TransformedSampler(
        base_sampler=base_sampler,
        transform_x=normal_cdf,
        transform_y=normal_cdf,
    )

    return Task(
        sampler=sampler,
        task_id=f"normal_cdf-{base_task.id}",
        task_name=task_name or f"Normal CDF @ {base_task.name}",
        task_params=base_task.params,
    )
