from typing import Optional

import bmi.samplers.api as samplers
from bmi.benchmark.task import Task
from bmi.transforms.normal_cdf import normal_cdf


def transform_normal_cdf_task(
    base_task,
    task_name: Optional[str] = None,
) -> Task:
    base_sampler = base_task.sampler

    sampler = samplers.TransformedSampler(
        base_sampler=base_sampler,
        transform_x=lambda x: normal_cdf(x),
        transform_y=lambda y: normal_cdf(y),
    )

    return Task(
        sampler=sampler,
        task_id=f"normal-cdf-{base_task.id}",
        task_name=task_name or f"Normal CDF @ {base_task.name}",
        task_params=base_task.params,
    )
