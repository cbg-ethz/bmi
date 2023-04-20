from typing import Optional

import jax.numpy as jnp

import bmi.samplers as samplers
from bmi.benchmark.task import Task


def transform_asinh_task(
    base_task: Task,
    task_name: Optional[str] = None,
) -> Task:
    base_sampler = base_task.sampler

    sampler = samplers.TransformedSampler(
        base_sampler=base_sampler,
        transform_x=jnp.arcsinh,
        transform_y=jnp.arcsinh,
    )

    return Task(
        sampler=sampler,
        task_id=f"asinh-{base_task.id}",
        task_name=task_name or f"Asinh @ {base_task.name}",
        task_params=base_task.params,
    )
