from typing import Optional

import jax.numpy as jnp

import bmi.samplers as samplers
from bmi.benchmark.task import Task


def transform_power_task(
    base_task: Task,
    alpha: float,
    task_name: Optional[str] = None,
) -> Task:
    base_sampler = base_task.sampler

    alpha = float(alpha)

    def pow(x):
        return jnp.sign(x) * jnp.power(jnp.abs(x), alpha)

    sampler = samplers.TransformedSampler(
        base_sampler=base_sampler,
        transform_x=pow,
        transform_y=pow,
        vectorise=False,
    )

    return Task(
        sampler=sampler,
        task_id=f"power_{alpha:.5f}-{base_task.id}",
        task_name=task_name or f"Power {alpha:.2f} @ {base_task.name}",
        task_params=base_task.params | {"alpha": alpha},
    )
