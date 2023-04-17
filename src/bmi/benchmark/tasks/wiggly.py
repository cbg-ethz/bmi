from typing import Optional

import jax.numpy as jnp

import bmi.samplers as samplers
from bmi.benchmark.task import Task


def wiggly_x(x: float) -> float:
    return x + 0.4 * jnp.sin(1.0 * x) + 0.2 * jnp.sin(1.7 * x + 1) + 0.03 * jnp.sin(3.3 * x - 2.5)


def wiggly_y(x: float) -> float:
    return (
        x - 0.4 * jnp.sin(0.4 * x) + 0.17 * jnp.sin(1.3 * x + 3.5) + 0.02 * jnp.sin(4.3 * x - 2.5)
    )


def transform_wiggly_task(
    base_task: Task,
    task_name: Optional[str] = None,
) -> Task:
    base_sampler = base_task.sampler

    sampler = samplers.TransformedSampler(
        base_sampler=base_sampler,
        transform_x=wiggly_x,
        transform_y=wiggly_y,
    )

    return Task(
        sampler=sampler,
        task_id=f"wiggly-{base_task.id}",
        task_name=task_name or f"Wiggly @ {base_task.name}",
        task_params=base_task.params,
    )
