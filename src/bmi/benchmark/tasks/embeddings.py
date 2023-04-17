import bmi.samplers as samplers
from bmi.benchmark.task import Task
from bmi.transforms import swissroll2d


def transform_swissroll_task(
    base_task: Task,
    task_name: str,  # dims change, better force a new name
) -> Task:
    assert base_task.dim_x == 1

    sampler = samplers.TransformedSampler(
        base_sampler=base_task.sampler,
        transform_x=swissroll2d,
        add_dim_x=1,
    )

    return Task(
        sampler=sampler,
        task_id=f"swissroll_x-{base_task.id}",
        task_name=task_name,
        task_params=base_task.params,
    )
