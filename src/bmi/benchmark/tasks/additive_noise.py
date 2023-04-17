import bmi.samplers.api as samplers
from bmi.benchmark.task import Task


def task_additive_noise(
    epsilon: float,
) -> Task:
    sampler = samplers.AdditiveUniformSampler(epsilon=epsilon)

    return Task(
        sampler=sampler,
        task_id=f"1v1-additive-{epsilon}",
        task_name=f"Uniform 1 × 1 (additive noise={epsilon})",
        task_params={
            "epsilon": epsilon,
        },
    )
