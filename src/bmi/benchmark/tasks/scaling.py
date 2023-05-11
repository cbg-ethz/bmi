from typing import Literal, Optional

import numpy as np
import sklearn.preprocessing as preproc

import bmi.samplers as samplers
from bmi.benchmark.task import Task


def _rescale(x) -> np.ndarray:
    """Wrapper around SciKit's function,
    as we need 64-bit precision -- otherwise
    a warning is raised.
    """
    x_new = np.asarray(x, dtype=np.float64)
    return preproc.scale(x_new)


def transform_rescale(
    base_task: Task,
    task_name: Optional[str] = None,
    task_id: Optional[str] = None,
) -> Task:
    base_sampler = base_task.sampler

    sampler = samplers.TransformedSampler(
        base_sampler=base_sampler,
        transform_x=_rescale,
        transform_y=_rescale,
        vectorise=False,
    )

    return Task(
        sampler=sampler,
        task_id=task_id or f"rescale-{base_task.id}",
        task_name=task_name or f"Rescale @ {base_task.name}",
        task_params=base_task.params,
    )


def _quantile_transform_wrapper(
    n_quantiles: int, distribution: Literal["uniform", "normal"], random_state: int
):
    def _inner_func(x) -> np.ndarray:
        return preproc.quantile_transform(
            x,
            output_distribution=distribution,
            random_state=random_state,
            n_quantiles=n_quantiles,
        )

    return _inner_func


def transform_uniformise(
    base_task: Task,
    task_name: Optional[str] = None,
    n_quantiles: int = 100,
    random_seed: int = 0,
) -> Task:
    """

    Args:
        n_quantiles: number of quantiles to be used
        random_seed: random seed for the preprocessing method
    """
    transform = _quantile_transform_wrapper(
        n_quantiles=n_quantiles, distribution="uniform", random_state=random_seed
    )

    base_sampler = base_task.sampler

    sampler = samplers.TransformedSampler(
        base_sampler=base_sampler,
        transform_x=transform,
        transform_y=transform,
        vectorise=False,
    )

    return Task(
        sampler=sampler,
        task_id=f"uniformise-{base_task.id}",
        task_name=task_name or f"Uniformise @ {base_task.name}",
        task_params=base_task.params | {"n_quantiles": n_quantiles},
    )


def transform_gaussianise(
    base_task: Task,
    task_name: Optional[str] = None,
    n_quantiles: int = 100,
    random_seed: int = 0,
) -> Task:
    transform = _quantile_transform_wrapper(
        n_quantiles=n_quantiles, distribution="normal", random_state=random_seed
    )

    base_sampler = base_task.sampler

    sampler = samplers.TransformedSampler(
        base_sampler=base_sampler,
        transform_x=transform,
        transform_y=transform,
        vectorise=False,
    )

    return Task(
        sampler=sampler,
        task_id=f"gaussianise-{base_task.id}",
        task_name=task_name or f"Gaussianise @ {base_task.name}",
        task_params=base_task.params | {"n_quantiles": n_quantiles},
    )
