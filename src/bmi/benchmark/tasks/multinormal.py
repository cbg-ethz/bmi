from typing import Optional

import numpy as np

import bmi.samplers.api as samplers
from bmi.benchmark.task import Task


def task_multinormal_dense(
    dim_x: int,
    dim_y: int,
    off_diag: float = 0.5,
    task_name: Optional[str] = None,
) -> Task:
    covariance = np.eye(dim_x + dim_y) * (1 - off_diag) + off_diag

    sampler = samplers.SplitMultinormal(
        dim_x=dim_x,
        dim_y=dim_y,
        covariance=covariance,
    )

    return Task(
        sampler=sampler,
        task_id=f"multinormal-dense-{dim_x}-{dim_y}",
        task_name=task_name or f"Multinormal (dense) {dim_x} × {dim_y}",
        task_params={
            "off_diag": off_diag,
            "covariance": covariance,
        },
    )


def task_multinormal_sparse(
    dim_x: int,
    dim_y: int,
    n_interacting: int = 2,
    correlation_signal: float = 0.8,
    correlation_noise: float = 0.1,
    task_name: Optional[str] = None,
) -> Task:
    covariance = samplers.parametrised_correlation_matrix(
        dim_x=dim_x,
        dim_y=dim_y,
        k=n_interacting,
        correlation=correlation_signal,
        correlation_x=correlation_noise,
        correlation_y=correlation_noise,
    )

    sampler = samplers.SplitMultinormal(
        dim_x=dim_x,
        dim_y=dim_y,
        covariance=covariance,
    )

    task_id = (
        f"multinormal-sparse-{dim_x}-{dim_y}"
        f"-{n_interacting}"
        f"-{correlation_signal}-{correlation_noise}"
    )

    return Task(
        sampler=sampler,
        task_id=task_id,
        task_name=task_name or f"Multinormal (sparse) {dim_x} × {dim_y}",
        task_params={
            "n_interacting": n_interacting,
            "correlation_signal": correlation_signal,
            "correlation_noise": correlation_noise,
            "covariance": covariance,
        },
    )
