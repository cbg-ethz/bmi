from dataclasses import asdict
from typing import Optional

import numpy as np

import bmi.samplers as samplers
from bmi.benchmark.task import Task


def task_multinormal_lvm(
    dim_x: int,
    dim_y: int,
    n_interacting: int,
    alpha: float,
    lambd: float,
    beta: float = 0.0,
    eta: Optional[float] = None,
    task_name: Optional[str] = None,
) -> Task:
    eta = eta if eta is not None else lambd

    params = samplers.GaussianLVMParametrization(
        dim_x=dim_x,
        dim_y=dim_y,
        n_interacting=n_interacting,
        alpha=alpha,
        lambd=lambd,
        beta_x=beta,
        eta_x=eta,
    )

    sampler = samplers.SplitMultinormal(
        dim_x=dim_x,
        dim_y=dim_y,
        covariance=params.correlation,
    )

    task_id = (
        f"multinormal-lvm-{dim_x}-{dim_y}-{n_interacting}"
        f"-{alpha:.5f}-{lambd:.5f}-{beta:.5f}-{eta:.5f}"
    )

    return Task(
        sampler=sampler,
        task_id=task_id,
        task_name=task_name or f"Multinormal {dim_x} × {dim_y} (LVM)",
        task_params=asdict(params),
    )


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
        task_id=f"multinormal-dense-{dim_x}-{dim_y}-{off_diag}",
        task_name=task_name or f"Multinormal {dim_x} × {dim_y} (dense)",
        task_params={
            "off_diag": off_diag,
            "covariance": covariance.tolist(),
        },
    )


def task_multinormal_sparse(
    dim_x: int,
    dim_y: int,
    n_interacting: int = 2,
    strength: float = 2.0,
    task_name: Optional[str] = None,
) -> Task:
    task_base = task_multinormal_lvm(
        dim_x=dim_x,
        dim_y=dim_y,
        n_interacting=n_interacting,
        alpha=0.0,
        lambd=strength,
        beta=0.0,
        eta=strength,
    )

    task_id = f"multinormal-sparse-{dim_x}-{dim_y}" f"-{n_interacting}-{strength}"

    return Task(
        sampler=task_base.sampler,
        task_id=task_id,
        task_name=task_name or f"Multinormal {dim_x} × {dim_y} (sparse)",
        task_params={
            "n_interacting": n_interacting,
            "strength": strength,
        },
    )


def task_multinormal_2pair(
    dim_x: int,
    dim_y: int,
    strength: float = 2.0,
    task_name: Optional[str] = None,
) -> Task:
    task_name = task_name or f"Multinormal {dim_x} × {dim_y} (2-pair)"
    return task_multinormal_sparse(
        dim_x=dim_x,
        dim_y=dim_y,
        n_interacting=2,
        strength=strength,
        task_name=task_name,
    )
