from typing import Optional

import numpy as np

import bmi.samplers.api as samplers
from bmi.benchmark.task import Task


def task_student_dense(
    dim_x: int,
    dim_y: int,
    df: int,
    off_diag: float = 0.5,
    task_name: Optional[str] = None,
) -> Task:
    dispersion = np.eye(dim_x + dim_y) * (1 - off_diag) + off_diag

    sampler = samplers.SplitStudentT(
        dim_x=dim_x,
        dim_y=dim_y,
        dispersion=dispersion,
        df=df,
    )

    return Task(
        sampler=sampler,
        task_id=f"student-dense-{dim_x}-{dim_y}-{df}-{off_diag}",
        task_name=task_name or f"Student-t {dim_x} × {dim_y} (dof={df}, dense)",
        task_params={
            "dof": df,
            "off_diag": off_diag,
            "dispersion": dispersion,
        },
    )


def task_student_sparse(
    dim_x: int,
    dim_y: int,
    df: int,
    n_interacting: int = 2,
    correlation_signal: float = 0.8,
    correlation_noise: float = 0.1,
    task_name: Optional[str] = None,
) -> Task:
    dispersion = samplers.parametrised_correlation_matrix(
        dim_x=dim_x,
        dim_y=dim_y,
        k=n_interacting,
        correlation=correlation_signal,
        correlation_x=correlation_noise,
        correlation_y=correlation_noise,
    )

    sampler = samplers.SplitStudentT(
        dim_x=dim_x,
        dim_y=dim_y,
        dispersion=dispersion,
        df=df,
    )

    task_id = (
        f"student-sparse-{dim_x}-{dim_y}-{df}"
        f"-{n_interacting}"
        f"-{correlation_signal}-{correlation_noise}"
    )

    return Task(
        sampler=sampler,
        task_id=task_id,
        task_name=task_name or f"Student-t {dim_x} × {dim_y} (dof={df}, sparse)",
        task_params={
            "n_interacting": n_interacting,
            "dof": df,
            "correlation_signal": correlation_signal,
            "correlation_noise": correlation_noise,
            "dispersion": dispersion,
        },
    )
