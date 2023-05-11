from typing import Optional

import numpy as np

import bmi.samplers as samplers
from bmi.benchmark.task import Task


def task_student_identity(
    dim_x: int,
    dim_y: int,
    df: int,
    task_name: Optional[str] = None,
) -> Task:
    sampler = samplers.SplitStudentT(
        dim_x=dim_x,
        dim_y=dim_y,
        df=df,
        dispersion=np.eye(dim_x + dim_y),
    )

    task_id = f"student-identity-{dim_x}-{dim_y}-{df}"
    return Task(
        sampler=sampler,
        task_id=task_id,
        task_name=task_name or f"Student-t {dim_x} × {dim_y} (dof={df})",
        task_params={
            "dof": df,
        },
    )


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
            "dispersion": dispersion.tolist(),
        },
    )


def task_student_sparse(
    dim_x: int,
    dim_y: int,
    df: int,
    n_interacting: int = 2,
    strength: float = 2.0,
    task_name: Optional[str] = None,
) -> Task:
    params = samplers.GaussianLVMParametrization(
        dim_x=dim_x,
        dim_y=dim_y,
        n_interacting=n_interacting,
        alpha=0.0,
        lambd=strength,
        beta_x=0.0,
        eta_x=strength,
    )

    sampler = samplers.SplitStudentT(
        dim_x=dim_x,
        dim_y=dim_y,
        dispersion=params.correlation,
        df=df,
    )

    task_id = f"student-sparse-{dim_x}-{dim_y}-{df}" f"-{n_interacting}-{strength}"

    return Task(
        sampler=sampler,
        task_id=task_id,
        task_name=task_name or f"Student-t {dim_x} × {dim_y} (dof={df}, sparse)",
        task_params={
            "dof": df,
            "n_interacting": n_interacting,
            "strength": strength,
        },
    )


def task_student_2pair(
    dim_x: int,
    dim_y: int,
    df: int,
    strength: float = 2.0,
    task_name: Optional[str] = None,
) -> Task:
    task_name = task_name or f"Student-t {dim_x} × {dim_y} (dof={df}, 2-pair)"
    return task_student_sparse(
        dim_x=dim_x,
        dim_y=dim_y,
        df=df,
        n_interacting=2,
        strength=strength,
        task_name=task_name,
    )
