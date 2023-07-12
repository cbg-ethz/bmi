"""Utilities for generating tasks for sparsity plots."""

from _high_mi_utils import binsearch

import bmi


def get_initial_alpha(mi: float, dim: int) -> float:
    def alpha_to_task(alpha):
        return bmi.benchmark.tasks.task_multinormal_lvm(
            dim_x=dim,
            dim_y=dim,
            n_interacting=0,
            alpha=alpha,
            lambd=0.0,
        )

    alpha_best = binsearch(
        lambda a: alpha_to_task(a).mutual_information, mi, min_value=0.0, max_value=10.0
    )
    assert alpha_best

    return alpha_best


def get_lambda_task(mi: float, dim: int, alpha: float) -> bmi.Task:
    def lambd_to_task(lambd):
        return bmi.benchmark.tasks.task_multinormal_lvm(
            dim_x=dim,
            dim_y=dim,
            n_interacting=dim,
            alpha=alpha,
            lambd=lambd,
        )

    lambd_best = binsearch(
        lambda lambd: lambd_to_task(lambd).mutual_information, mi, min_value=0.0, max_value=10.0
    )

    return lambd_to_task(lambd_best)


def get_n_interacting_task(mi: float, dim: int, n_interacting: int) -> bmi.Task:
    def lambd_to_task(lambd):
        return bmi.benchmark.tasks.task_multinormal_lvm(
            dim_x=dim,
            dim_y=dim,
            n_interacting=n_interacting,
            alpha=0.0,
            lambd=lambd,
        )

    lambd_best = binsearch(
        lambda lambd: lambd_to_task(lambd).mutual_information, mi, min_value=0.0, max_value=10.0
    )

    return lambd_to_task(lambd_best)
