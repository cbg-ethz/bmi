import pandas as pd

import jax.numpy as jnp

import bmi.benchmark.tasks.multinormal as multinormal
import bmi.benchmark.tasks.student as student
from bmi.benchmark.tasks.power import transform_power_task as powerise

from bmi.plot_utils.subplots_from_axsize import subplots_from_axsize
from _common_figure_utils import (
    ESTIMATORS,
    read_results, format_axs, plot_mi, scale_tasks
)


# === CONFIG ===

# ESTIMATORS from _common_figure_utils

TASKS = scale_tasks({
    task.id: task for task in [
        powerise(multinormal.task_multinormal_sparse(3, 3), alpha=alpha)
        for alpha in jnp.linspace(1, 5, 10)
    ] + [
        student.task_student_sparse(dim_x=3, dim_y=3, df=df)
        for df in range(1, 21, 2)
    ]
})

N_SAMPLES = [10_000]
SEEDS = list(range(5))


# === WORKDIR ===
workdir: "generated/figure_tails/"


# === RULES ===
rule all:
    input: 'figures/tails.pdf'

rule figure_tails:
    input: 'results.csv'
    output: 'figures/tails.pdf'
    run:
        results = read_results(str(input))
        fig, axs = subplots_from_axsize(
            axsize=(2.0, 1.5), ncols=2,
            left=0.6, right=1.75,
        )
        format_axs(axs)
        plot_mi(axs[0], results, 'alpha')
        plot_mi(axs[1], results, 'dof')
        axs[1].legend(bbox_to_anchor=(1.1, 1.05))
        fig.savefig(str(output))

include: "_core_rules.smk"
