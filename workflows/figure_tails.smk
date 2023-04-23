import pandas as pd

import jax.numpy as jnp

import bmi.benchmark.tasks.multinormal as multinormal
import bmi.benchmark.tasks.student as student
from bmi.benchmark.tasks.power import transform_power_task as powerise

from common_estimators import ESTIMATORS
from common_plotting import read_results, prepare_fig_axs, plot_mi


# === CONFIG ===
#ESTIMATORS = dict(list(ESTIMATORS.items())[:1])

TASKS = {
    task.id: task for task in [
        powerise(multinormal.task_multinormal_sparse(3, 3), alpha=alpha)
        for alpha in jnp.linspace(1, 3, 31)
    ] + [
        student.task_student_sparse(dim_x=3, dim_y=3, df=df)
        for df in range(1, 21)
    ]
}

N_SAMPLES = [10000]
SEEDS = [0]


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
        fig, axs = prepare_fig_axs(ncols=2)
        plot_mi(axs[0], results, 'alpha')
        plot_mi(axs[1], results, 'dof')
        fig.savefig(str(output))

include: "_core_rules.smk"
