import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import yaml

import bmi
from bmi.plot_utils.subplots_from_axsize import subplots_from_axsize
from _common_figure_utils import (
    ESTIMATORS,
    read_results, format_axs, plot_mi, scale_tasks
)


def spiral_task(speed: float, correlation: float) -> bmi.benchmark.Task:
    # Correlation cor(X1, Y) is non-zero.
    # We have cor(X1, X2) = cor(X2, Y) = 0.
    covariance = np.eye(3)
    covariance[0, 2] = correlation
    covariance[2, 0] = correlation

    generator = bmi.transforms.so_generator(2, 0, 1)

    sampler = bmi.samplers.TransformedSampler(
        base_sampler=bmi.samplers.SplitMultinormal(dim_x=2, dim_y=1, covariance=covariance),
        transform_x=bmi.transforms.Spiral(generator=generator,speed=speed),
    )
    return bmi.benchmark.Task(
        sampler=sampler,
        task_params={
            "correlation": correlation,
            "speed": speed,
        },
        task_name=f"Speed {speed:.2f}",
        task_id=f"spiral-{correlation}-{speed}",
    )


# === CONFIG ===

# ESTIMATORS from _common_figure_utils

TASK_LIST = [
    spiral_task(speed, correlation)
    for speed in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5]
    for correlation in [0.8]
]
TASKS = scale_tasks({task.id: task for task in TASK_LIST})

N_SAMPLES = [10_000]
SEEDS = list(range(5))


# === WORKDIR ===
workdir: "generated/figure_spiral/"


# === RULES ===
rule all:
    input:
        'figures/performance.pdf', 'figures/visualisation.pdf'

def find_speed(task_id: str) -> float:
    for task in TASK_LIST:
        if task.id == task_id:
            return task.params["speed"]
    raise KeyError(f"Task ID {task_id} not found.")

rule plot_performance:
    input: 'results.csv'
    output: 'figures/performance.pdf'
    run:
        results = read_results(str(input), max_mi_estimate=3.0)
        results['speed'] = results['task_id'].apply(find_speed)

        fig, ax = plt.subplots(figsize=(5, 3))
        plot_mi(ax, results, 'speed', x_label="Speed", plot_std=True)
        ax.legend(bbox_to_anchor=(1.1, 1.05), frameon=False)
        fig.tight_layout()
        fig.savefig(str(output))


include: "_core_rules.smk"


rule plot_spiral_visalisation:
    output: 'figures/visualisation.pdf'
    script:
        "scripts/plot_spiral.py"
