import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import yaml

import bmi
from _common_figure_utils import ESTIMATORS, ESTIMATOR_NAMES, ESTIMATOR_COLORS


ESTIMATORS = dict(list(ESTIMATORS.items())[:2])

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
TASK_LIST = [
    spiral_task(speed, correlation)
    for speed in [0.0, 0.1] #, 0.3, 0.5, 1.0]
    for correlation in [0.8]
]
TASKS = {task.id: task for task in TASK_LIST}

N_SAMPLES = [1000]
SEEDS = [0]


# === WORKDIR ===
workdir: "generated/figure_spiral/"


# === RULES ===
rule all:
    input:
        'figures/performance.pdf', 'figures/visualisation.pdf'

rule plot_performance:
    input: 'results.csv'
    output: 'figures/performance.pdf'
    run:
        data = pd.read_csv(str(input))
        fig, ax = plt.subplots()

        #ax.hlines([data['mi_true'].mean()], min(SPEEDS), max(SPEEDS), colors="k",linestyles="--")
        data['speed'] = data['task_params'].apply(lambda x: yaml.safe_load(x)['speed'])

        print(data['task_params'].head())

        mi_true = data['mi_true'].mean()

        ax.hlines(
            mi_true,
            xmin=data['speed'].min(),
            xmax=data['speed'].max(),
            linestyles="dashed",
            label="True",
            colors="black",
        )

        means = data.groupby(["task_id", "estimator_id"]).mean().reset_index()

        for estimator_id, group in means.groupby("estimator_id"):
            ax.scatter(
                group['speed'],
                group['mi_estimate'],
                label=ESTIMATOR_NAMES[estimator_id],
                color=ESTIMATOR_COLORS[estimator_id],
                alpha=0.3,
            )
            #sns.regplot(
            #    data=group,
            #    x="speed",
            #    y="mi_estimate",
            #    ax=ax,
            #    x_estimator=np.mean,
            #    lowess=True,
            #    label=ESTIMATOR_NAMES[estimator_id],
            #)

        ax.set_xlabel("Speed")
        ax.set_ylabel("Mutual Information estimate")

        ax.legend()

        fig.tight_layout()
        fig.savefig(str(output), dpi=350)

include: "_core_rules.smk"


rule plot_spiral_visalisation:
    output: 'figures/visualisation.pdf'
    script:
        "scripts/plot_spiral.py"
