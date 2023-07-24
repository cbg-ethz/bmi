# Workflow used to generate figures where we increase
# mutual information in a given family of distributions
import pandas as pd
import seaborn as sns
import yaml

import _high_mi_utils as hmu
import bmi.estimators
from bmi.plot_utils.subplots_from_axsize import subplots_from_axsize
from _common_figure_utils import ESTIMATORS, ESTIMATOR_NAMES, ESTIMATOR_COLORS, scale_tasks, format_axs, plot_mi

# On the X axis in each plot we will have the following mutual information
# values
DESIRED_MUTUAL_INFORMATION = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

# ESTIMATORS from _common_figure_utils

# Description of the distribution families plotted at each plot axis
# This is the dictionary of the format
# family_name: task_generator
# where `family_name` is the name of the distribution family
# and `task_generator` is a function of signature `float, str -> Optional[Task]`
# generating a task with (approximately) desired mutual information
# and with specified `task_params["family_name"]`
# (or None if the task cannot be created)
DISTRIBUTION_FAMILIES = {
    "Sparse Gaussian": lambda mi, family_name: hmu.generate_sparse_gaussian_task(mi=mi, family_name=family_name, dim=3),
    "Spiral": lambda mi, family_name: hmu.generate_spiral_task(mi=mi, family_name=family_name, dim=3, speed=1/3),
    "Half-cube": lambda mi, family_name: hmu.generate_half_cube_task(mi=mi, family_name=family_name, dim=3),
}

RAW_TASK_LIST = [
    task_generator(mi, family_name)
    for family_name, task_generator in DISTRIBUTION_FAMILIES.items()
    for mi in DESIRED_MUTUAL_INFORMATION
]
TASK_LIST = [item for item in RAW_TASK_LIST if item is not None]


TASKS = scale_tasks({task.id: task for task in TASK_LIST})

N_SAMPLES = [10_000]
SEEDS = list(range(5))


# === WORKDIR ===
workdir: "generated/figure_high_mi/"


# === RULES ===
rule all:
    input: 'figure_high_mi.pdf'

rule plot:
    input: 'results.csv'
    output: 'figure_high_mi.pdf'
    run:
        data = pd.read_csv(str(input))

        data = pd.read_csv(str(input))
        data['desired_mi'] = data['task_params'].apply(lambda x: yaml.safe_load(x)['desired_mi'])
        data['family'] = data['task_params'].apply(lambda x: yaml.safe_load(x)['family_name'])

        fig, axs = subplots_from_axsize(
            axsize=(2, 1.5), ncols=3,
            left=0.8, right=1.75, wspace=0.85, top=0.3, bottom=0.6,
        )
        format_axs(axs)

        for ax, (family, mini_df) in zip(axs, data.groupby("family")):
            ax.set_title(family)
            plot_mi(ax, mini_df, "desired_mi", ESTIMATOR_COLORS, ESTIMATOR_NAMES, x_label="True MI [nats]", plot_std=True)

        axs[-1].legend(bbox_to_anchor=(1.07, 1.05), frameon=False)
        fig.savefig(str(output))

include: "_core_rules.smk"
