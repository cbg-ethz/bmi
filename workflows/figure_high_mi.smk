# Workflow used to generate figures where we increase
# mutual information in a given family of distributions
import pandas as pd
import seaborn as sns
import yaml

import _high_mi_utils as hmu
import bmi.estimators
from _common_figure_utils import ESTIMATORS, ESTIMATOR_NAMES, ESTIMATOR_COLORS

# On the X axis in each plot we will have the following mutual information
# values
DESIRED_MUTUAL_INFORMATION = [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

ESTIMATORS = ESTIMATORS

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


TASKS = {task.id: task for task in TASK_LIST}

# TODO(Pawel, Frederic): Think about the number of samples to use.
N_SAMPLES = [10_000]
SEEDS = list(range(5))


# === WORKDIR ===
workdir: "generated/figure_high_mi/"


# === RULES ===
rule all:
    input: 'figure_high_mi.pdf'

rule plot:
    input: 'summary.csv'
    output: 'figure_high_mi.pdf'
    run:
        data = pd.read_csv(str(input))

        xticks = data["Mutual Information"].unique()

        sns.set_palette("bright")

        g = sns.FacetGrid(
            data,
            col="Distribution",
            hue="Estimator",
            sharex=True,
            sharey=True,
            height=3,
            legend_out=True,
            palette=ESTIMATOR_COLORS,
        )
        g.map(sns.lineplot, "Mutual Information", "Mean estimate", alpha=0.5)
        g.map(sns.scatterplot,"Mutual Information", "Mean estimate", alpha=0.5)
        g.add_legend()
        g.set(xticks=xticks,xlim=(xticks.min() - 0.2, xticks.max() + 0.2))
        g.set_xticklabels([round(x) if abs(round(x) - x) < 1e-2 else x for x in xticks])

        # Add the diagonal (ground truth MI)
        for ax in g.axes_dict.values():
            ax.axline((0, 0),(1, 1),linestyle=":",c="k")

        g.tight_layout()
        g.savefig(str(output))

rule summarize_results:
    input: 'results.csv'
    output: 'summary.csv'
    run:
        data = pd.read_csv(str(input))
        data['desired_mi'] = data['task_params'].apply(lambda x: yaml.safe_load(x)['desired_mi'])
        data['family'] = data['task_params'].apply(lambda x: yaml.safe_load(x)['family_name'])

        # Calculate data summary
        summary = data.groupby(['family', 'desired_mi', 'estimator_id']).mean(numeric_only=True).reset_index()
        summary = summary.rename(columns={
            "estimator_id": "Estimator",
            "desired_mi": "Mutual Information",
            "family": "Distribution",
            "mi_estimate": "Mean estimate",
        })
        summary["Estimator"].replace(ESTIMATOR_NAMES, inplace=True)
        summary.to_csv(str(output), index=False)

include: "_core_rules.smk"
