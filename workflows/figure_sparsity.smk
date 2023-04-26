import numpy as np
import matplotlib.ticker as ticker

import bmi
import _sparsity_utils as su
from bmi.plot_utils.subplots_from_axsize import subplots_from_axsize
from _common_figure_utils import (
    read_results, format_axs, plot_mi,
    ESTIMATORS, ESTIMATOR_NAMES, ESTIMATOR_COLORS,
)

DESIRED_MUTUAL_INFORMATION = 1.0
DIM = 5
INITIAL_ALPHA = su.get_initial_alpha(DESIRED_MUTUAL_INFORMATION, DIM)

# Estimators which performance is plotted
ESTIMATORS = {
    "KSG": bmi.estimators.KSGEnsembleFirstEstimator(),
    "InfoNCE": bmi.estimators.InfoNCEEstimator(verbose=False),
}
ESTIMATOR_NAMES = {key: key for key in ESTIMATORS}
ESTIMATOR_COLORS = {key: f"C{i}" for i, key in enumerate(ESTIMATORS)}

TASK_LIST = [
    su.get_lambda_task(DESIRED_MUTUAL_INFORMATION, DIM, alpha)
    for alpha in np.linspace(0.0, INITIAL_ALPHA, 5)
] + [
    su.get_n_interacting_task(DESIRED_MUTUAL_INFORMATION, DIM, n_interacting)
    for n_interacting in np.arange(1, DIM, 2)
]
TASKS = {task.id: task for task in TASK_LIST}

# TODO(Pawel, Frederic): Think about the number of samples to use.
N_SAMPLES = [10_000]
SEEDS = [0]  # [0, 1, 2, 3, 4]


# === WORKDIR ===
workdir: "generated/figure_sparsity/"


# === RULES ===
rule all:
    input: 'figures/sparsity.pdf'

rule figure_tails:
    input: 'results.csv'
    output: 'figures/sparsity.pdf'
    run:
        results = read_results(str(input))
        fig, axs = subplots_from_axsize(
            axsize=(2.0, 1.5), ncols=2,
            left=0.8, right=1.75, wspace=0.85,
        )
        format_axs(axs)

        # decreasing alpha
        ax = axs[0]
        data = results[results['n_interacting'] == DIM]
        plot_mi(ax, data, 'alpha', ESTIMATOR_COLORS, ESTIMATOR_NAMES)
        #ax.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
        ax.set_xlim(ax.get_xlim()[::-1])

        # decreasing n_interacting
        ax = axs[1]
        data = results[results['alpha'] == 0.0]
        plot_mi(ax, data, 'n_interacting', ESTIMATOR_COLORS, ESTIMATOR_NAMES)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.set_xlim(ax.get_xlim()[::-1])

        axs[1].legend(bbox_to_anchor=(1.07, 1.05))
        fig.savefig(str(output))


include: "_core_rules.smk"
