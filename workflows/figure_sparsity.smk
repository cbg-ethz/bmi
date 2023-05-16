import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker

import bmi
import _sparsity_utils as su
from bmi.plot_utils.subplots_from_axsize import subplots_from_axsize
from _common_figure_utils import (
    read_results, format_axs, plot_mi,
    ESTIMATORS, ESTIMATOR_NAMES, ESTIMATOR_COLORS, scale_tasks
)

DESIRED_MUTUAL_INFORMATION = 1.0
DIM = 10
INITIAL_ALPHA = su.get_initial_alpha(DESIRED_MUTUAL_INFORMATION, DIM)

# ESTIMATORS from _common_figure_utils

TASK_LIST = [
    su.get_lambda_task(DESIRED_MUTUAL_INFORMATION, DIM, alpha)
    for alpha in np.linspace(0.0, INITIAL_ALPHA, 10)
] + [
    su.get_n_interacting_task(DESIRED_MUTUAL_INFORMATION, DIM, n_interacting)
    for n_interacting in np.arange(1, DIM, 1)
]
TASKS = scale_tasks({task.id: task for task in TASK_LIST})

N_SAMPLES = [10_000]
SEEDS = list(range(5))


# === WORKDIR ===
workdir: "generated/figure_sparsity/"


# === RULES ===
rule all:
    input: ['figures/sparsity.pdf', 'figures/interpolating_matrices.pdf']

rule figure_estimates:
    input: 'results.csv'
    output: 'figures/sparsity.pdf'
    run:
        results = read_results(str(input), max_mi_estimate=3.0)
        fig, axs = subplots_from_axsize(
            axsize=(2.0, 1.5), ncols=2,
            left=0.8, right=1.75, wspace=0.85,
        )
        format_axs(axs)

        # decreasing alpha
        ax = axs[0]
        data = results[results['n_interacting'] == DIM]
        plot_mi(ax, data, 'alpha', ESTIMATOR_COLORS, ESTIMATOR_NAMES, x_label=r"$\alpha$", plot_std=True)
        #ax.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
        ax.set_xlim(ax.get_xlim()[::-1])

        # decreasing n_interacting
        ax = axs[1]
        data = results[results['alpha'] == 0.0]
        plot_mi(ax, data, 'n_interacting', ESTIMATOR_COLORS, ESTIMATOR_NAMES, x_label="Num. of interactions", plot_std=True)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.set_xlim(ax.get_xlim()[::-1])

        axs[1].legend(bbox_to_anchor=(1.07, 1.05), frameon=False)
        fig.savefig(str(output))


def params_to_lvm(params):
    return bmi.samplers.GaussianLVMParametrization(
        dim_x=DIM,
        dim_y=DIM,
        n_interacting=params["n_interacting"],
        alpha=params["alpha"],
        lambd=params["lambd"],
        beta_x=params["beta_x"],
        eta_x=params["eta_x"],
    )


rule figure_matrices:
    output: 'figures/interpolating_matrices.pdf'
    run:
        fig, axs = subplots_from_axsize(
            axsize=(2, 2), ncols=3, nrows=2,
            left=0.8, wspace=0.3, top=0.3,
        )

        tasks = [
            su.get_lambda_task(DESIRED_MUTUAL_INFORMATION, DIM, INITIAL_ALPHA),  # Alpha: initial (max)
            su.get_lambda_task(DESIRED_MUTUAL_INFORMATION, DIM, 0.8 * INITIAL_ALPHA),  # Alpha: medium
            su.get_lambda_task(DESIRED_MUTUAL_INFORMATION, DIM, 0.4 * INITIAL_ALPHA),  # Alpha: low
            su.get_n_interacting_task(DESIRED_MUTUAL_INFORMATION, DIM, DIM),  # Alpha: 0, max interacting
            su.get_n_interacting_task(DESIRED_MUTUAL_INFORMATION, DIM, DIM // 2),  # Half interacting
            su.get_n_interacting_task(DESIRED_MUTUAL_INFORMATION, DIM, 1),  # One interacting
        ]

        labels = [
            f"$\\alpha$={INITIAL_ALPHA:.2}",
            f"$\\alpha$={0.8 * INITIAL_ALPHA:.1}",
            f"$\\alpha$={0.4 * INITIAL_ALPHA:.1}",
            f"$K$={DIM}",
            f"$K$={DIM//2}",
            f"$K$={1}",
        ]

        matrices = [params_to_lvm(task.params).correlation for task in tasks]

        for label, ax, matrix in zip(labels, axs.ravel(), matrices):
            sns.heatmap(matrix, ax=ax, vmin=0, vmax=1, cmap="Greys", cbar=False, square=True)

            ax.vlines(DIM, 0, 2 * DIM, linestyles="solid", colors="black")
            ax.hlines(DIM, 0, 2 * DIM, linestyles="solid", colors="black")

            ax.set_xticks([DIM//2, DIM//2 + DIM], ["$X$", "$Y$"], rotation=0)
            ax.set_yticks([DIM//2, DIM//2 + DIM], ["$X$", "$Y$"], rotation=0)
            ax.set_title(label)


        fig.savefig(str(output))


include: "_core_rules.smk"
