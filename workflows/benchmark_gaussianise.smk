import bmi.estimators as estimators
import bmi.estimators.external.r_estimators as r_estimators
import bmi.estimators.external.julia_estimators as julia_estimators
from bmi.benchmark.tasks import transform_gaussianise
from bmi.benchmark import BENCHMARK_TASKS

from bmi.plot_utils.subplots_from_axsize import subplots_from_axsize
from _common_figure_utils import (
    ESTIMATORS,
    read_results,
    plot_benchmark_mi_estimate,
    plot_benchmark_n_samples,
    plot_benchmark_neural_fails,
)


# === CONFIG ===

# ESTIMATORS from _common_figure_utils

# Apply transformation to all tasks
TASKS = {
    task.id: task
    for task in [
        transform_gaussianise(task)
        for task in BENCHMARK_TASKS.values()
    ]
}

N_SAMPLES = [10_000]

SEEDS = list(range(10))


# === WORKDIR ===
workdir: "generated/benchmark_gaussianise/"


# === RULES ===
rule all:
    input: 'figures/mi_estimate.pdf', 'figures/n_samples.pdf', 'figures/neural.pdf'

rule figure_mi_estimate:
    input: 'results.csv'
    output: 'figures/mi_estimate.pdf'
    run:
        results = read_results(str(input))
        fig, ax = subplots_from_axsize(
            axsize=(len(TASKS) * 0.3, len(ESTIMATORS) * 0.3),
            left=1.2, bottom=4.
        )
        plot_benchmark_mi_estimate(ax, results, ESTIMATORS, TASKS)
        fig.savefig(str(output))

rule figure_n_samples:
    input: 'results.csv'
    output: 'figures/n_samples.pdf'
    run:
        results = read_results(str(input))
        fig, ax = subplots_from_axsize(
            axsize=(len(TASKS) * 0.4, len(ESTIMATORS) * 0.35),
            left=1.2, bottom=4.
        )
        plot_benchmark_n_samples(ax, results, ESTIMATORS, TASKS)
        fig.savefig(str(output))

rule figure_neural:
    input: 'results.csv'
    output: 'figures/neural.pdf'
    run:
        results = read_results(str(input))
        fig, ax = subplots_from_axsize(
            axsize=(len(TASKS) * 0.3, 4 * 0.3),
            left=1.2, bottom=4.
        )
        plot_benchmark_neural_fails(ax, results, ESTIMATORS, TASKS)
        fig.savefig(str(output))

include: "_core_rules.smk"
