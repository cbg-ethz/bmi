import bmi.estimators as estimators
from bmi.benchmark import BENCHMARK_TASKS

from bmi.plot_utils.subplots_from_axsize import subplots_from_axsize
from _common_figure_utils import (
    read_results,
    plot_benchmark_mi_estimate,
    plot_benchmark_neural_fails,
    scale_tasks,
)


# === CONFIG ===

ESTIMATORS = {
    'MINE-D': estimators.MINEEstimator(hidden_layers=(8, 8, 8), verbose=False),
    'MINE-L': estimators.MINEEstimator(hidden_layers=(24, 12), verbose=False),
    'MINE-M': estimators.MINEEstimator(hidden_layers=(16, 8), verbose=False),
    'MINE-S': estimators.MINEEstimator(hidden_layers=(10, 5), verbose=False),
    'InfoNCE-D': estimators.InfoNCEEstimator(hidden_layers=(8, 8, 8), verbose=False),
    'InfoNCE-L': estimators.InfoNCEEstimator(hidden_layers=(24, 12), verbose=False),
    'InfoNCE-M': estimators.InfoNCEEstimator(hidden_layers=(16, 8), verbose=False),
    'InfoNCE-S': estimators.InfoNCEEstimator(hidden_layers=(10, 5), verbose=False),
}

ESTIMATOR_NAMES = {
    estimator_id: estimator_id.replace('-', ' ') for estimator_id in ESTIMATORS
}

TASKS = scale_tasks({
    task_id: BENCHMARK_TASKS[task_id]
    for task_id in {
        '1v1-bimodal-0.75',
        'asinh-student-identity-1-1-1',
        'swissroll_x-normal_cdf-1v1-normal-0.75',
        'multinormal-dense-25-25-0.5',
        'multinormal-sparse-5-5-2-2.0',
        'multinormal-sparse-25-25-2-2.0',
        'normal_cdf-multinormal-sparse-5-5-2-2.0',
        'normal_cdf-multinormal-sparse-25-25-2-2.0',
        'student-identity-3-3-2',
        'student-identity-5-5-2',
        'asinh-student-identity-3-3-2',
        'asinh-student-identity-5-5-2',
    }
})

N_SAMPLES = [10_000]
SEEDS = list(range(5))


# === WORKDIR ===
workdir: "generated/benchmark_neural_arch/"


# === RULES ===
rule all:
    input: 'figures/mi_estimate.pdf', 'figures/neural.pdf'

rule figure_mi_estimate:
    input: 'results.csv'
    output: 'figures/mi_estimate.pdf'
    run:
        results = read_results(str(input))
        fig, ax = subplots_from_axsize(
            axsize=(len(TASKS) * 0.3, len(ESTIMATORS) * 0.3),
            left=1.2, bottom=4.
        )
        plot_benchmark_mi_estimate(ax, results, ESTIMATORS, TASKS, ESTIMATOR_NAMES)
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
        plot_benchmark_neural_fails(ax, results, ESTIMATORS, TASKS, estimator_names=ESTIMATOR_NAMES)
        fig.savefig(str(output))

include: "_core_rules.smk"
