import bmi.estimators as estimators
import bmi.estimators.external.r_estimators as r_estimators
import bmi.estimators.external.julia_estimators as julia_estimators
from bmi.benchmark import BENCHMARK_TASKS

from bmi.plot_utils.subplots_from_axsize import subplots_from_axsize
from _common_figure_utils import (
    ESTIMATORS,
    read_results,
    plot_benchmark_mi_estimate,
    plot_benchmark_n_samples,
    plot_benchmark_neural_fails,
    scale_tasks,
)


# === CONFIG ===

# ESTIMATORS = {
#     'MINE': estimators.MINEEstimator(verbose=False),
#     'InfoNCE': estimators.InfoNCEEstimator(verbose=False),
#     'NWJ': estimators.NWJEstimator(verbose=False),
#     'Donsker-Varadhan': estimators.DonskerVaradhanEstimator(verbose=False),
#
#     #'KSG-5': estimators.KSGEnsembleFirstEstimator(neighborhoods=(5,)),
#     #'KSG-10': estimators.KSGEnsembleFirstEstimator(neighborhoods=(10,)),
#     #'Hist-10': estimators.HistogramEstimator(n_bins_x=10),
#
#     #'R-KSG-I-5': r_estimators.RKSGEstimator(variant=1, neighbors=5),
#     'R-KSG-I-10': r_estimators.RKSGEstimator(variant=1, neighbors=10),
#     #'R-KSG-II-5': r_estimators.RKSGEstimator(variant=2, neighbors=5),
#     #'R-KSG-II-10': r_estimators.RKSGEstimator(variant=2, neighbors=10),
#     #'R-BNSL': r_estimators.RBNSLEstimator(),
#     'R-LNN': r_estimators.RLNNEstimator(),
#
#     'Julia-Hist-10': julia_estimators.JuliaHistogramEstimator(bins=10),
#     #'Julia-Kernel': julia_estimators.JuliaKernelEstimator(),
#     'Julia-Transfer-30': julia_estimators.JuliaTransferEstimator(bins=30),
#     #'Julia-KSG-I-5': julia_estimators.JuliaKSGEstimator(variant=1, neighbors=5),
# }

# We use scaled versions of all tasks, so that all the estimators
# have equal chances, even if they don't do scaling on their own
TASKS = scale_tasks(BENCHMARK_TASKS)

N_SAMPLES = [100, 500, 1_000, 3_000, 5_000, 10_000]

SEEDS = list(range(10))


# === WORKDIR ===
workdir: "generated/benchmark/"


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
