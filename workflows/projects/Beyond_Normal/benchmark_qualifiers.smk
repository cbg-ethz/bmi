import bmi.estimators as estimators
import bmi.estimators.external.r_estimators as r_estimators
import bmi.estimators.external.julia_estimators as julia_estimators
from bmi.benchmark import BENCHMARK_TASKS

from bmi.plot_utils.subplots_from_axsize import subplots_from_axsize
from _common_figure_utils import (
    read_results,
    plot_benchmark_mi_estimate,
    scale_tasks,
)


# === CONFIG ===

ESTIMATORS = {
    'MINE': estimators.MINEEstimator(verbose=False),
    'InfoNCE': estimators.InfoNCEEstimator(verbose=False),
    'NWJ': estimators.NWJEstimator(verbose=False),
    'Donsker-Varadhan': estimators.DonskerVaradhanEstimator(verbose=False),

    #'KSG-5': estimators.KSGEnsembleFirstEstimator(neighborhoods=(5,)),
    #'KSG-10': estimators.KSGEnsembleFirstEstimator(neighborhoods=(10,)),
    #'Hist-10': estimators.HistogramEstimator(n_bins_x=10),

    #'R-KSG-I-5': r_estimators.RKSGEstimator(variant=1, neighbors=5),
    'R-KSG-I-10': r_estimators.RKSGEstimator(variant=1, neighbors=10),
    #'R-KSG-II-5': r_estimators.RKSGEstimator(variant=2, neighbors=5),
    #'R-KSG-II-10': r_estimators.RKSGEstimator(variant=2, neighbors=10),
    'R-BNSL': r_estimators.RBNSLEstimator(),
    'R-LNN': r_estimators.RLNNEstimator(),

    'Julia-Hist-10': julia_estimators.JuliaHistogramEstimator(bins=10),
    'Julia-Kernel': julia_estimators.JuliaKernelEstimator(),
    'Julia-Transfer-30': julia_estimators.JuliaTransferEstimator(bins=30),
    #'Julia-KSG-I-5': julia_estimators.JuliaKSGEstimator(variant=1, neighbors=5),
}

TASKS = scale_tasks({
    task_id: BENCHMARK_TASKS[task_id]
    for task_id in {
        '1v1-bimodal-0.75',
        'asinh-student-identity-1-1-1',
        'swissroll_x-normal_cdf-1v1-normal-0.75',
        'multinormal-sparse-3-3-2-2.0',
        'multinormal-sparse-5-5-2-2.0',
    }
})

N_SAMPLES = [10_000]
SEEDS = list(range(5))


# === WORKDIR ===
workdir: "generated/benchmark_qualifiers/"


# === RULES ===
rule all:
    input: 'figures/mi_estimate.pdf'

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

include: "_core_rules.smk"
