import bmi.estimators as estimators
import bmi.estimators.external.r_estimators as r_estimators
import bmi.estimators.external.julia_estimators as julia_estimators
from bmi.benchmark import BENCHMARK_TASKS


# === CONFIG ===

ESTIMATORS = {
    'MINE': estimators.MINEEstimator(verbose=False),
    'InfoNCE': estimators.InfoNCEEstimator(verbose=False),
    'NWJ': estimators.NWJEstimator(verbose=False),
    'Donsker-Varadhan': estimators.DonskerVaradhanEstimator(verbose=False),

    'KSG-5': estimators.KSGEnsembleFirstEstimator(neighborhoods=(5,)),
    'KSG-10': estimators.KSGEnsembleFirstEstimator(neighborhoods=(10,)),
    'Hist-10': estimators.HistogramEstimator(n_bins_x=10),

    'R-KSG-I-5': r_estimators.RKSGEstimator(variant=1, neighbors=5),
    'R-KSG-I-10': r_estimators.RKSGEstimator(variant=1, neighbors=10),
    'R-KSG-II-5': r_estimators.RKSGEstimator(variant=2, neighbors=5),
    'R-KSG-II-10': r_estimators.RKSGEstimator(variant=2, neighbors=10),
    'R-BNSL': r_estimators.RBNSLEstimator(),
    'R-LNN': r_estimators.RLNNEstimator(),

    'Julia-Hist-10': julia_estimators.JuliaHistogramEstimator(bins=10),
    'Julia-Kernel': julia_estimators.JuliaKernelEstimator(),
    'Julia-Transfer-30': julia_estimators.JuliaTransferEstimator(bins=30),
    #'Julia-KSG-I-5': julia_estimators.JuliaKSGEstimator(variant=1, neighbors=5),
}

TASKS = BENCHMARK_TASKS
#TASKS = {
#    task_id: BENCHMARK_TASKS[task_id]
#    for task_id in {
#        '1v1-bimodal-0.75',
#        'student-dense-1-1-5-0.75',
#        'swissroll_x-1v1-normal-0.75',
#        'multinormal-sparse-3-3-2-0.8-0.1',
#        'multinormal-sparse-5-5-2-0.8-0.1',
#    }
#}

N_SAMPLES = [1000, 3000, 10000]

SEEDS = [0, 1, 2]


# === WORKDIR ===
workdir: "generated/benchmark/"


# === RULES ===
rule all:
    input: 'results.csv'

include: "_core_rules.smk"
