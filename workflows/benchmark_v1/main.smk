# Main workflow used for configuring and running the benchmark.

import bmi.estimators as estimators
import bmi.estimators.external.r_estimators as r_estimators
import bmi.estimators.external.julia_estimators as julia_estimators
from bmi.benchmark import BENCHMARK_TASKS
from bmi.benchmark.tasks import transform_rescale


# Estimators used in the benchmark
ESTIMATORS = {
   "MINE": estimators.MINEEstimator(verbose=False),
   "InfoNCE": estimators.InfoNCEEstimator(verbose=False),
   "NWJ": estimators.NWJEstimator(verbose=False),
   "Donsker-Varadhan": estimators.DonskerVaradhanEstimator(verbose=False),

   # 'KSG-5': estimators.KSGEnsembleFirstEstimator(neighborhoods=(5,)),
   # 'KSG-10': estimators.KSGEnsembleFirstEstimator(neighborhoods=(10,)),
   # 'Hist-10': estimators.HistogramEstimator(n_bins_x=10),

   "R-KSG-I-10": r_estimators.RKSGEstimator(variant=1, neighbors=5),
   # 'R-KSG-I-10': r_estimators.RKSGEstimator(variant=1, neighbors=10),
   # 'R-KSG-II-5': r_estimators.RKSGEstimator(variant=2, neighbors=5),
   # 'R-KSG-II-10': r_estimators.RKSGEstimator(variant=2, neighbors=10),
   # 'R-BNSL': r_estimators.RBNSLEstimator(),
   "R-LNN": r_estimators.RLNNEstimator(),

   "Julia-Hist-10": julia_estimators.JuliaHistogramEstimator(bins=10),
   # 'Julia-Kernel': julia_estimators.JuliaKernelEstimator(),
   "Julia-Transfer-30": julia_estimators.JuliaTransferEstimator(bins=30),
   # 'Julia-KSG-I-5': julia_estimators.JuliaKSGEstimator(variant=1, neighbors=5),

   "CCA": estimators.CCAMutualInformationEstimator(),

   # TODO(frdrc): sth like:
   # "Your Estimator": estimators.ExternalScript('/path/to/your/estimator')
}

# We use scaled versions of all tasks, so that all the estimators
# have equal chances, even if they don't do scaling on their own
# TODO(frdrc): list tasks explicitly so it's easy to comment out
UNSCALED_TASKS = dict(list(BENCHMARK_TASKS.items()))
TASKS = {
    task_name: transform_rescale(
        base_task=base_task,
        task_name=base_task.name,
        task_id=base_task.id,
    )
    for task_name, base_task in UNSCALED_TASKS.items()
}

# Number of samples drawn from the tasks
N_SAMPLES = [100, 500, 1_000, 3_000, 5_000, 10_000]

# Seeds used for task sampling
SEEDS = list(range(10))


# Set location where results will be saved
workdir: "generated/benchmark_v1/"


# Define workflow targets
rule all:
    input: 'results.csv', 'figures/benchmark.pdf'


# Include rules for running the tasks and plotting results
# TODO(frdrc): _append_precomputed.smk which tries to merge precomputed results?
include: "_plot.smk"
include: "_run.smk"
