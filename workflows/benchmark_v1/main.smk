# Main workflow for running the benchmark.
# To configure the benchmark see config.py

from config import ESTIMATORS, TASKS, N_SAMPLES, SEEDS

# Set location where results will be saved
workdir: "generated/benchmark_v1/"

# Define workflow targets
rule all:
    input: 'results.csv', 'figures/benchmark.pdf'

# Include rules for running the tasks and plotting results
# TODO(frdrc): _append_precomputed.smk which tries to merge precomputed results?
include: "_plot.smk"
include: "_run.smk"
