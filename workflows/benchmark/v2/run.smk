# ==============================================
# == Main workflow for running the benchmark. ==
# ==============================================
from config import ESTIMATORS_DICT, TASKS, N_SAMPLES, SEEDS

workdir: "generated/benchmark/v2/"


rule all:
    input: 'results.csv', 'benchmark.html'


include: "../_common_benchmark_rules.smk"
