# ==============================================
# == Main workflow for running the benchmark. ==
# ==============================================
from config import ESTIMATORS_DICT, TASKS, N_SAMPLES, SEEDS

workdir: "generated/benchmark/demo/"


rule all:
    input: 'results.csv', 'benchmark.html'


include: "../_common_benchmark_rules.smk"
