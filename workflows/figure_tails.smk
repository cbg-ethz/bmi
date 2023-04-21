import pandas as pd
from common_estimators import ESTIMATORS


# === CONFIG ===
#TASKS = BENCHMARK_TASKS
TASKS = {
    task_id: BENCHMARK_TASKS[task_id]
    for task_id in {
        '1v1-bimodal-0.75',
    }
}

N_SAMPLES = [10000]
SEEDS = [0]


# === WORKDIR ===
workdir: "generated/figure_tails/"


# === RULES ===
rule all:
    input: 'results.csv'

include: "_core_rules.smk"
