# Workflow used to visualise tasks

import matplotlib
import pandas as pd
import seaborn as sns

import bmi

# === WORKDIR ===
workdir: "generated/figure_visualise_tasks/"

matplotlib.use("agg")

N_SAMPLES = 5_000

TASK_IDS = [
    task.id for task in bmi.benchmark.task_list.BENCHMARK_TASKS_LIST[:3]
]
BENCHMARK_TASKS = {
    task_id: bmi.benchmark.BENCHMARK_TASKS[task_id]
    for task_id in TASK_IDS
}

def plot_sampler_pairwise(
    task: bmi.Task,
    n_samples: int = 2_000,
    seed: int = 0,
    color: str = "black",
) -> sns.PairGrid:
    xs, ys = task.sample(n_samples, seed)
    dim_x = task.dim_x
    dim_y = task.dim_y

    dfx = pd.DataFrame(xs, columns=[f"$X_{i}$" for i in range(1, dim_x + 1)])
    dfy = pd.DataFrame(ys, columns=[f"$Y_{i}$" for i in range(1, dim_y + 1)])
    df = dfx.join(dfy)
    grid = sns.pairplot(
        df,
        plot_kws={
            "color": color,
            "alpha": 0.1,
            "size": 1,
        },
        diag_kws={"color": color}
    )

    grid.map_lower(sns.kdeplot,levels=4,color=".2")
    grid.figure.suptitle(task.name)
    grid.figure.tight_layout()

    return grid

rule all:
    input: expand("{task_id}.pdf", task_id=TASK_IDS)

rule visualise_task:
    output: "{task_id}.pdf"
    run:
        task_id = str(wildcards.task_id)
        task = BENCHMARK_TASKS[task_id]

        grid = plot_sampler_pairwise(task)
        grid.savefig(str(output))
