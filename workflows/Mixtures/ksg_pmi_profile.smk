import numpy as np
import pandas as pd
import matplotlib
from subplots_from_axsize import subplots_from_axsize
matplotlib.use("agg")

import bmi


N_SAMPLES = [100]
SEEDS = list(range(20))

ESTIMATORS = {
    'KSG-10': bmi.estimators.KSGEnsembleFirstEstimator(neighborhoods=(10,)),
    "CCA": bmi.estimators.CCAMutualInformationEstimator(),
}
ESTIMATOR_NAMES = {
    "KSG-10": "KSG",
    "CCA": "CCA",
}
ESTIMATOR_COLORS = {
    "KSG-10": "red",
    "CCA": "blue",
}
assert set(ESTIMATOR_NAMES.keys()) == set(ESTIMATORS.keys())


def get_sampler(n: int) -> bmi.samplers.SplitMultinormal:
    return bmi.samplers.SplitMultinormal(
        dim_x=n,
        dim_y=n,
        covariance=bmi.samplers.canonical_correlation(rho=[0.5] * n)
    )

UNSCALED_TASKS = {
    "normal-1": bmi.benchmark.Task(
        sampler=get_sampler(1),
        task_id="normal-1",
        task_name="Normal 1 x 1",
    ),
    "normal-3": bmi.benchmark.Task(
        sampler=get_sampler(3),
        task_id="normal-3",
        task_name="Normal 3 x 3",
    ),
    "normal-5": bmi.benchmark.Task(
        sampler=get_sampler(5),
        task_id="normal-5",
        task_name="Normal 5 x 5",
    ),
}


# === WORKDIR ===
workdir: "generated/mixtures/aistats-rebuttal/"

rule all:
    input:
        'results.csv',
        'rebuttal_figure.pdf',



rule plot_results:
    output: 'rebuttal_figure.pdf'
    input: 'results.csv'
    run:
        data = pd.read_csv(str(input))
        fig, ax = subplots_from_axsize(1, 1, (2, 1.5), right=1.3)

        data_5k = data[data['n_samples'] == 100]
        tasks = ["normal-1", "normal-3", "normal-5"]
        tasks_official = ["$n=1$", "$n=3$", "$n=5$"]

        for estimator_id, data_est in data_5k.groupby('estimator_id'):
            ax.scatter(
                data_est['task_id'].apply(lambda e: tasks.index(e)) + 0.05 * np.random.normal(size=len(data_est)),
                data_est['mi_estimate'],
                label=ESTIMATOR_NAMES[estimator_id],
                color=ESTIMATOR_COLORS[estimator_id],
                alpha=0.2, s=3**2,
            )
        
        _flag = True        
        for task_id, data_task in data_5k.groupby('task_id'):
            true_mi = data_task['mi_true'].mean()
            x = tasks.index(task_id)
            ax.plot([x - 0.2, x + 0.2], [true_mi, true_mi], ':k', label="True MI" if _flag else None)
            _flag = False

        ax.set_xticks(range(len(tasks)), tasks_official)
            
        ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1, 1))
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylim(-0.1, 1.3)
        ax.set_ylabel('MI')
        fig.savefig(str(output))


include: "_benchmark_rules.smk"
