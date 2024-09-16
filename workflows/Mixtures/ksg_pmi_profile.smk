import numpy as np
import pandas as pd
import matplotlib
from subplots_from_axsize import subplots_from_axsize
matplotlib.use("agg")

import bmi
from bmi.samplers import fine

import jax
import jax.numpy as jnp


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
    "KSG-10": "#d62728",
    "CCA": "#1f77b4",
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

HEIGHT: float = 1.3

# === WORKDIR ===
workdir: "generated/mixtures/aistats-rebuttal/"

rule all:
    input:
        'results.csv',
        'rebuttal_figure.pdf',
        'pmi_hist.pdf'



rule plot_results:
    output: 'rebuttal_figure.pdf'
    input: 'results.csv'
    run:
        data = pd.read_csv(str(input))
        fig, ax = subplots_from_axsize(1, 1, (2, HEIGHT), right=1.3)

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
                rasterized=True,
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


rule plot_pmi_hist:
    output: 'pmi_hist.pdf'
    run:
        n_dim: int = 1
        n_points: int = 5_000
        k: int = 20

        dist = fine.MultivariateNormalDistribution(
            dim_x=n_dim,
            dim_y=n_dim,
            mean=jnp.zeros(2 * n_dim),
            covariance=bmi.samplers.canonical_correlation([0.8] * n_dim),
        )

        xs, ys = dist.sample(n_points, jax.random.PRNGKey(42))
        pmis = dist.pmi(xs, ys)

        min_pmi = jnp.min(pmis) - 0.1
        max_pmi = jnp.max(pmis) + 0.1

        fig, axs = subplots_from_axsize(1, 3, (2, 1.3), wspace=[0.7, 0.3])

        bins = jnp.linspace(-1.5, 3.5, 31)

        ax = axs[0]
        ax.hist(pmis, bins=bins, density=True, alpha=0.5, color="black")
        ax.set_xlabel("True PMI")
        ax.set_ylabel("Frequency")

        estimator = bmi.estimators.KSGEnsembleFirstEstimatorSlow(neighborhoods=(k,), standardize=False)
        pseudo_pmis = estimator._calculate_digammas(xs, ys, ks=(k,))[k]
        ax = axs[1]
        ax.hist(pseudo_pmis, bins=bins, density=True, alpha=0.5, color="red")
        ax.set_xlabel("KSG PMI")
        ax.set_ylabel("Frequency")

        ax = axs[2]
        ts = jnp.linspace(min_pmi, max_pmi, 3)
        ax.plot(ts, jnp.zeros_like(ts), color="darkblue", linestyle="--")

        ax.scatter(pmis, pseudo_pmis - pmis, s=2, alpha=0.1, c="k", rasterized=True)
        ax.set_xlabel("True PMI")
        ax.set_ylabel("KSG PMI $-$ True PMI")

        ax.set_xlim(min_pmi, max_pmi)
        ax.set_ylim(min_pmi, max_pmi)
        ax.set_aspect("equal")

        corr = np.corrcoef(pmis, pseudo_pmis)[0, 1]
        # ax.annotate(f"$r={corr:.2f}$", xy=(0.05, 0.95), xycoords="axes fraction", ha="left", va="top")

        for ax in axs:
            ax.spines[['top', 'right']].set_visible(False)

        fig.savefig(str(output))

include: "_benchmark_rules.smk"
