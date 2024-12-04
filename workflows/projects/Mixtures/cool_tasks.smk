"""Demonstration of the capabilities of the BMM family."""
import numpy as np
import pandas as pd
import matplotlib
from subplots_from_axsize import subplots_from_axsize
matplotlib.use("agg")

import jax.numpy as jnp

import bmi
from bmi.samplers import bmm

import example_distributions as ed


N_SAMPLES = [1_000, 5_000 ]
SEEDS = list(range(10))

ESTIMATORS = {
    "MINE": bmi.estimators.MINEEstimator(verbose=False),
    "InfoNCE": bmi.estimators.InfoNCEEstimator(verbose=False),
    # "NWJ": bmi.estimators.NWJEstimator(verbose=False),
    # "Donsker-Varadhan": bmi.estimators.DonskerVaradhanEstimator(verbose=False),
    'KSG-10': bmi.estimators.KSGEnsembleFirstEstimator(neighborhoods=(10,)),
    'Hist-10': bmi.estimators.HistogramEstimator(n_bins_x=10),
    "CCA": bmi.estimators.CCAMutualInformationEstimator(),
}
ESTIMATOR_NAMES = {
    "MINE": "MINE",
    "InfoNCE": "InfoNCE",
    "KSG-10": "KSG",
    "Hist-10": "Histogram",
    "CCA": "CCA",
}
ESTIMATOR_COLORS = {
    "MINE": '#377eb8',
    "InfoNCE": '#ff7f00',
    "KSG-10": '#4daf4a',
    "Hist-10": '#f781bf',
    "CCA": '#a65628',
}

ESTIMATOR_MARKERS = {
    "MINE": '.',
    "InfoNCE": 'v',
    "KSG-10": '^',
    "Hist-10": 'D',
    "CCA": 'X',
}

assert set(ESTIMATOR_NAMES.keys()) == set(ESTIMATORS.keys())
assert set(ESTIMATOR_COLORS.keys()) == set(ESTIMATORS.keys())
assert set(ESTIMATOR_MARKERS.keys()) == set(ESTIMATORS.keys())

_SAMPLE_ESTIMATE: int = 200_000

x_sampler = ed.create_x_distribution(_sample=_SAMPLE_ESTIMATE).sampler 
ai_sampler = ed.create_ai_distribution(_sample=_SAMPLE_ESTIMATE).sampler
waves_sampler = ed.create_waves_distribution(_sample=_SAMPLE_ESTIMATE).sampler 
galaxy_sampler = ed.create_galaxy_distribution(_sample=_SAMPLE_ESTIMATE).sampler

UNSCALED_TASKS = {
    "X": bmi.benchmark.Task(
        sampler=x_sampler,
        task_id="X",
        task_name="X",
    ),
    "AI": bmi.benchmark.Task(
        sampler=ai_sampler,
        task_id="AI",
        task_name="AI",
    ),
    "Fence": bmi.benchmark.Task(
        sampler=waves_sampler,
        task_id="Fence",
        task_name="Fence",
    ),
    "Balls": bmi.benchmark.Task(
        sampler=galaxy_sampler,
        task_id="Balls",
        task_name="Balls",
    ),
}


# === WORKDIR ===
workdir: "generated/mixtures/cool_tasks/"

rule all:
    input:
        'cool_tasks.pdf',
        'results.csv',
        'cool_tasks-results.pdf',
        # 'profiles.pdf'

rule plot_distributions:
    output: "cool_tasks.pdf"
    run:
        fig, axs = subplots_from_axsize(1, 4, axsize=(1.5, 1.5), wspace=0.4)

        # Plot the X distribution
        ax = axs[0]
        xs, ys = x_sampler.sample(1000, 0)

        size = 2**2

        ax.scatter(xs[:, 0], ys[:, 0], s=size, alpha=0.3, color="k", rasterized=True)
        ax.set_xlabel("$X$")
        ax.set_ylabel("$Y$")

        # Plot the AI distribution
        ax = axs[1]
        xs, ys = ai_sampler.sample(2000, 0)
        ax.scatter(xs[:, 0], ys[:, 0], s=size, alpha=0.3, color="k", rasterized=True)
        ax.set_xlabel("$X$")
        ax.set_ylabel("$Y$")

        # Plot the fence distribution
        ax = axs[2]
        xs, ys = waves_sampler.sample(2000, 0)

        ax.scatter(xs[:, 0], xs[:, 1], c=ys[:, 0], s=size, alpha=0.3, rasterized=True)
        ax.set_xlabel("$X_1$")
        ax.set_ylabel("$X_2$")

        # Plot transformed balls distribution
        ax = axs[3]
        xs, ys = galaxy_sampler.sample(2000, 0)
        ax.scatter(xs[:, 0], xs[:, 1], c=ys[:, 0], s=size, alpha=0.3, rasterized=True)
        ax.set_xlabel("$X_1$")
        ax.set_ylabel("$X_2$")

        for ax in axs:
            ticks = [-1, 0, 1]
            ax.set_xticks(ticks, ticks)
            ax.set_yticks(ticks, ticks)
            ax.set_xlim(-2., 2.)
            ax.set_ylim(-2., 2.)
            ax.spines[['right', 'top']].set_visible(False)

        fig.savefig(str(output), dpi=300)

# rule plot_pmi_profiles:
#     output: "profiles.pdf"
#     run:
#         fig, axs = subplots_from_axsize(1, 4, axsize=(4, 3))
#         dists = [x_dist, ai_dist, fence_base_dist, balls_mixt]
#         tasks_official = ['X', 'AI', 'Waves', 'Galaxy']
#         for dist, task_name, ax in zip(dists, tasks_official, axs):
#             import jax
#             key = jax.random.PRNGKey(1024)
#             pmi_values = bmm.pmi_profile(key=key, dist=dist, n=100_000)
#             bins = np.linspace(-5, 5, 101)
#             ax.hist(pmi_values, bins=bins, density=True, alpha=0.5)
#             ax.set_xlabel(task_name)
#         axs[0].set_ylabel("Density")
#         fig.savefig(str(output))


rule plot_results:
    output: 'cool_tasks-results.pdf'
    input: 'results.csv'
    run:
        data = pd.read_csv(str(input))
        fig, ax = subplots_from_axsize(1, 1, (2, 1.5), right=1.3)

        data_5k = data[data['n_samples'] == 5000]
        tasks = ['X', 'AI', 'Fence', 'Balls']
        tasks_official = ['X', 'AI', 'Waves', 'Galaxy']

        for estimator_id, data_est in data_5k.groupby('estimator_id'):
            ax.scatter(
                data_est['task_id'].apply(lambda e: tasks.index(e)) + 0.05 * np.random.normal(size=len(data_est)),
                data_est['mi_estimate'],
                label=ESTIMATOR_NAMES[estimator_id],
                alpha=0.4, s=3**2,
                marker=ESTIMATOR_MARKERS[estimator_id],
                c=ESTIMATOR_COLORS[estimator_id],
                edgecolor="none",
            )
            
        for task_id, data_task in data_5k.groupby('task_id'):
            true_mi = data_task['mi_true'].mean()
            x = tasks.index(task_id)
            ax.plot([x - 0.2, x + 0.2], [true_mi, true_mi], ':k')

        ax.set_xticks(range(len(tasks)), tasks_official)
            
        ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1, 1))
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylim(-0.1, 1.4)
        ax.set_ylabel('MI')
        fig.savefig(str(output))


include: "_benchmark_rules.smk"
