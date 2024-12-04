"""This workflow tests how the performance of different estimators changes
when mixing with "outlier" distribution is performed.
"""
from typing import Callable, Optional
import dataclasses
import json
import yaml

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import jax
import jax.numpy as jnp

import bmi
from bmi.samplers import bmm


# === WORKDIR ===
workdir: "generated/mixtures/outliers"


@dataclasses.dataclass
class ChangeMixingSetup:
    dist_true: bmm.JointDistribution
    dist_noise: bmm.JointDistribution

    def mixture(self, alpha: float) -> bmm.JointDistribution:
        """Return a mixture of the two distributions.
        
        Args:
            alpha: contamination level, i.e. the probability of the noise distribution
        """
        return bmm.mixture(
            components=[self.dist_true, self.dist_noise],
            proportions=jnp.asarray([1.0 - alpha, alpha]),
        )

signal_cov_parametrization = bmi.samplers.SparseLVMParametrization(dim_x=2, dim_y=2, n_interacting=2, beta=0.1, lambd=2.0)
signal_cov_matrix = signal_cov_parametrization.correlation

dist_signal_gauss = bmm.MultivariateNormalDistribution(
    dim_x=2,
    dim_y=2,
    covariance=signal_cov_matrix,
)

covariance_inlier = jnp.eye(dist_signal_gauss.dim_y)
dist_gaussian_inlier = bmm.ProductDistribution(
    dist_x=dist_signal_gauss.dist_x,
    dist_y=bmm.construct_multivariate_normal_distribution(mean=jnp.zeros(dist_signal_gauss.dim_y), covariance=covariance_inlier),
)

# Outliers: 5 sigma
covariance_outlier = 25 * jnp.eye(dist_signal_gauss.dim_y)
dist_gaussian_outlier = bmm.ProductDistribution(
    dist_x=dist_signal_gauss.dist_x,
    dist_y=bmm.construct_multivariate_normal_distribution(mean=jnp.zeros(dist_signal_gauss.dim_y), covariance=covariance_outlier),
)

CHANGE_MIXING_SETUPS = {
    "Gauss-Inlier": ChangeMixingSetup(dist_true=dist_signal_gauss, dist_noise=dist_gaussian_inlier),
    "Gauss-Outlier": ChangeMixingSetup(dist_true=dist_signal_gauss, dist_noise=dist_gaussian_outlier),
}


VARIANCES = 2**np.concatenate([np.arange(-7, -2, 1.0), np.arange(-2, 4, 0.5), np.arange(4, 8, 1.0)])
ALPHAS = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

MIXING_TASKS = {}
# Add mixing tasks
for setup_name, setup in CHANGE_MIXING_SETUPS.items():
    for alpha in ALPHAS:
        sampler = bmm.BMMSampler(dist=setup.mixture(alpha=alpha), mi_estimate_sample=100_000)
        task = bmi.Task(
            sampler=sampler,
            task_id=f"mixing-{setup_name}-{alpha}",
            task_name=f"mixing-{setup_name}-{alpha}",
            task_params={"mixing": alpha},
        )
        MIXING_TASKS[task.id] = task

# Add the task with changing the variance
VARIANCE_TASKS = {}
VARIANCE_MIXING: float = 0.2
for variance in VARIANCES:
    dist_noise_variance =  bmm.ProductDistribution(
        dist_x=dist_signal_gauss.dist_x,
        dist_y=bmm.construct_multivariate_normal_distribution(
            mean=jnp.zeros(dist_signal_gauss.dim_y),
            covariance=variance * jnp.eye(dist_signal_gauss.dim_y)
        ),
    )
    sampler = bmm.BMMSampler(
        dist=bmm.mixture(
            proportions=jnp.array([1.0-VARIANCE_MIXING, VARIANCE_MIXING]),
            components=[dist_signal_gauss, dist_noise_variance],
        ),
        mi_estimate_sample=100_000,
    )
    task = bmi.Task(
        sampler=sampler,
        task_id=f"variance-{variance}",
        task_name=f"variance-{variance}",
        task_params={"variance": variance},
    )
    VARIANCE_TASKS[task.id] = task

UNSCALED_TASKS = {**MIXING_TASKS, **VARIANCE_TASKS}

ESTIMATOR_COLORS = {
    "InfoNCE": '#ff7f00',
    "MINE": '#377eb8',
    "KSG": '#4daf4a',
    "CCA": '#a65628',
}

ESTIMATOR_MARKERS = {
    "InfoNCE": 'v',
    "MINE": '.',
    "KSG": '^',
    "CCA": 'X',
}


ESTIMATORS = {
    "KSG": bmi.estimators.KSGEnsembleFirstEstimator(neighborhoods=(10,)),
    "CCA": bmi.estimators.CCAMutualInformationEstimator(),
    "InfoNCE": bmi.estimators.InfoNCEEstimator(),
    "MINE": bmi.estimators.MINEEstimator(),
}
SEEDS = list(range(10))
N_SAMPLES = [5000]


rule all:
    input:
        mixing_ground_truths = expand("{setup}/mixing_ground_truths.done", setup=CHANGE_MIXING_SETUPS.keys()),
        estimates = "results.csv",
        outliers_plot = "outliers_plot.pdf",
        parameters = "parameters.json",
        covariance_heatmap = "covariance_heatmap.pdf"

rule plot_parameters:
    output:
        params_json = "parameters.json",
        covariance_heatmap = "covariance_heatmap.pdf"
    run:
        with open(str(output.params_json), "w") as fh:
            json.dump({
                "signal_covariance": signal_cov_matrix.tolist(),
                "signal_mutual_information": dist_signal_gauss.analytic_mi,
            }, fp=fh, indent=4)
        fig, ax = plt.subplots()
        sns.heatmap(signal_cov_matrix, ax=ax, annot=True, fmt=".2f", cmap="coolwarm")
        fig.tight_layout()
        fig.savefig(str(output.covariance_heatmap))


def plot_data(ax: plt.Axes, data: pd.DataFrame, key: str = "mixing", use_legend: bool = False):
    data[key] = data["task_params"].map(lambda x: yaml.load(x, Loader=yaml.SafeLoader)[key])
    grouped = data.groupby([key, "estimator_id"])["mi_estimate"].agg(["mean", "std"]).reset_index()

    for estimator in grouped['estimator_id'].unique():
        subset = grouped[grouped['estimator_id'] == estimator]

        color = ESTIMATOR_COLORS[estimator]
        ax.plot(subset[key], subset['mean'], color=color)
        ax.scatter(subset[key], subset['mean'], color=color, marker=ESTIMATOR_MARKERS[estimator], label=estimator)
        ax.fill_between(subset[key], subset['mean'] - subset['std'], subset['mean'] + subset['std'], alpha=0.3, color=color)

    if use_legend:
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))

def plot_ground_truth_mixing(inputs, ax):
    mixings = []
    means = []
    errs = []
    for pth in inputs:
        with open(pth) as fh:
            dct = json.load(fh)
            mixings.append(dct["mixing"])
            means.append(dct["mi_mean"])
            errs.append(dct["mi_std"])
    
    mixings = np.array(mixings)
    means = np.array(means)
    errs = np.array(errs)
    idx = np.argsort(mixings)
    mixings = mixings[idx]
    means = means[idx]
    errs = errs[idx]
    ax.plot(mixings, means, c="black", linestyle=":")
    ax.fill_between(mixings, means - errs, means + errs, alpha=0.1, color="black")

rule plot:
    input:
        estimates = "results.csv",
        ground_truth_inlier = expand("Gauss-Inlier/_mixed_ground_truth/summaries/{mixing}.json", mixing=ALPHAS),
        ground_truth_outlier = expand("Gauss-Outlier/_mixed_ground_truth/summaries/{mixing}.json", mixing=ALPHAS)
    output: "outliers_plot.pdf"
    run:
        fig, axs = plt.subplots(1, 3, figsize=(8, 2), sharey=True)
        df = pd.read_csv(str(input.estimates))

        for ax in axs:
            ax.spines[['right', 'top']].set_visible(False)

        # Plot inliers
        ax = axs[0]
        plot_data(ax, data=df[df["task_id"].str.contains("mixing-Gauss-Inlier")].copy())
        plot_ground_truth_mixing(input.ground_truth_inlier, ax)
        ax.set_xlabel("Contamination level")
        ax.set_ylabel("MI estimate")
        ax.set_xlim(np.min(ALPHAS), np.max(ALPHAS))

        # Plot outliers
        ax = axs[1]
        plot_data(ax, data=df[df["task_id"].str.contains("mixing-Gauss-Outlier")].copy())
        plot_ground_truth_mixing(input.ground_truth_outlier, ax)
        ax.set_xlabel("Contamination level")
        ax.set_ylabel("MI estimate")
        ax.set_xlim(np.min(ALPHAS), np.max(ALPHAS))

        # Plot variance
        ax = axs[2]
        plot_data(ax, data=df[df["task_id"].str.contains("variance")].copy(), key="variance", use_legend=True)
        mi_var = [VARIANCE_TASKS[f"variance-{var}"].mutual_information for var in VARIANCES]
        ax.plot(VARIANCES, mi_var, c="black", linestyle=":")

        ax.set_xlabel("Noise variance")
        ax.set_ylabel("MI estimate")
        ax.set_xscale("log")
        ax.set_xlim(np.min(VARIANCES), np.max(VARIANCES))


        fig.tight_layout()
        fig.savefig(str(output))


rule make_one_setup:
    output: touch("{setup}/mixing_ground_truths.done")
    input:
        signal_estimate = "{setup}/signal.json",
        mixing_ground_truth = expand("{setup}/_mixed_ground_truth/summaries/{mixing}.json", mixing=ALPHAS, allow_missing=True)


rule estimate_mi_mixed_all_seeds:
    output: "{setup}/_mixed_ground_truth/summaries/{mixing}.json"
    input: expand("{setup}/_mixed_ground_truth/seeds/{mixing}-{seed}.json", seed=SEEDS, allow_missing=True) 
    run:
        estim_dicts = []
        for inp in input:
            with open(inp) as fh:
                estim_dicts.append(json.load(fh))

        mi_mean = np.mean([estim["mi"] for estim in estim_dicts])
        mi_std = np.std([estim["mi"] for estim in estim_dicts], ddof=1)
        mi_stderr_mean = np.mean([estim["mi_stderr"] for estim in estim_dicts])

        with open(str(output), "w") as fh:
            json.dump({
                "mi_mean": float(mi_mean),
                "mi_std": float(mi_std),
                "mi_stderr_mean": float(mi_stderr_mean),
                "mixing": float(wildcards.mixing),
                "samples": estim_dicts,
            },
            fh,
            indent=4,
        )

rule estimate_mi_mixed_one_seed:
    """Estimate MI in the signal distribution for one seed.
    This is not dependent on the mixing."""
    output: "{setup}/_mixed_ground_truth/seeds/{mixing}-{seed}.json"
    run:
        N_SAMPLES: int = 100_000
        setup = CHANGE_MIXING_SETUPS[wildcards.setup]
        dist = setup.mixture(alpha=float(wildcards.mixing))

        key = jax.random.PRNGKey(int(wildcards.seed))
        mi, mi_stderr = bmm.monte_carlo_mi_estimate(key=key, dist=dist, n=N_SAMPLES)
        mi, mi_stderr = float(mi), float(mi_stderr)
        with open(str(output), "w") as fh:
            json.dump({
                "mi": mi,
                "mi_stderr": mi_stderr,
                "mixing": float(wildcards.mixing),
            },
            fh,
            indent=4,
        )


rule estimate_mi_signal_one_seed:
    """Estimate MI in the signal distribution for one seed.
    This is not dependent on the mixing."""
    output: "{setup}/_signal_seeds/{seed}.json"
    run:
        N_SAMPLES: int = 100_000
        setup = CHANGE_MIXING_SETUPS[wildcards.setup]
        dist = setup.dist_true

        key = jax.random.PRNGKey(int(wildcards.seed))
        mi, mi_stderr = bmm.monte_carlo_mi_estimate(key=key, dist=dist, n=N_SAMPLES)
        mi, mi_stderr = float(mi), float(mi_stderr)
        with open(str(output), "w") as fh:
            json.dump({
                "mi": mi,
                "mi_stderr": mi_stderr,
            },
            fh,
            indent=4,
        )

N_SEEDS_GROUND_TRUTH: int = 2
rule estimate_mi_signal_all_seeds:
    """Estimate MI in the signal distribution for all seeds, assembling other runs."""
    output: "{setup}/signal.json"
    input: expand("{setup}/_signal_seeds/{seed}.json", seed=range(N_SEEDS_GROUND_TRUTH), allow_missing=True)
    run:
        estim_dicts = []
        for inp in input:
            with open(inp) as fh:
                estim_dicts.append(json.load(fh))

        mi_analytic = CHANGE_MIXING_SETUPS[wildcards.setup].dist_true.analytic_mi
        mi_mean = np.mean([estim["mi"] for estim in estim_dicts])
        mi_std = np.std([estim["mi"] for estim in estim_dicts], ddof=1)
        mi_stderr_mean = np.mean([estim["mi_stderr"] for estim in estim_dicts])

        with open(str(output), "w") as fh:
            json.dump({
                "mi_analytic": float(mi_analytic),
                "mi_mean": float(mi_mean),
                "mi_std": float(mi_std),
                "mi_stderr_mean": float(mi_stderr_mean),
                "samples": estim_dicts,
            },
            fh,
            indent=4,
        )


include: "_benchmark_rules.smk"
