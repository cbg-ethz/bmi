"""This workflow tests how the performance of different estimators changes
when mixing with "outlier" distribution is performed.
"""
from typing import Callable, Optional
import dataclasses
import json

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import jax
import jax.numpy as jnp

import bmi
from bmi.samplers import fine


# === WORKDIR ===
workdir: "generated/mixtures/outliers"


@dataclasses.dataclass
class ChangeMixingSetup:
    dist_true: fine.JointDistribution
    dist_noise: fine.JointDistribution

    def mixture(self, alpha: float) -> fine.JointDistribution:
        """Return a mixture of the two distributions.
        
        Args:
            alpha: contamination level, i.e. the probability of the noise distribution
        """
        return fine.mixture(
            components=[self.dist_true, self.dist_noise],
            proportions=jnp.asarray([1.0 - alpha, alpha]),
        )


dist_signal_gauss = fine.MultivariateNormalDistribution(
    dim_x=2,
    dim_y=2,
    covariance=bmi.samplers.SparseLVMParametrization(dim_x=2, dim_y=2, n_interacting=2, beta=0.1, lambd=2.0).correlation
)

covariance_noise = 0.3 * jnp.eye(dist_signal_gauss.dim_y)
dist_gaussian_noise = fine.ProductDistribution(
    dist_x=dist_signal_gauss.dist_x,
    dist_y=fine.construct_multivariate_normal_distribution(mean=jnp.zeros(dist_signal_gauss.dim_y), covariance=covariance_noise),
)

# We will now define a Student distribution with the same covariance matrix
student_dof = 3.0
assert student_dof > 2.0
student_dispersion = covariance_noise * (student_dof - 2.0) / student_dof

dist_student_noise = fine.ProductDistribution(
    dist_x=dist_signal_gauss.dist_x,
    dist_y=fine.construct_multivariate_student_distribution(mean=jnp.zeros(dist_signal_gauss.dim_y), dispersion=student_dispersion, df=student_dof),
)

CHANGE_MIXING_SETUPS = {
    "Gauss-Gauss": ChangeMixingSetup(dist_true=dist_signal_gauss, dist_noise=dist_gaussian_noise),
    "Gauss-Student": ChangeMixingSetup(dist_true=dist_signal_gauss, dist_noise=dist_student_noise),
}


VARIANCES = [0.001, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 2.0, 4.0, 10.0]
ALPHAS = [0.001, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

UNSCALED_TASKS = {}
# Add mixing tasks
for setup_name, setup in CHANGE_MIXING_SETUPS.items():
    for alpha in ALPHAS:
        sampler = fine.FineSampler(dist=setup.mixture(alpha=alpha), mi_estimate_sample=100_000)
        task = bmi.Task(
            sampler=sampler,
            task_id=f"mixing-{setup_name}-{alpha}",
            task_name=f"mixing-{setup_name}-{alpha}",
            task_params={"mixing": alpha},
        )
        UNSCALED_TASKS[task.id] = task

# Add the task with changing the variance
for variance in VARIANCES:
    dist_noise_variance =  fine.ProductDistribution(
        dist_x=dist_signal_gauss.dist_x,
        dist_y=fine.construct_multivariate_normal_distribution(
            mean=jnp.zeros(dist_signal_gauss.dim_y),
            covariance=variance * jnp.eye(dist_signal_gauss.dim_y)
        ),
    )
    sampler = fine.FineSampler(
        dist=fine.mixture(
            proportions=jnp.array([0.9, 0.1]),
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
    UNSCALED_TASKS[task.id] = task


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
        estimates = "results.csv"


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
        mi, mi_stderr = fine.monte_carlo_mi_estimate(key=key, dist=dist, n=N_SAMPLES)
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
        mi, mi_stderr = fine.monte_carlo_mi_estimate(key=key, dist=dist, n=N_SAMPLES)
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