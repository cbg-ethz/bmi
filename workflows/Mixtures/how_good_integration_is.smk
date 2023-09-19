"""This workflow implements the following experiment:
  - We have a distribution with known PMI.
  - We sample N points (varying N) and see what different lower bounds say about the (integrated) MI value.
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
from bmi.estimators.neural._backend_linear_memory import infonce, donsker_varadhan, nwj

def monte_carlo(pmi: Callable, xs: np.ndarray, ys: np.ndarray):
    return jnp.mean(pmi(xs, ys))

def nwj_shifted(pmi: Callable, xs, ys):
    """For NWJ the optimal critic is PMI + 1."""
    def critic(x, y):
        return pmi(x, y) + 1

    return nwj(critic, xs, ys)

@dataclasses.dataclass
class DistributionAndPMI:
    dist: fine.JointDistribution
    # If `pmi` is None, then `dist.pmi` will be used
    pmi: Optional[Callable] = None


# === WORKDIR ===
workdir: "generated/mixtures/how_good_integration_is"


ESTIMATORS: dict[str, Callable] = {
    "NWJ": nwj,
    "NWJ-Shifted": nwj_shifted,
    "InfoNCE": infonce,
    "DV": donsker_varadhan,
    "MC": monte_carlo
}

_normal_dist = fine.MultivariateNormalDistribution(dim_x=2, dim_y=2, covariance=bmi.samplers.canonical_correlation(rho=[0.8, 0.8])) 
_DISTRIBUTIONS: dict[str, DistributionAndPMI] = {
    "Normal": DistributionAndPMI(
        dist=_normal_dist,
    ),
    "NormalBiased": DistributionAndPMI(
        dist=_normal_dist,
        pmi=lambda x, y: _normal_dist.pmi(x, y) + 0.5,
    ),
    "NormalSinSquare": DistributionAndPMI(
        dist=_normal_dist,
        pmi=lambda x, y: _normal_dist.pmi(x, y) + jnp.sin(jnp.square(x[..., 0])),
    ),
    "Student": DistributionAndPMI(
        dist=fine.MultivariateStudentDistribution(dim_x=2, dim_y=2, dispersion=bmi.samplers.canonical_correlation(rho=[0.8, 0.8]), df=3),
    ),
}
# If PMI is left as None, override it with the PMI of the distribution
DISTRIBUTION_AND_PMIS = {
    name: DistributionAndPMI(dist=value.dist, pmi=value.dist.pmi) if value.pmi is None else value for name, value in _DISTRIBUTIONS.items()
}

N_POINTS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
SEEDS = list(range(10))

rule all:
    input:
        ground_truth = expand("{setup}/ground_truth.json", setup=DISTRIBUTION_AND_PMIS),
        estimates=expand("{setup}/estimates.csv", setup=DISTRIBUTION_AND_PMIS),
        performance_plots=expand("{setup}/performance.pdf", setup=DISTRIBUTION_AND_PMIS)


rule sample_distribution:
    output: "{setup}/samples/{n_points}-{seed}.npz"
    run:
        setup = DISTRIBUTION_AND_PMIS[wildcards.setup]
        dist = setup.dist
        key = jax.random.PRNGKey(int(wildcards.seed))
        xs, ys = dist.sample(int(wildcards.n_points), key)

        np.savez(str(output), xs=xs, ys=ys)


rule apply_estimator:
    input: "{setup}/samples/{n_points}-{seed}.npz"
    output: "{setup}/estimates/{estimator}/{n_points}-{seed}.json"
    run:
        samples = np.load(str(input))
        xs, ys = samples["xs"], samples["ys"]

        setup = DISTRIBUTION_AND_PMIS[wildcards.setup]
        estimator = ESTIMATORS[wildcards.estimator]

        estimate = float(estimator(setup.pmi, xs, ys))
        with open(str(output), "w") as fh:
            json.dump({
                "estimator": wildcards.estimator,
                "setup": wildcards.setup,
                "estimate": estimate,
                "n_points": int(wildcards.n_points),
            },
            fh,
            indent=4,
        )

rule assemble_estimates:
    input: expand("{setup}/estimates/{estimator}/{n_points}-{seed}.json", n_points=N_POINTS, seed=SEEDS, estimator=ESTIMATORS, allow_missing=True)
    output: "{setup}/estimates.csv"
    run:
        dcts = []
        for inp in input:
            with open(inp) as fh:
                dcts.append(json.load(fh))
        df = pd.DataFrame(dcts)
        df.to_csv(str(output), index=False)


_N_SEEDS_GROUND_TRUTH: int = 10
rule estimate_ground_truth:
    output: "{setup}/ground_truth.json"
    input: expand("{setup}/_ground_truth_seeds/{seed}.json", seed=range(_N_SEEDS_GROUND_TRUTH), allow_missing=True)
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
                "samples": estim_dicts,
            },
            fh,
            indent=4,
        )


rule estimate_ground_truth_single_seed:
    output: "{setup}/_ground_truth_seeds/{seed}.json"
    run:
        N_SAMPLES: int = 100_000
        dist = DISTRIBUTION_AND_PMIS[wildcards.setup].dist
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

rule plot_performance:
    input:
        ground_truth="{setup}/ground_truth.json",
        estimates="{setup}/estimates.csv"
    output: "{setup}/performance.pdf"
    run:
        df = pd.read_csv(input.estimates)
        with open(input.ground_truth) as fh:
            ground_truth = json.load(fh)
        
        fig, ax = plt.subplots(figsize=(4, 3), dpi=150)

        # Add ground-truth information
        x_axis =[df["n_points"].min(), df["n_points"].max()] 
        ax.plot(x_axis, [ground_truth["mi_mean"]] * 2, c="k", linestyle=":")
        ax.fill_between(
            x_axis,
            [ground_truth["mi_mean"] - ground_truth["mi_std"]] * 2,
            [ground_truth["mi_mean"] + ground_truth["mi_std"]] * 2,
            alpha=0.3,
            color="k",
        )


        # Plot means for estimators
        grouped = df.groupby(['n_points', 'estimator']).estimate.agg(['mean', 'std']).reset_index()
        sns.lineplot(x='n_points', y='mean', hue='estimator', data=grouped, palette='tab10', ax=ax)

        # Plot standard deviations
        for estimator in grouped['estimator'].unique():
            subset = grouped[grouped['estimator'] == estimator]
            ax.fill_between(subset['n_points'], subset['mean'] - subset['std'], subset['mean'] + subset['std'], alpha=0.1)

        ax.set_xlabel("Number of points")
        ax.set_ylabel('Estimate')

        ax.set_xscale("log", base=2)
        ax.set_xticks(df["n_points"].unique(), df["n_points"].unique())
        ax.legend(title='Estimator', frameon=False)

        fig.tight_layout()
        fig.savefig(str(output))