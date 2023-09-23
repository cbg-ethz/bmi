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
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from subplots_from_axsize import subplots_from_axsize

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
    "NWJ": nwj_shifted,
    "InfoNCE": infonce,
    "DV": donsker_varadhan,
    "MC": monte_carlo
}

ESTIMATOR_COLORS = {
    "InfoNCE": "magenta",
    "DV": "red",
    "NWJ": "limegreen",
    "MC": "mediumblue",
}

four_balls = fine.mixture(
    proportions=jnp.array([0.3, 0.3, 0.2, 0.2]),
    components=[
        fine.MultivariateNormalDistribution(
            covariance=bmi.samplers.canonical_correlation([0.0]),
            mean=jnp.array([-1.25, -1.25]),
            dim_x=1, dim_y=1,
        ),
        fine.MultivariateNormalDistribution(
            covariance=bmi.samplers.canonical_correlation([0.0]),
            mean=jnp.array([+1.25, +1.25]),
            dim_x=1, dim_y=1,
        ),
        fine.MultivariateNormalDistribution(
            covariance=0.2 * bmi.samplers.canonical_correlation([0.0]),
            mean=jnp.array([-2.5, +2.5]),
            dim_x=1, dim_y=1,
        ),
        fine.MultivariateNormalDistribution(
            covariance=0.2 * bmi.samplers.canonical_correlation([0.0]),
            mean=jnp.array([+2.5, -2.5]),
            dim_x=1, dim_y=1,
        ),
    ]
)


_DISTRIBUTIONS: dict[str, DistributionAndPMI] = {
    "Four_Balls": DistributionAndPMI(
        dist=four_balls,
    ),
    "Four_Balls_Biased": DistributionAndPMI(
        dist=four_balls,
        pmi=lambda x, y: four_balls.pmi(x, y) + 0.5,
    ),
    "Four_Balls_SinSquare": DistributionAndPMI(
        dist=four_balls,
        pmi=lambda x, y: four_balls.pmi(x, y) + jnp.sin(jnp.square(x[..., 0])),
    ),
    "Normal-25Dim": DistributionAndPMI(
        dist=fine.MultivariateNormalDistribution(dim_x=25, dim_y=25, covariance=bmi.samplers.canonical_correlation(rho=[0.8] * 25))
    ),
}
# If PMI is left as None, override it with the PMI of the distribution
DISTRIBUTION_AND_PMIS = {
    name: DistributionAndPMI(dist=value.dist, pmi=value.dist.pmi) if value.pmi is None else value for name, value in _DISTRIBUTIONS.items()
}

N_POINTS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
SEEDS = list(range(20))


rule all:
    input:
        ground_truth = expand("{setup}/ground_truth.json", setup=DISTRIBUTION_AND_PMIS),
        estimates=expand("{setup}/estimates.csv", setup=DISTRIBUTION_AND_PMIS),
        performance_plots=expand("{setup}/performance.pdf", setup=DISTRIBUTION_AND_PMIS),
        plot_for_publication = "how_good_integration_is.pdf"


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

def plot_estimates(ax: plt.Axes, estimates_path, ground_truth_path) -> None:
    df = pd.read_csv(estimates_path)
    with open(ground_truth_path) as fh:
        ground_truth = json.load(fh)

    # Add ground-truth information
    x_axis =[df["n_points"].min(), df["n_points"].max()] 
    ax.plot(x_axis, [ground_truth["mi_mean"]] * 2, c="k", linestyle=":")
    # ax.fill_between(
    #     x_axis,
    #     [ground_truth["mi_mean"] - ground_truth["mi_std"]] * 2,
    #     [ground_truth["mi_mean"] + ground_truth["mi_std"]] * 2,
    #     alpha=0.3,
    #     color="k",
    # )

    grouped = df.groupby(['n_points', 'estimator']).estimate.agg(['mean', 'std']).reset_index()
    for estimator in grouped["estimator"].unique():
        sub_df = grouped[grouped["estimator"] == estimator]
        sub_df = sub_df.sort_values("n_points")
    
        points = sub_df["n_points"].values
        mean = sub_df["mean"].values
        std = sub_df["std"].values
    
        color = ESTIMATOR_COLORS[estimator]
    
        ax.plot(points, mean, color=color, label=estimator)
        ax.fill_between(points, mean - std, mean + std, alpha=0.1, color=color)


rule plot_performance_all:
    input:
        simple_ground_truth="Four_Balls/ground_truth.json",
        simple_estimates = "Four_Balls/estimates.csv",
        biased_ground_truth="Four_Balls_Biased/ground_truth.json",
        biased_estimates = "Four_Balls_Biased/estimates.csv",
        func_ground_truth="Four_Balls_SinSquare/ground_truth.json",
        func_estimates = "Four_Balls_SinSquare/estimates.csv",
        highdim_ground_truth="Normal-25Dim/ground_truth.json",
        highdim_estimates = "Normal-25Dim/estimates.csv"
    output:
        "how_good_integration_is.pdf"
    run:
        fig, axs = subplots_from_axsize(1, 4, axsize=(2.5, 1.5), right=1.2, top=0.3)

        ax = axs[0]
        ax.set_title("Mixture")
        plot_estimates(ax, input.simple_estimates, input.simple_ground_truth)

        ax = axs[1]
        ax.set_title("Constant bias")
        plot_estimates(ax, input.biased_estimates, input.biased_ground_truth)

        ax = axs[2]
        ax.set_title("Functional bias")
        plot_estimates(ax, input.func_estimates, input.func_ground_truth)

        ax = axs[3]
        ax.set_title("High-dimensional")
        plot_estimates(ax, input.highdim_estimates, input.highdim_ground_truth)


        for ax in axs:
            ax.set_xlabel("Number of points")
            ax.set_ylabel('Estimate')
            ax.set_xscale("log", base=2)
            ticks = [16, 64, 256, 1024, 4096]
            ax.set_xticks(ticks, ticks)
            ax.spines[['right', 'top']].set_visible(False)
        
        axs[3].legend(title='Estimator', frameon=False, bbox_to_anchor=(1.01, 1), loc='upper left')

        fig.savefig(str(output))

rule plot_performance_single:
    input:
        ground_truth="{setup}/ground_truth.json",
        estimates="{setup}/estimates.csv"
    output: "{setup}/performance.pdf"
    run:
        df = pd.read_csv(input.estimates)
        with open(input.ground_truth) as fh:
            ground_truth = json.load(fh)
        
        fig, ax = plt.subplots(figsize=(4, 3), dpi=150)

        plot_estimates(ax, input.estimates, input.ground_truth)

        ax.set_xlabel("Number of points")
        ax.set_ylabel('Estimate')

        ax.set_xscale("log", base=2)
        # ax.set_xticks(df["n_points"].unique(), df["n_points"].unique())
        ax.legend(title='Estimator', frameon=False)

        fig.tight_layout()
        fig.savefig(str(output))