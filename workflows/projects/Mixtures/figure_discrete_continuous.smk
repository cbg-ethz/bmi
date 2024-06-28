"""Distribution with continuous two-dimensional X and binary Y."""
import json
from jax import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np

import bmi
from bmi.samplers import bmm

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions


# === WORKDIR ===
workdir: "generated/mixtures/discrete_continuous/"


def construct_bernoulli(p: float, dtype=jnp.float64) -> tfd.Distribution:
    """Constructs a Bernoulli distribution, as
    TensorFlow Probability disallows products of continuous
    and discrete distributions."""
    return tfd.Independent(
            tfd.Bernoulli(probs=jnp.asarray([p], dtype=dtype), dtype=dtype),
            reinterpreted_batch_ndims=1,
    )


def define_distribution():
    """Defines the joint distribution P(X, Y)."""
    # Define the distributions for X_k
    x1 = bmm.construct_multivariate_student_distribution(
        mean=-jnp.ones(2),
        dispersion=0.2 * jnp.eye(2),
        df=8,
    )
    x2 = bmm.construct_multivariate_normal_distribution(
        mean=jnp.zeros(2),
        covariance=0.1 * bmi.samplers.canonical_correlation([0.95]),
    )
    x3 = bmm.construct_multivariate_student_distribution(
        mean=jnp.ones(2),
        dispersion=0.2 * jnp.eye(2),
        df=5,
    )

    # Define the distributions for Y_k
    y1 = construct_bernoulli(0.95)
    y2 = construct_bernoulli(0.5)
    y3 = construct_bernoulli(0.05)

    # Define the mixture distribution
    components = [
        bmm.ProductDistribution(dist_x, dist_y)
        for dist_x, dist_y in zip([x1, x2, x3], [y1, y2, y3])
    ]
    connect_prob = 0.5
    bulk_prob = 0.5 * (1 - connect_prob)
    joint_distribution = bmm.mixture(proportions=[bulk_prob, connect_prob, bulk_prob], components=components)

    return joint_distribution

rule all:
    input:
        "estimate.json",
        "figure_discrete_continuous_example.pdf"


rule estimate_mi:
    output: "estimate.json"
    run:
        key = jax.random.PRNGKey(121)
        mi, mi_std_err = bmm.monte_carlo_mi_estimate(key, dist=define_distribution(), n=1_000_000)
        with open(output[0], "w") as fp:
            json.dump({"estimate": float(mi), "std_err": float(mi_std_err)}, fp=fp)

rule sample_joint:
    output: "joint_samples.npz"
    run:
        key = jax.random.PRNGKey(1000)
        x, y = define_distribution().sample(1000, key)
        np.savez(output[0], x=x, y=y)

rule sample_pmi:
    output: "pmi_samples.npz"
    run:
        key = jax.random.PRNGKey(101)
        profile_samples = bmm.pmi_profile(key=key, dist=define_distribution(), n=1_000_000)
        np.savez(output[0], samples=profile_samples)


rule plot_figure:
    input:
        joint_samples="joint_samples.npz",
        pmi_samples="pmi_samples.npz"
    output:
        figure = "figure_discrete_continuous_example.pdf"
    run:
        samples = np.load(input.joint_samples)
        xs, ys = samples["x"], samples["y"]
        pmis = np.load(input.pmi_samples)["samples"]

        fig, axs = plt.subplots(1, 2, figsize=(5, 2.5), dpi=300)

        ax = axs[0]
        colors = ['blue' if y < 0.5 else 'orange' for y in ys.ravel()]
        ax.scatter(xs[..., 0], xs[..., 1], c=colors, s=1, alpha=0.8, rasterized=True, marker=".")
        ax.set_title("Samples from $P_{XY}$")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

        ax = axs[1]
        ax.hist(pmis, bins=100, density=True, color='black', alpha=0.7, rasterized=True)
        ax.set_xlim(-3, 1)
        ax.set_title("PMI profile")
        ax.set_xlabel("PMI")
        ax.set_ylabel("Density")    

        for ax in axs:
            ax.spines[['top', 'right']].set_visible(False)

        fig.tight_layout()
        fig.savefig(output.figure)
