# This workflow creates figure with four axes:
#   1. Samples from bivariate normal distribution.
#   2. Samples from transformed bivariate normal distribution (four modes).
#   3. Mixture of three bivariate normal distributions ("U" shape).
#   4. The PMI profiles of all of these.

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tensorflow_probability.substrates import jax as tfp

import jax
import jax.numpy as jnp

import bmi.samplers._tfp as bmi_tfp
from bmi.transforms import invert_cdf, normal_cdf

mpl.use("Agg")


n_samples = 100_000
correlation: float = 0.805

bivariate_normal = bmi_tfp.MultivariateNormalDistribution(
    dim_x=1,
    dim_y=1,
    covariance=jnp.array([[1.0, correlation], [correlation, 1.0]]),
)

# === WORKDIR ===
workdir: "generated/mixtures/distinct_profiles/"

rule all:
    input: "figure_distinct_profiles.pdf"


rule sample_normal:
    output: "normal.npz"
    run:
        key = jax.random.PRNGKey(0)
        x, y = bivariate_normal.sample(key, n_samples)
        pmi_profile = bivariate_normal.pmi(x, y)

        np.savez(
            str(output),
            samples=jnp.stack([x, y], axis=1),
            pmi_profile=pmi_profile,
        )


rule sample_transformed_normal:
    input: "normal.npz"
    output: "transformed_normal.npz"
    run:
        icdf_x = jax.vmap(invert_cdf(lambda x: 0.3 * normal_cdf(x + 0) + 0.7 * normal_cdf(x - 5), delta=1e-4, x_min=-100.0, x_max=100.0))
        icdf_y = jax.vmap(invert_cdf(lambda x: 0.5 * normal_cdf(x + 1) + 0.5 * normal_cdf(x - 3), delta=1e-4, x_min=-100.0, x_max=100.0))

        old_samples = np.load(str(input))["samples"]

        x = icdf_x(jax.vmap(normal_cdf)(old_samples[:, 0]))
        y = icdf_y(jax.vmap(normal_cdf)(old_samples[:, 1]))

        samples = jnp.stack([x, y], axis=1)
        np.savez(
            str(output),
            samples=samples,
            pmi_profile=np.load(str(input))["pmi_profile"],
        )


rule sample_u_shape:
    output: "u_shape.npz"
    run:
        def diag(a):
            return jnp.diag(jnp.array(a))

        width = 0.2

        dist = bmi_tfp.mixture(
            proportions=[1/3, 1/3, 1/3],
            components=[
                bmi_tfp.MultivariateNormalDistribution(
                dim_x=1,
                dim_y=1,
                mean=jnp.asarray([-3., 0.]),
                covariance=diag([width, 1.0]),
            ),
            bmi_tfp.MultivariateNormalDistribution(
                dim_x=1,
                dim_y=1,
                mean=jnp.asarray([0., -3.]),
                covariance=diag([1.0, width]),
            ),
            bmi_tfp.MultivariateNormalDistribution(
                dim_x=1,
                dim_y=1,
                mean=jnp.asarray([3., 0.]),
                covariance=diag([width, 1.0]),
            ),
            ]
        )

        key = jax.random.PRNGKey(0)
        x, y = dist.sample(key, n_samples)
        pmi_profile = dist.pmi(x, y)
        np.savez(
            str(output),
            samples=jnp.stack([x, y], axis=1),
            pmi_profile=pmi_profile,
        )

def hide_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("$X$")
    ax.set_ylabel("$Y$")

rule plot_samples:
    input:
      normal = "normal.npz",
      transformed_normal = "transformed_normal.npz",
      u_shape = "u_shape.npz"
    output:
        "figure_distinct_profiles.pdf"
    run:
        fig, axs = plt.subplots(1, 4, figsize=(8, 2))

        color1 = "navy"
        color2 = "salmon"

        # Plot normal distribution
        ax = axs[0]
        data = np.load(str(input.normal))["samples"]
        ax.scatter(data[:10_000, 0], data[:10_000, 1], s=1, alpha=0.1, c=color1, rasterized=True)
        hide_ticks(ax)
        ax.set_title("Bivariate normal")

        # Plot transformed normal
        ax = axs[1]
        data = np.load(str(input.transformed_normal))["samples"]
        ax.scatter(data[:10_000, 0], data[:10_000, 1], s=1, alpha=0.1, c=color1, rasterized=True)
        hide_ticks(ax)
        ax.set_title("Transformed")

        # Plot U shape
        ax = axs[2]
        data = np.load(str(input.u_shape))["samples"]
        ax.scatter(data[:10_000, 0], data[:10_000, 1], s=1, alpha=0.1, c=color2, rasterized=True)
        hide_ticks(ax)
        ax.set_title("Mixture")

        # Plot PMI profiles
        ax = axs[3]
        pmi_normal = np.load(str(input.normal))["pmi_profile"]
        pmi_u = np.load(str(input.u_shape))["pmi_profile"]

        bins = np.linspace(-2, 2, 51)
        ax.hist(pmi_normal, bins=bins, density=True, color=color1, alpha=0.5, label="Normal")
        ax.hist(pmi_u, bins=bins, density=True, color=color2, alpha=0.5, label="Mixture")
        ax.set_title("PMI profiles")
        ax.set_xlabel("PMI")
        ax.set_ylabel("Density")

        mi_1 = jnp.mean(pmi_normal)
        mi_2 = jnp.mean(pmi_u)
        
        if abs(mi_1 - mi_2) > 0.01:
            raise ValueError(f"MI different: {mi_1:.2f} != {mi_2:.2f}")
        
        ax.axvline(mi_1, c="k",  linewidth=0.5, linestyle="--")

        fig.tight_layout()
        fig.savefig(str(output))
