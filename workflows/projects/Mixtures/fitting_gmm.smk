import dataclasses
from typing import Optional
import numpy as np
import jax
import jax.numpy as jnp

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import bmi
from bmi.samplers import fine

from subplots_from_axsize import subplots_from_axsize

import example_distributions as ed


N_WARMUP: int = 2000
N_MCMC_SAMPLES: int = 800
N_MC_PROFILE_SAMPLES: int = 100_000


# === WORKDIR ===
workdir: "generated/mixtures/fitting_gmm/"


def model(data, K: int = 10, alpha: Optional[float] = None, jitter: float = 1e-6, lkj_conc: float = 1.0) -> None:
    """Builds a Gaussian mixture model.
    
    Args:
        data: shape (n_points, n_dim)
        K: number of components
        alpha: concentration parameter for the Dirichlet prior
        jitter: jitter for the covariance matrix to assure that it is positive definite
        lkj_conc: concentration parameter for the LKJ prior. Use 1.0 for uniform prior.
                
    Note:
        To use sparse Dirichlet prior we advise setting alpha = 1/K.
    
    Sample attributes:
        "pi": mixing proportions, shape (K,)
        "mu": component means, shape (K, n_dim)
        "cov": component covariances, shape (K, n_dim, n_dim)
    """
    alpha = alpha or 1.0 / K

    n_points, n_dim = data.shape
    
    # Prior for mixing proportions
    pi = numpyro.sample("pi", dist.Dirichlet(concentration=alpha * jnp.ones(K)))
    
    with numpyro.plate("components", K):
        # Prior for component means
        mu = numpyro.sample("mu", dist.MultivariateNormal(loc=jnp.zeros(n_dim), scale_tril=3*jnp.eye(n_dim)))
        
        # Prior for component covariances using LKJ and HalfCauchy
        
        with numpyro.plate("dimensions", n_dim):
            tau_ = numpyro.sample("tau", dist.HalfCauchy(1.0))

        tau = tau_.T  # (components, dimensions)
    
        # (components, dimensions, dimensions)
        L_omega = numpyro.sample("L_omega", dist.LKJCholesky(n_dim, concentration=lkj_conc))
        L_scaled = tau[:, :, None] * L_omega + jitter * jnp.eye(n_dim)[None, :, :]
    
        cov_matrix = L_scaled @ jnp.transpose(L_scaled, (0, 2, 1))
        numpyro.deterministic("cov", cov_matrix)
        
    # Likelihood
    numpyro.sample("obs", dist.MixtureSameFamily(
        mixing_distribution=dist.Categorical(probs=pi),
        component_distribution=dist.MultivariateNormal(mu, scale_tril=L_scaled)),
        obs=data)


def sample_into_fine_distribution(
    means: jnp.ndarray,
    covariances: jnp.ndarray,
    proportions: jnp.ndarray,
    dim_x: int,
    dim_y: int,
) -> fine.JointDistribution:
    """Builds a fine distribution from a Gaussian mixture model parameters."""
    # Check if the dimensions are right
    n_components = proportions.shape[0]
    n_dims = dim_x + dim_y
    assert means.shape == (n_components, n_dims)
    assert covariances.shape == (n_components, n_dims, n_dims)
    
    # Build components
    components = [
        fine.MultivariateNormalDistribution(
            dim_x=dim_x,
            dim_y=dim_y,
            mean=mean,
            covariance=cov,
        )
        for mean, cov in zip(means, covariances)
    ]

    # Build a mixture model
    return fine.mixture(proportions=proportions, components=components)

DISTRIBUTIONS = {
    "Galaxy": ed.create_galaxy_distribution(_sample=3),
    "AI": ed.create_ai_distribution(_sample=3),
    "Waves": ed.create_waves_distribution(_sample=3),
    "X": ed.create_x_distribution(_sample=3),
}

rule all:
    # For the main part of the manuscript
    input:
        expand("plots/{dist_name}-{n_points}-10.pdf", dist_name=["AI", "Galaxy"], n_points=[500])


rule plots_all:
    input:
        expand("plots/{dist_name}-{n_points}-10.pdf", dist_name=DISTRIBUTIONS.keys(), n_points=[125, 250, 500, 1000])


rule sample_dist:
    output: "data/{dist_name}-{n_points}.npz"
    run:
        example_dist = DISTRIBUTIONS[wildcards.dist_name]
        n_points = int(wildcards.n_points)

        xs, ys = example_dist.sampler.sample(n_points, rng=0)
        np.savez(
            str(output),
            xs=xs,
            ys=ys,
        )


rule fit_model:
    input: "data/{dist_name}-{n_points}.npz"
    output: "gmm_fit/{dist_name}-{n_points}-{n_components}.npz"
    run:
        n_components: int = int(wildcards.n_components)

        arrays = np.load(str(input))
        xs, ys = arrays["xs"], arrays["ys"]
        pts = np.hstack([xs, ys])

        # MCMC Sampling
        nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel, num_samples=N_MCMC_SAMPLES, num_warmup=N_WARMUP)
        mcmc.run(jax.random.PRNGKey(1024), data=pts, K=n_components)

        samples = mcmc.get_samples()
        np.savez(
            str(output),
            **samples,
        )


def visualise_points(xs, ys, ax):
    assert len(xs) == len(ys), f"X and Y have different lengths: {len(xs)} and {len(ys)}"
    assert ys.shape[1] == 1, f"Y dimension is {ys.shape[1]} and cannot be visualised"
    
    dim_x = xs.shape[1]

    if dim_x == 2:
        ax.scatter(xs[..., 0], xs[..., 1], c=ys, s=3, rasterized=True)
        ax.set_xlabel("$X_1$")
        ax.set_ylabel("$X_2$")
    elif dim_x == 1:
        ax.scatter(xs[..., 0], ys[..., 0], c="k", s=3, alpha=0.3, rasterized=True)
        ax.set_xlabel("$X$")
        ax.set_ylabel("$Y$")
    else:
        raise ValueError(f"X dimension is {dim_x} and cannot be visualised")

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ticks = [-1, 0, 1]
    ax.set_xticks(ticks, ticks)
    ax.set_yticks(ticks, ticks)

rule plot_pdf:
    input:
        pmi_true = "pmi_true/{dist_name}-{n_points}.npz",
        pmi_approx = "pmi_collected/{dist_name}-{n_points}-{n_components}.npz",
        true_sample = "data/{dist_name}-{n_points}.npz",
        approx_sample = "approx_samples/{dist_name}-{n_points}-{n_components}-0.npz",
    output: "plots/{dist_name}-{n_points}-{n_components}.pdf"
    run:
        fig, axs = subplots_from_axsize(1, 4, axsize=(1.2, 1.2), top=0.3, wspace=[0.3, 0.05, 0.05], left=0.5, right=0.15)

        for ax in axs:
            ax.spines[['right', 'top']].set_visible(False)

        FONTDICT = {'fontsize': 10}

        # Visualise true sample
        ax = axs[0]
        ax.set_title("Ground-truth sample", fontdict=FONTDICT)
        true_sample = np.load(input.true_sample)
        visualise_points(true_sample["xs"], true_sample["ys"], ax)

        # Visualise approximate sample
        ax = axs[1]
        ax.set_title("Simulated sample", fontdict=FONTDICT)
        approx_sample = np.load(input.approx_sample)
        visualise_points(approx_sample["xs"], approx_sample["ys"], ax)

        pmi_true = np.load(input.pmi_true)["pmi"]
        pmi_approx = np.load(input.pmi_approx)["pmi"]  # (num_mcmc_samples, num_mc_evaluations)

        # Visualise posterior on mutual information
        ax = axs[2]
        ax.set_title("Posterior MI", fontdict=FONTDICT)
        mi_true = np.mean(pmi_true)
        mi_approx = np.mean(pmi_approx, axis=1)  # (num_mcmc_samples,)
        ax.set_xlabel("MI")

        ax.hist(mi_approx, bins=50, density=True, alpha=0.5, color="red", rasterized=True)
        ax.axvline(mi_approx.mean(), color="red")  # Visualise posterior mean
        ax.axvline(mi_true, color="k", linestyle=":")  # Visualise true value

        # Visualise posterior on profile
        ax = axs[3]
        ax.set_title("Posterior PMI profile", fontdict=FONTDICT)
        ax.set_xlabel("PMI")

        quantile_min = 0.02
        quantile_max = 1 - quantile_min
        min_val = np.min([np.quantile(pmi_true, quantile_min), np.quantile(pmi_approx, quantile_min)])
        max_val = np.max([np.quantile(pmi_true, quantile_max), np.quantile(pmi_approx, quantile_max)])

        bins = np.linspace(min_val, max_val, 50)
        for pmi_vals in pmi_approx:
            prof, _ = np.histogram(pmi_vals, bins=bins, density=True)
            ax.stairs(prof, edges=bins, color="red", alpha=0.05, rasterized=True)
        
        prof_true, _ = np.histogram(pmi_true, bins=bins, density=True)
        ax.stairs(prof_true, edges=bins, color="k", alpha=1, rasterized=True)

        for ax in [axs[2], axs[3]]:
            ax.set_ylabel("")
            ax.set_yticks([])
            ax.spines[['right', 'top', 'left']].set_visible(False)

        fig.savefig(str(output), dpi=300)

rule get_pmi_true:
    output: "pmi_true/{dist_name}-{n_points}.npz"
    run:
        dist = DISTRIBUTIONS[wildcards.dist_name].dist
        xs, ys = dist.sample(N_MC_PROFILE_SAMPLES, jax.random.PRNGKey(2048))
        pmi_true = dist.pmi(xs, ys)
        np.savez(str(output), pmi=pmi_true)


rule collect_pmi_profiles:
    output: "pmi_collected/{dist_name}-{n_points}-{n_components}.npz"
    input: expand("pmi_individual/{dist_name}-{n_points}-{n_components}-{sample_index}.npz", sample_index=range(N_MCMC_SAMPLES), allow_missing=True)
    run:
        arrs = []
        for pth in input:
            arrs.append(np.load(str(pth))["pmi"])

        np.savez(
            str(output),
            pmi=np.asarray(arrs),
        )


rule create_approx_sample:
    """Samples from the approximate distribution data set of original size."""
    output: "approx_samples/{dist_name}-{n_points}-{n_components}-{sample_index}.npz"
    input: "gmm_fit/{dist_name}-{n_points}-{n_components}.npz"
    run:
        true_distribution = DISTRIBUTIONS[wildcards.dist_name].dist
        dim_x = true_distribution.dim_x
        dim_y = true_distribution.dim_y

        samples = np.load(str(input))
        
        idx = int(wildcards.sample_index)

        approx_dist = sample_into_fine_distribution(
            means=samples["mu"][idx],
            covariances=samples["cov"][idx],
            proportions=samples["pi"][idx],
            dim_x=dim_x,
            dim_y=dim_y,
        )

        xs, ys = approx_dist.sample(int(wildcards.n_points), jax.random.PRNGKey(2048))
        np.savez(str(output), xs=xs, ys=ys)


rule estimate_pmi_in_sample:
    """Construct the GMM approximation using the given sample index"""
    input: "gmm_fit/{dist_name}-{n_points}-{n_components}.npz"
    output: "pmi_individual/{dist_name}-{n_points}-{n_components}-{sample_index}.npz"
    run:
        true_distribution = DISTRIBUTIONS[wildcards.dist_name].dist
        dim_x = true_distribution.dim_x
        dim_y = true_distribution.dim_y

        samples = np.load(str(input))
        
        idx = int(wildcards.sample_index)

        approx_dist = sample_into_fine_distribution(
            means=samples["mu"][idx],
            covariances=samples["cov"][idx],
            proportions=samples["pi"][idx],
            dim_x=dim_x,
            dim_y=dim_y,
        )

        xs, ys = approx_dist.sample(N_MC_PROFILE_SAMPLES, jax.random.PRNGKey(2048))
        approx_pmi = approx_dist.pmi(xs, ys)

        np.savez(str(output), pmi=approx_pmi)
