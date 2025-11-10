import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import bmi
from bmi.samplers import fine

n_dim = 5
n_points: int = 5_000
ks = (5, 10, 20, 50)

dist = fine.MultivariateNormalDistribution(
    dim_x=n_dim,
    dim_y=n_dim,
    mean=jnp.zeros(2 * n_dim),
    covariance=bmi.samplers.canonical_correlation([0.8] * n_dim),
)

mi, mi_stderr = fine.monte_carlo_mi_estimate(jax.random.PRNGKey(10), dist, n=10_000)


xs, ys = dist.sample(n_points, jax.random.PRNGKey(42))
pmis = dist.pmi(xs, ys)

min_pmi = jnp.min(pmis) - 0.1
max_pmi = jnp.max(pmis) + 0.1


fig, axs = plt.subplots(len(ks), 3, figsize=(6, 2 * len(ks)), dpi=250)

for i, k in enumerate(ks):
    estimator = bmi.estimators.KSGEnsembleFirstEstimatorSlow(neighborhoods=(k,), standardize=False)

    pseudo_pmis = estimator._calculate_digammas(xs, ys, ks=(k,))[k]

    bins = jnp.linspace(min_pmi, max_pmi, 21)

    ax = axs[i, 0]
    ax.hist(pmis, bins=bins, density=True)
    ax.set_title("True PMI")

    ax.set_xlabel(f"$I(X; Y) = {mi:.2f}$")
    ax.set_ylabel(f"$k={k}$")

    ax = axs[i, 1]
    ax.hist(pseudo_pmis, bins=bins, density=True)
    ax.set_title("KSG PMI")
    ax.set_xlabel(f"$I(X; Y) = {np.mean(pseudo_pmis):.2f}$")

    ax = axs[i, 2]
    ts = jnp.linspace(min_pmi, max_pmi, 3)
    ax.plot(ts, ts, color="maroon", linestyle="--")

    ax.scatter(pmis, pseudo_pmis, s=2, alpha=0.1, c="k")
    ax.set_xlabel("True PMI")
    ax.set_ylabel("KSG PMI")

    ax.set_xlim(min_pmi, max_pmi)
    ax.set_ylim(min_pmi, max_pmi)
    ax.set_aspect("equal")

    corr = np.corrcoef(pmis, pseudo_pmis)[0, 1]
    ax.set_title(f"$r={corr:.2f}$")

fig.tight_layout()

fig.savefig("figure.pdf")
