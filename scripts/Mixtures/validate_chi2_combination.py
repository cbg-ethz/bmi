"""This script is used to validate the chi2 combination for
the multivariate normal PMI profile."""
import jax
import matplotlib.pyplot as plt
import numpy as np

import bmi.samplers as samplers
import bmi.samplers._tfp as bmi_tfp


def plot_samples(ax: plt.Axes, *, rho: np.ndarray, additional_y: int = 0) -> None:
    """Plots the PMI profile as well as the mean and 1 std region on `ax`.

    Covariance matrix is controlled by `rho` (canonical correlations)
    and `additional_y` as `dim_x = len(rho)` and `dim_y = dim_x + additional_y`.
    """
    dim_x = len(rho)
    dim_y = dim_x + additional_y

    # Covariance matrix
    cov = samplers.canonical_correlation(rho, additional_y=additional_y)

    # Mutual information formulas
    mi1 = 0.5 * np.log(
        np.linalg.det(cov[:dim_x, :dim_x])
        * np.linalg.det(cov[dim_x:, dim_x:])
        / np.linalg.det(cov)
    )
    mi2 = -0.5 * np.sum(np.log(1 - rho**2))
    assert abs(mi1 - mi2) < 0.01

    dist = bmi_tfp.MultivariateNormalDistribution(dim_x=dim_x, dim_y=dim_y, covariance=cov)

    key = jax.random.PRNGKey(21)
    key, *subkeys = jax.random.split(key, 3)

    # Sample PMI profile
    n_samples: int = 20_000
    profile = bmi_tfp.pmi_profile(subkeys[0], dist, n=n_samples)

    # Construct profile by combining chi2 distributions
    chi2_samples = np.square(jax.random.normal(subkeys[1], shape=(2 * len(rho), n_samples)))
    chi2_profile = np.full(shape=(n_samples,), fill_value=mi1)
    for i, r in enumerate(rho):
        # We use +- 0.5 * rho weights
        weight = 0.5 * r
        chi2_profile = chi2_profile + weight * (chi2_samples[i, :] - chi2_samples[i + len(rho), :])

    # Calculate variance in two manners
    var_analytic = np.sum(np.square(rho))
    var_empirical = np.var(profile)

    # Plot the histograms
    bins = np.linspace(-2, 4, 21)
    ax.hist(profile, bins=bins, density=True, histtype="step", label="PMI")
    ax.hist(chi2_profile, bins=bins, density=True, histtype="step", label="$\\chi^2$ combination")

    ax.legend()

    # Plot the variances
    ax.set_title(
        f"Variances: $\\sum \\rho_i^2$ = {var_analytic:.2f}, samples = {var_empirical:.2f}"
    )
    ax.axvline(mi1, linestyle="--", color="black", alpha=0.8)

    # Fill 1 std region from the mean
    std = np.sqrt(var_analytic)
    ax.axvspan(mi1 - std, mi1 + std, alpha=0.1, color="navy")


def main() -> None:
    fig, axs = plt.subplots(2, 2, figsize=(8, 6), dpi=300)

    plot_samples(axs[0, 0], rho=np.array([0.8, 0.5]))
    plot_samples(axs[0, 1], rho=np.array([0.7, 0.7, 0.7]))

    # These PMI profiles should be the same
    plot_samples(axs[1, 0], rho=np.array([0.5, 0.0, 0.0]), additional_y=1)
    plot_samples(axs[1, 1], rho=np.array([0.5, 0.0, 0.0]), additional_y=3)

    fig.tight_layout()
    fig.savefig("validate_chi2_combination.pdf", dpi=300)


if __name__ == "__main__":
    main()
