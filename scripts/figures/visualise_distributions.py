"""Visualisation of several samplers used."""
import matplotlib.pyplot as plt

import bmi.api as bmi
import bmi.benchmark.tasks.one_dimensional as od


def main() -> None:
    gaussian_correlation: float = od.DEFAULT_CORRELATION
    n_points: int = 5_000
    seed: int = 0
    alpha: float = 0.5
    size: int = 3

    sampler_gaussian = bmi.samplers.BivariateNormalSampler(correlation=gaussian_correlation)
    sampler_uniform = bmi.samplers.BivariateUniformMarginsSampler(
        gaussian_correlation=gaussian_correlation
    )
    sampler_swissroll = bmi.samplers.SwissRollSampler(sampler=sampler_uniform)
    sampler_halfnormal = od.get_half_cube_sampler(gaussian_correlation=gaussian_correlation)
    sampler_bimodal = od.get_bimodal_sampler(gaussian_correlation=gaussian_correlation)

    plt.set_cmap("jet")

    fig = plt.figure(figsize=plt.figaspect(0.19))

    ax = fig.add_subplot(1, 5, 1)
    x, y = sampler_gaussian.sample(n_points=n_points, rng=seed)
    y_original = y.ravel()
    artist = ax.scatter(x.ravel(), y.ravel(), c=y_original, alpha=alpha, s=size)
    artist.set_rasterized(True)
    ax.set_title("Bivariate Gaussian")
    ax.set_xlabel("$X$")
    ax.set_ylabel("$Y$")

    ax = fig.add_subplot(1, 5, 2)
    x, y = sampler_uniform.sample(n_points=n_points, rng=seed)
    artist = ax.scatter(x.ravel(), y.ravel(), c=y_original, alpha=alpha, s=size)
    artist.set_rasterized(True)
    ax.set_title("Correlated uniform")
    ax.set_xlabel("CDF$(X)$")
    ax.set_ylabel("CDF$(Y)$")

    ax = fig.add_subplot(1, 5, 3)
    x, y = sampler_halfnormal.sample(n_points=n_points, rng=seed)
    artist = ax.scatter(x.ravel(), y.ravel(), c=y_original, alpha=alpha, s=size)
    artist.set_rasterized(True)
    ax.set_title("Half-cube mapping")
    ax.set_xlabel("$h(X)$")
    ax.set_ylabel("$h(Y)$")

    ax = fig.add_subplot(1, 5, 4)
    x, y = sampler_bimodal.sample(n_points=n_points, rng=seed)
    artist = ax.scatter(x.ravel(), y.ravel(), c=y_original, alpha=alpha, s=size)
    artist.set_rasterized(True)
    ax.set_title("Making margins bimodal")
    ax.set_xlabel("bimodal$(X)$")
    ax.set_ylabel("bimodal$(Y)$")

    x, y = sampler_swissroll.sample(n_points=n_points, rng=seed)
    ax = fig.add_subplot(1, 5, 5, projection="3d")
    artist = ax.scatter(x[:, 0], x[:, 1], y.ravel(), c=y_original, alpha=alpha, s=size)
    artist.set_rasterized(True)
    ax.set_title("Swiss roll mapping")

    fig.tight_layout()
    fig.savefig("transformed_samplers.pdf", dpi=350)


if __name__ == "__main__":
    main()
