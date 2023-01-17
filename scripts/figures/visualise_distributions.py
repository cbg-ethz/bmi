"""Visualisation of several samplers used."""
import matplotlib.pyplot as plt

import bmi.api as bmi
from bmi.benchmark.tasks.one_dimensional import get_half_cube_sampler


def main() -> None:
    gaussian_correlation: float = 0.85
    n_points: int = 5_000
    seed: int = 0
    alpha: float = 0.5
    size: int = 3

    sampler_gaussian = bmi.samplers.BivariateNormalSampler(correlation=gaussian_correlation)
    sampler_uniform = bmi.samplers.BivariateUniformMarginsSampler(
        gaussian_correlation=gaussian_correlation
    )
    sampler_swissroll = bmi.samplers.SwissRollSampler(sampler=sampler_uniform)
    sampler_halfnormal = get_half_cube_sampler(gaussian_correlation=gaussian_correlation)

    fig = plt.figure(figsize=plt.figaspect(0.25))

    ax = fig.add_subplot(1, 4, 1)
    x, y = sampler_gaussian.sample(n_points=n_points, rng=seed)
    y_original = y.ravel()
    ax.scatter(x.ravel(), y.ravel(), c=y_original, alpha=alpha, s=size)
    ax.set_title("Bivariate Gaussian variables")
    ax.set_xlabel("$X$")
    ax.set_ylabel("$Y$")

    ax = fig.add_subplot(1, 4, 2)
    x, y = sampler_uniform.sample(n_points=n_points, rng=seed)
    ax.scatter(x.ravel(), y.ravel(), c=y_original, alpha=alpha, s=size)
    ax.set_title("Correlated uniform variables")
    ax.set_xlabel("CDF$(X)$")
    ax.set_ylabel("CDF$(Y)$")

    ax = fig.add_subplot(1, 4, 3)
    x, y = sampler_halfnormal.sample(n_points=n_points, rng=seed)
    ax.scatter(x.ravel(), y.ravel(), c=y_original, alpha=alpha, s=size)
    ax.set_title("Bivariate Gaussian transformed\nwith half-cube mapping")
    ax.set_xlabel("$h(X)$")
    ax.set_ylabel("$h(Y)$")

    x, y = sampler_swissroll.sample(n_points=n_points, rng=seed)
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    ax.scatter(x[:, 0], x[:, 1], y.ravel(), c=y_original, alpha=alpha, s=size)
    ax.set_title("Correlated uniform variabled\ntransformed with Swiss roll mapping")

    fig.tight_layout()
    fig.savefig("transformed_samplers.pdf")


if __name__ == "__main__":
    main()
