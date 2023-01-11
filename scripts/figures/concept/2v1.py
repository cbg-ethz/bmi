import matplotlib.pyplot as plt
import numpy as np

import bmi.api as bmi


def get_estimates(
    sampler: bmi.ISampler,
    estimator: bmi.IMutualInformationPointEstimator,
    n_points: int,
    n_seeds: int,
) -> tuple[float, float]:
    estimates = []
    for seed in range(n_seeds):
        x, y = sampler.sample(n_points=n_points, rng=seed)
        estimates.append(estimator.estimate(x, y))

    mean = np.mean(estimates)
    # Note that this is the sample standard deviation, rather than population SD
    # In particular, it is biased (and even if we used ddof=1 it would be).
    # For more information see the documentation of np.std or use Jensen's inequality.
    std = np.std(estimates, ddof=0)
    return mean, std


def plot_sampler(
    ax: plt.Axes, sampler: bmi.ISampler, n_points_estimate: int, n_points_plot: int, n_seeds: int
) -> None:
    estimator = bmi.estimators.KSGEnsembleFirstEstimator(neighborhoods=(10,))
    mean, std = get_estimates(
        sampler=sampler, estimator=estimator, n_points=n_points_estimate, n_seeds=n_seeds
    )

    mi_true = sampler.mutual_information()

    ax.set_title(f"True: {mi_true:.2f}, Estimate: ${mean:.2f} \\pm {std:.2f}$")

    x, y = sampler.sample(n_points=n_points_plot, rng=42)
    ax.scatter(x[:, 0], x[:, 1], c=y.ravel(), s=1, alpha=0.3, cmap="jet")


def main() -> None:
    xx = 0.5
    xy = 0.805

    # xx = 0.7
    # xy = 0.905

    n_seeds = 10
    n_points_estimate = 5_000
    n_points_plot = 5_000
    lims = 5
    speeds = np.asarray([0.5, 1])

    cov = 0.8 * np.asarray(
        [
            [1.0, xx, xy],
            [xx, 1.0, xy],
            [xy, xy, 1.0],
        ]
    )

    gaussian_sampler = bmi.samplers.SplitMultinormal(dim_x=2, dim_y=1, covariance=cov)

    generator = bmi.transforms.so_generator(2)
    spiral_slow = bmi.transforms.Spiral(generator=generator, speed=speeds[0])
    spiral_fast = bmi.transforms.Spiral(generator=generator, speed=speeds[1])

    slow_sampler = bmi.samplers.TransformedSampler(
        base_sampler=gaussian_sampler, transform_x=spiral_slow, vectorise=True
    )
    fast_sampler = bmi.samplers.TransformedSampler(
        base_sampler=gaussian_sampler, transform_x=spiral_fast, vectorise=True
    )

    fig, axs = plt.subplots(3, figsize=(4, 13))

    samplers = [gaussian_sampler, slow_sampler, fast_sampler]

    for ax, sampler in zip(axs.ravel(), samplers):
        plot_sampler(
            ax=ax,
            sampler=sampler,
            n_seeds=n_seeds,
            n_points_plot=n_points_plot,
            n_points_estimate=n_points_estimate,
        )

    # fig.suptitle("Mutual Information")

    axs[0].set_ylabel("$X$")
    axs[1].set_ylabel("$f_1(X)$")
    axs[2].set_ylabel("$f_2(X)$")

    for ax in axs:
        ax.set_xlim(-lims, lims)
        ax.set_ylim(-lims, lims)

    fig.tight_layout()
    fig.savefig("2v1-figure.pdf")


if __name__ == "__main__":
    main()
