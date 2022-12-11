"""Main figure for the paper, like a visual abstract."""
from typing import Optional

import jax
import matplotlib.pyplot as plt
import numpy as np

import bmi.api as bmi


def _assert_ravel_1d_array(a) -> np.ndarray:
    a = np.asarray(a)
    n = len(a)
    if a.shape not in {(n,), (n, 1)}:
        raise ValueError(f"Array of shape {a.shape} is not one dimensional.")

    return a.ravel()


def _min_max_grid_size(
    a: np.ndarray,
    a_min: Optional[float],
    a_max: Optional[float],
    a_grid_size: Optional[int],
) -> tuple[float, float, float]:
    if a_min is None:
        a_min = np.min(a)
    if a_max is None:
        a_max = np.max(a)
    if a_grid_size is None:
        a_grid_size = len(a)

    return a_min, a_max, a_grid_size


def plot_points(
    x: np.ndarray,
    y: np.ndarray,
    ax: plt.Axes,
    alpha: float = 0.1,
) -> None:
    """Plots the points on `ax`."""
    x = _assert_ravel_1d_array(x)
    y = _assert_ravel_1d_array(y)
    ax.scatter(x, y, alpha=alpha, s=1)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-8 * x))


def stairs(t: np.ndarray) -> np.ndarray:
    return sigmoid(t) + sigmoid(t - 1) + sigmoid(t + 2)


def identity(t: np.ndarray) -> np.ndarray:
    return t


# Conventions:
#  - f1(X), g1(Y) are the slightly distorted variables
#  - f2(X), g2(Y) are the heavily distorted variables
# Note that all of these should be continuous injections R -> R
# For example, strictly monotonic smooth functions are fine.


def f1(x: np.ndarray) -> np.ndarray:
    return x


def g1(y: np.ndarray) -> np.ndarray:
    return jax.scipy.special.erf(y)


def f2(x: np.ndarray) -> np.ndarray:
    return stairs(x)


def g2(y: np.ndarray) -> np.ndarray:
    return -stairs(y)


def get_estimates(sampler, estimator, f, g, n_points: int, n_seeds: int) -> tuple[float, float]:
    estimates = []
    for seed in range(n_seeds):
        x, y = sampler.sample(n_points=n_points, rng=seed)
        x_, y_ = f(x), g(y)
        estimates.append(estimator.estimate(x_, y_))

    mean = np.mean(estimates)
    # Note that this estimator of population standard deviation
    # is NOT unbiased, even though we use ddof=1. For more information
    # see the documentation of np.std or use Jensen's inequality.
    std = np.std(estimates, ddof=1)

    return mean, std


def main() -> None:
    n_seeds: int = 20
    n_points_plot = 10_000
    n_points_sample = 1_000

    correlation = 0.95

    covariance = np.asarray(
        [
            [1.0, correlation],
            [correlation, 1],
        ]
    )

    sampler = bmi.samplers.SplitMultinormal(dim_x=1, dim_y=1, covariance=covariance)
    estimator = bmi.estimators.KSGEnsembleFirstEstimator(neighborhoods=(10,))

    mi_true = sampler.mutual_information()

    def get_estimates_simple(f, g) -> tuple[float, float]:
        return get_estimates(
            sampler=sampler,
            estimator=estimator,
            f=f,
            g=g,
            n_points=n_points_sample,
            n_seeds=n_seeds,
        )

    # No diffeomorphisms
    original_mean, original_std = get_estimates_simple(identity, identity)

    # "Slight" diffeomorphisms applied to X and Y
    slight_mean, slight_std = get_estimates_simple(f1, g1)

    # More "dramatic" diffeomorphisms applied to X and Y
    dramatic_mean, dramatic_std = get_estimates_simple(f2, g2)

    fig, axs = plt.subplots(3, 1, figsize=(3, 10))

    axs[0].set_title(
        f"True: {mi_true:.2f} Estimate: {original_mean:.2f} $\\pm$ {original_std:.2f}"
    )
    axs[1].set_title(f"True: {mi_true:.2f} Estimate: {slight_mean:.2f} $\\pm$ {slight_std:.2f}")
    axs[2].set_title(
        f"True: {mi_true:.2f} Estimate: {dramatic_mean:.2f} $\\pm$ {dramatic_std:.2f}"
    )

    x, y = sampler.sample(n_points_plot, rng=42)

    plot_points(x, y, axs[0])
    plot_points(f1(x), g1(y), axs[1])
    plot_points(f2(x), g2(y), axs[2])

    # We turn off the ticks as the scale is not important for this plot
    for ax in axs:
        ax.tick_params(
            bottom=False,
            top=False,
            left=False,
            right=False,
            which="both",
            labelbottom=False,
            labelleft=False,
        )

    # Set axes labels
    axs[0].set_xlabel("$X$")
    axs[0].set_ylabel("$Y$")

    axs[1].set_xlabel("$f_1(X)$")
    axs[1].set_ylabel("$g_1(Y)$")

    axs[2].set_xlabel("$f_2(X)$")
    axs[2].set_ylabel("$g_2(Y)$")

    fig.tight_layout()
    fig.savefig("figure1.pdf")


if __name__ == "__main__":
    main()
