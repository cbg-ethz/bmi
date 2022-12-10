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
    """

    Code based on the solution presented in:
      https://stackoverflow.com/a/36958298/7705397
    """
    x = _assert_ravel_1d_array(x)
    y = _assert_ravel_1d_array(y)
    ax.scatter(x, y, alpha=alpha, s=1)


def sigmoid(x):
    return 1 / (1 + np.exp(-10 * x))


def f(t):
    return sigmoid(t) + sigmoid(t - 1) + sigmoid(t + 2)


def g(u):
    return u + 0.05 * (u - 2) ** 3 + 0.0003 * (u + 3) ** 5


def h(u):
    return jax.scipy.stats.logistic.cdf(u) + u**3 / 2


def main() -> None:
    correlation = 0.95
    n_points = 1_000

    covariance = np.asarray(
        [
            [1.0, correlation],
            [correlation, 1],
        ]
    )

    sampler = bmi.samplers.SplitMultinormal(dim_x=1, dim_y=1, covariance=covariance)

    x, y = sampler.sample(n_points=n_points, rng=42)

    estimator = bmi.estimators.KSGEnsembleFirstEstimator(neighborhoods=(10,))

    # "Slight" diffeomorphisms applied to X and Y
    x_slight = x + 0.1 * (x - 0.3) ** 3
    y_slight = jax.scipy.special.erf(y)

    # Drastic diffeomorphisms applied to X and Y
    # TODO(Pawel): This diffeomorphism doesn't look cool and the differene in MI
    #   is not that big. Update.
    x_very = x
    y_very = f(y)

    mi_true = sampler.mutual_information()
    estimate_original = estimator.estimate(x, y)
    estimate_slight = estimator.estimate(x_slight, y_slight)
    estimate_very = estimator.estimate(x_very, y_very)

    fig, axs = plt.subplots(3, 1, figsize=(3, 10))

    axs[0].set_title(f"True: {mi_true:.2f} Estimate: {estimate_original:.2f}")
    axs[1].set_title(f"True: {mi_true:.2f} Estimate: {estimate_slight:.2f}")
    axs[2].set_title(f"True: {mi_true:.2f} Estimate: {estimate_very:.2f}")

    plot_points(x, y, axs[0])
    plot_points(x_slight, y_slight, axs[1])
    plot_points(x_very, y_very, axs[2])

    fig.tight_layout()
    fig.savefig("figure1.pdf")


if __name__ == "__main__":
    main()
