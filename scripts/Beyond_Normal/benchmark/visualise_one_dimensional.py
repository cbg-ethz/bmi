from pathlib import Path
from typing import Optional

import jax
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import mutual_info_regression

import bmi.api as bmi
import bmi.benchmark.tasks.one_dimensional as od


def show_sampler(
    sampler: bmi.ISampler,
    n_samples: int = 5000,
    seed: int = 0,
    axs: Optional[tuple[plt.Axes, plt.Axes, plt.Axes]] = None,
    n_bins: Optional[int] = None,
    **plot_kwargs,
) -> Optional[plt.Figure]:
    if axs is None:
        fig, axs = plt.subplots(ncols=3, figsize=(12, 4))
    else:
        fig = None

    xs, ys = sampler.sample(n_samples, seed)
    assert xs.shape == (n_samples, 1)
    assert ys.shape == (n_samples, 1)

    if n_bins is None and n_samples > 1000:
        n_bins = n_samples // 200

    n_bins = n_bins or 20

    ax = axs[0]
    ax.scatter(xs, ys, **(dict(s=5, color="black", alpha=0.5) | plot_kwargs))
    ax.set_title("$P_{XY}$")
    ax.set_xlabel("$X$")
    ax.set_ylabel("$Y$")

    ax = axs[1]
    ax.hist(xs[..., 0], color="black", bins=n_bins, density=True)
    ax.set_title("$P_X$")

    ax = axs[2]
    ax.hist(ys[..., 0], color="black", bins=n_bins, density=True)
    ax.set_title("$P_Y$")

    return fig


def ksg_mi(
    sampler: bmi.ISampler, n_samples: int = 2000, seed: int = 42, verbose: bool = False
) -> float:
    x, y = sampler.sample(n_samples, rng=seed)
    estimate = mutual_info_regression(x, y.ravel())[0]
    if verbose:
        print(f"  True: {sampler.mutual_information():.2f}. Estimated: {estimate:.2f}")
    return estimate


def visualise_additive_uniform(figures_dir: Path, epsilon: float) -> None:
    sampler = bmi.samplers.AdditiveUniformSampler(epsilon=epsilon)
    path = figures_dir / f"additive-uniform-{epsilon:.2f}.pdf"
    fig = show_sampler(
        sampler=sampler,
    )
    assert fig is not None
    print(f"Additive uniform, epsilon = {epsilon}")
    ksg_mi(sampler=sampler, verbose=True)
    fig.tight_layout()
    fig.savefig(path)


def visualise_bivariate_student_t(figures_dir: Path) -> None:
    sampler = od.get_student_sampler()
    path = figures_dir / f"student-{sampler.df}.pdf"
    fig = show_sampler(sampler=sampler)
    assert fig is not None
    print(f"Student-t with dof = {sampler.df}")
    ksg_mi(sampler, verbose=True)
    fig.tight_layout()
    fig.savefig(path)


def visualise_marginal_uniform(figures_dir: Path) -> None:
    sampler = od.get_marginal_uniform_sampler()
    path = figures_dir / "marginal-uniform.pdf"
    fig = show_sampler(sampler)
    assert fig is not None
    print("Marginal uniform")
    ksg_mi(sampler, verbose=True)
    fig.tight_layout()
    fig.savefig(path)


def visualise_half_cube(figures_dir: Path) -> None:
    fig, axs = plt.subplots(ncols=4, figsize=(16, 4))
    path = figures_dir / "half-cube.pdf"

    sampler = od.get_half_cube_sampler()

    print("Half-cube")
    ksg_mi(sampler, verbose=True)

    show_sampler(sampler=sampler, axs=axs[:3])
    ax = axs[3]

    xs = np.linspace(-2, 2, 101)
    ys = jax.vmap(od.half_cube)(xs)

    ax.plot(xs, ys)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")

    fig.tight_layout()
    fig.savefig(path)


def visualise_bimodal(figures_dir: Path) -> None:
    sampler = od.get_bimodal_sampler()
    path = figures_dir / "bimodal.pdf"
    fig = show_sampler(sampler)
    assert fig is not None
    print("Bimodal")
    ksg_mi(sampler, verbose=True)
    fig.tight_layout()
    fig.savefig(path)


def visualise_wiggly(figures_dir: Path) -> None:
    fig, axs = plt.subplots(ncols=5, figsize=(20, 4))
    path = figures_dir / "wiggly.pdf"

    sampler = od.get_wiggly_sampler()

    print("Wiggly")
    ksg_mi(sampler, verbose=True)

    show_sampler(sampler=sampler, axs=axs[:3])

    # Sample some values X and Y
    xs, ys = sampler.sample(n_points=100, rng=42)

    # Note: now xs and ys are not paired anymore
    # They do *not* constitute a sample from P_XY
    xs = np.asarray(sorted(xs.ravel()))
    ys = np.asarray(sorted(ys.ravel()))

    # Visualise functions
    ax = axs[3]
    axs[3].plot(xs, jax.vmap(od.default_wiggly_x)(xs), label="Wiggly x")
    axs[3].plot(ys, jax.vmap(od.default_wiggly_y)(ys), label="Wiggly y")

    ax.legend()
    ax.set_xlabel("$t$")
    ax.set_ylabel("$f(t)$")

    # Visualise derivatives
    ax = axs[4]
    axs[4].plot(xs, jax.vmap(jax.grad(od.default_wiggly_x))(xs), label="Wiggly x")
    axs[4].plot(ys, jax.vmap(jax.grad(od.default_wiggly_y))(ys), label="Wiggly y")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$df/dt$")

    fig.tight_layout()
    fig.savefig(path)


def main() -> None:
    figures_dir = Path("one-dimensional-figures")
    figures_dir.mkdir(exist_ok=True)

    # Additive uniform
    for epsilon in od.ADDITIVE_UNIFORM_EPSILONS:
        visualise_additive_uniform(figures_dir=figures_dir, epsilon=epsilon)

    # Student-t
    visualise_bivariate_student_t(figures_dir=figures_dir)

    # Uniform margins
    visualise_marginal_uniform(figures_dir=figures_dir)

    # Half-cube
    visualise_half_cube(figures_dir=figures_dir)

    # Bimodal
    visualise_bimodal(figures_dir=figures_dir)

    # Wiggly
    visualise_wiggly(figures_dir=figures_dir)


if __name__ == "__main__":
    main()
