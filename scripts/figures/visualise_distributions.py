"""Visualisation of several samplers used."""
import matplotlib.pyplot as plt

import bmi.benchmark.task_list as tl


def main() -> None:
    n_points: int = 5_000
    seed: int = 0
    alpha: float = 0.5
    size: int = 3

    task_gaussian = tl.BINORMAL_BASE
    task_uniform = tl.UNIFORM_BASE

    assert len(tl.EMBEDDINGS_TASKS) == 1, "We expect only one embedding task (Swiss roll)."
    task_swissroll = tl.EMBEDDINGS_TASKS[0]

    task_halfcube = tl.half_cube(task_gaussian)
    task_bimodal = tl.bimodal_gaussians.task_bimodal_gaussians(
        gaussian_correlation=tl.GAUSSIAN_CORRELATION
    )

    plt.set_cmap("turbo")

    fig = plt.figure(figsize=plt.figaspect(0.19))

    ax = fig.add_subplot(1, 5, 1)
    x, y = task_gaussian.sample(n_points, seed)
    y_original = y.ravel()
    ax.scatter(x.ravel(), y.ravel(), c=y_original, alpha=alpha, s=size, rasterized=True)
    ax.set_title("Bivariate Gaussian")
    ax.set_xlabel("$X$")
    ax.set_ylabel("$Y$")
    ax.spines[["right", "top"]].set_visible(False)

    ax = fig.add_subplot(1, 5, 2)
    x, y = task_uniform.sample(n_points, seed)
    ax.scatter(x.ravel(), y.ravel(), c=y_original, alpha=alpha, s=size, rasterized=True)
    ax.set_title("Correlated uniform")
    ax.set_xlabel("CDF$(X)$")
    ax.set_ylabel("CDF$(Y)$")
    ax.spines[["right", "top"]].set_visible(False)

    ax = fig.add_subplot(1, 5, 3)
    x, y = task_halfcube.sample(n_points, seed)
    ax.scatter(x.ravel(), y.ravel(), c=y_original, alpha=alpha, s=size, rasterized=True)
    ax.set_title("Half-cube mapping")
    ax.set_xlabel("$h(X)$")
    ax.set_ylabel("$h(Y)$")
    ax.spines[["right", "top"]].set_visible(False)

    ax = fig.add_subplot(1, 5, 4)
    x, y = task_bimodal.sample(n_points, seed)
    ax.scatter(x.ravel(), y.ravel(), c=y_original, alpha=alpha, s=size, rasterized=True)
    ax.set_title("Making margins bimodal")
    ax.set_xlabel("bimodal$(X)$")
    ax.set_ylabel("bimodal$(Y)$")
    ax.spines[["right", "top"]].set_visible(False)

    x, y = task_swissroll.sample(n_points, seed)
    ax = fig.add_subplot(1, 5, 5, projection="3d")
    artist = ax.scatter(x[:, 0], x[:, 1], y.ravel(), c=y_original, alpha=alpha, s=size)
    artist.set_rasterized(True)
    ax.grid(False)
    # Turn off the panes (the gray background)
    ax.xaxis.pane.set_edgecolor("k")
    ax.yaxis.pane.set_edgecolor("k")
    ax.zaxis.pane.set_edgecolor("k")
    ax.xaxis._axinfo["juggled"] = (1, 2, 0)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_title("Swiss roll mapping")

    fig.tight_layout()
    fig.savefig("transformed_samplers.pdf", dpi=350)


if __name__ == "__main__":
    main()
