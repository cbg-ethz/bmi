"""This script visualises the 2D spirals, sampling some points
and applying spiral diffeomorphism with different speed parameter."""
import jax
import matplotlib.pyplot as plt
import numpy as np

import bmi.api as bmi


def get_points(
    random_seed: int, x_scale: float, y_scale: float, sample_uniform: bool, n_points: int
) -> np.ndarray:
    """Samples points to be visualised.

    Args:
        random_seed: random seed
        x_scale: characteristic scale of the X coordinate
        y_scale: characteristic scale of the Y coordinate
        sample_uniform: if True, the points will be sampled from a 2D uniform distribution.
            Otherwise, from a bivariate normal distribution
        n_points: number of points to be samples

    Returns:
        array of shape (n_points, 2)
    """
    rng = np.random.default_rng(random_seed)

    if sample_uniform:
        x = rng.uniform(-x_scale, x_scale, size=n_points)
        y = rng.uniform(-y_scale, y_scale, size=n_points)
        return np.dstack((x, y))[0]
    else:
        cov = np.eye(2)
        cov[0, 0] = x_scale
        cov[1, 1] = y_scale
        return rng.multivariate_normal(mean=np.zeros(2), cov=cov, size=n_points)


def get_transformed_points(points: np.ndarray, speed: float) -> np.ndarray:
    generator = bmi.transforms.so_generator(2)
    spiral = bmi.transforms.Spiral(generator=generator, speed=speed)
    return jax.vmap(spiral)(points)


def get_transformed_points_dictionary(
    points: np.ndarray, speed_list: list[float]
) -> dict[float, np.ndarray]:
    return {speed: get_transformed_points(points=points, speed=speed) for speed in speed_list}


def plot(
    points: np.ndarray,
    transformed_points: dict[float, np.ndarray],
    colors: list[str],
    alpha: float = 0.5,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim([-2.1, 2.1])
    ax.set_ylim([-2.1, 2.1])

    ax.scatter(points[:, 0], points[:, 1], label="Original", alpha=alpha, s=1, c=colors[0])

    colors_transformed = list(colors)[
        1 : 1 + len(transformed_points)  # noqa: E203 whitespace around :
    ]

    for color, (speed, tr_points) in zip(colors_transformed, transformed_points.items()):
        ax.scatter(
            tr_points[:, 0], tr_points[:, 1], label=f"Speed {speed}", alpha=alpha, s=1, c=color
        )

    ax.set_xlabel("Coordinate $x_1$")
    ax.set_ylabel("Coordinate $x_2$")

    # Hack to change the marker size in the legend
    lgnd = ax.legend()

    for handle in lgnd.legendHandles:
        handle._sizes = [20]

    return fig


def main() -> None:
    figure_location = "spiral.pdf"
    n_points: int = 2000
    x_scale: float = 2
    y_scale: float = 5e-2
    sample_uniform: bool = True
    random_seed: int = 42
    speed_list = [1, 10]
    colors = ["tab:red", "tab:purple", "tab:blue"]

    points = get_points(
        random_seed=random_seed,
        x_scale=x_scale,
        y_scale=y_scale,
        sample_uniform=sample_uniform,
        n_points=n_points,
    )
    transformed_points = get_transformed_points_dictionary(points=points, speed_list=speed_list)
    fig = plot(points=points, transformed_points=transformed_points, colors=colors)

    fig.tight_layout()
    fig.savefig(figure_location)


if __name__ == "__main__":
    main()
