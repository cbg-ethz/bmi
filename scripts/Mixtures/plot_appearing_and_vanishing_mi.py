"""Appearing and vanishing mutual information illustration."""
import matplotlib.pyplot as plt
import numpy as np


def plot_density(ax: plt.Axes, array: np.ndarray, title: str) -> None:
    """Plots density in `array` on `ax`."""
    ax.imshow(array, origin="lower", extent=[0, 2, 0, 2], cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(title)


def main() -> None:
    x = np.linspace(0, 2)
    y = np.linspace(0, 2)

    X, Y = np.meshgrid(x, y)

    fig, axs = plt.subplots(2, 3, dpi=300)

    # First row: appearing MI
    # Component 1 (bottom left)
    ax = axs[0, 0]
    mask1 = (0 < X) & (X < 1) & (0 < Y) & (Y < 1)
    plot_density(ax, mask1, "$I=0$")

    # Component 2 (top right)
    ax = axs[0, 1]
    mask2 = (1 < X) & (X < 2) & (1 < Y) & (Y < 2)
    plot_density(ax, mask2, "$I=0$")

    # Mixture
    ax = axs[0, 2]
    mask3 = mask1 | mask2
    plot_density(ax, 0.5 * mask3, "$I=\\log 2$")

    # Second row
    # Component 1: mixture from first row
    ax = axs[1, 0]
    plot_density(ax, 0.5 * mask3, "$I=\\log 2$")

    # Component 2: symmetric mixture
    ax = axs[1, 1]
    mask4 = (0 < X) & (X < 1) & (1 < Y) & (Y < 2) | (1 < X) & (X < 2) & (0 < Y) & (Y < 1)
    plot_density(ax, 0.5 * mask4, "$I=\\log 2$")

    # Mixture: independent
    ax = axs[1, 2]
    plot_density(ax, 0.25 * (mask3 | mask4), "$I=0$")

    fig.tight_layout()
    fig.savefig("appearing_and_vanishing_mi.pdf")


if __name__ == "__main__":
    main()
