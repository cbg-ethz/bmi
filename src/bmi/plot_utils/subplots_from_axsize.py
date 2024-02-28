"""Allows creating """

from collections.abc import Iterable
from typing import Union

try:
    import matplotlib.pyplot as plt  # pytype: disable=import-error
    import mpl_toolkits.axes_grid1 as ag  # pytype: disable=import-error
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Matplotlib is an optional dependency, "
        "but it needs to be installed to run the plots."
        "Run 'pip install matplotlib'."
    )

import numpy as np


def _normalize_axsizes_and_axcounts(axs, axc):
    # if ax sizes are iterable, compute new ax count
    if isinstance(axs, Iterable):
        axs = list(axs)
        if axc is None:
            axc = len(axs)
        else:
            assert axc == len(axs), "axes count doesn't agree with number of sizes"

    # if not, make sure they are a list with correct count
    else:
        if axc is None:
            axc = 1
        axs = [axs for _ in range(axc)]

    return axs, axc


def _make_sizes(start, axs, end, spaces):
    n = len(axs)
    assert len(spaces) == n - 1

    lengths = [start] + [e for p in zip(axs, spaces) for e in p] + [axs[-1], end]
    sizes = [ag.Size.Fixed(length) for length in lengths]

    return sizes, sum(lengths)


def subplots_from_axsize(
    axsize=(3, 2),
    nrows=None,
    ncols=None,
    top=0.1,
    bottom=0.5,
    left=0.5,
    right=0.1,
    hspace=0.5,
    wspace=0.5,
    squeeze=True,
) -> tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
    """
    Similar to plt.subplots() but uses fixed instead of relative sizes.
    This allows for more control over the final axes sizes.

    Examples:
    subplots_from_axsize(axsize=(3, 2), nrows=2) creates a figure with two axes of size (3, 2)
    subplots_from_axsize(axsize=(3, [2, 1])) creates a figure with two axes: (3, 2) and (3, 1)
    """
    axx, axy = axsize

    axx, ncols = _normalize_axsizes_and_axcounts(axx, ncols)
    axy, nrows = _normalize_axsizes_and_axcounts(axy, nrows)

    hs, _ = _normalize_axsizes_and_axcounts(hspace, nrows - 1)
    ws, _ = _normalize_axsizes_and_axcounts(wspace, ncols - 1)

    w_sizes, total_w = _make_sizes(left, axx, right, ws)
    h_sizes, total_h = _make_sizes(top, axy, bottom, hs)

    fig = plt.figure(figsize=(total_w, total_h))

    divider = ag.Divider(fig, (0, 0, 1, 1), w_sizes, h_sizes[::-1], aspect=False)
    axs = np.array(
        [
            [
                fig.add_axes(
                    divider.get_position(),
                    axes_locator=divider.new_locator(nx=2 * col + 1, ny=2 * row + 1),
                )
                for col in range(ncols)
            ]
            for row in range(nrows - 1, -1, -1)
        ]
    )

    if squeeze:
        axs = np.squeeze(axs)

    if len(axs.ravel()) == 1:
        return fig, axs.ravel()[0]
    else:
        return fig, axs
