"""Submodule with generic utilities.

Note:
    The content of this submodule is subject to change.
    When we refactor the code and observe patterns,
    the methods implemented here will move.
"""
from typing import Generator, Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn import preprocessing


class ProductSpace:
    """Represents points in `X x Y` space.

    Attrs:
        n_points: number of stored points
        x: points projected on X space
        y: points projected on Y space
        xy: points in `X x Y` space. This is a time-expensive operation (calculated on the fly)
          so if one only needs to access individual points/batches, we suggest to use `space[i]`
          syntax

    Example:
        >>> x = [[0], [1]]
        >>> y = [[2, 3], [4, 5]]
        >>> space = ProductSpace(x, y, standardize=False)
        >>> space.n_points
        2
        >>> space.x
        array([[0.], [1.]])
        >>> space.y
        array([[2., 3.], [4., 5.]])
        >>> space.xy
        array([[0., 2., 3.], [1., 4. 5.]])
        >>> space[0]  # Other slicing options are also allowed
        array([0., 2., 3.])
    """

    def __init__(self, x: ArrayLike, y: ArrayLike, *, standardize: bool = True) -> None:
        x, y = np.array(x), np.array(y)

        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Arrays have different length: {len(x)} != {len(y)}.")
        if standardize:
            x: np.ndarray = preprocessing.StandardScaler(copy=False).fit_transform(x)
            y: np.ndarray = preprocessing.StandardScaler(copy=False).fit_transform(y)

        self._x = x
        self._y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    @property
    def dim_x(self) -> int:
        return self._x.shape[1]

    @property
    def dim_y(self) -> int:
        return self._y.shape[1]

    @property
    def x(self) -> np.ndarray:
        return self._x

    @property
    def y(self) -> np.ndarray:
        return self._y

    @property
    def xy(self) -> np.ndarray:
        return np.hstack([self.x, self.y])

    def __getitem__(self, item: Union[int, slice]) -> np.ndarray:
        """Returns the `index`th point in the product space `X x Y`. Works also with
        the slice notation `start:end`."""
        return np.hstack([self._x[item], self._y[item]])


def chunker(n_items: int, chunk_size: int) -> Generator[slice, None, None]:
    """Used to process data in batches.

    Based on https://stackoverflow.com/a/434328

    Args:
        n_items: number of items in the list or array
        chunk_size: maximal chunk size

    Returns:
        a generator object with slices object
        (can be used to slice a list or array)

    Example:
        >>> X = list("abcd")
        >>> for batch_index in chunker(n_items=4, chunk_size=3):
                print(X[batch_index])
        ["a", "b", "c"]
        ["d"]

    Note:
        The last chunk may be smaller than `chunk_size`
    """
    if n_items <= 0 or chunk_size <= 0:
        raise ValueError("Size must be non-negative")

    for start in range(0, n_items, chunk_size):
        end = min(n_items, start + chunk_size)
        yield slice(start, end)
