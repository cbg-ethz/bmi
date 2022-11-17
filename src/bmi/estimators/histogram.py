"""Histogram-based approach.

Note: if a bin p(x, y) has zero counts, we assign zero contribution from it to the MI:
  MI \\approx \\sum p(x, y) \\log( p(x, y) / p(x)p(y) )
"""
from itertools import product
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from bmi.interface import IMutualInformationPointEstimator
from bmi.utils import ProductSpace


class HistogramEstimator(IMutualInformationPointEstimator):
    def __init__(
        self, n_bins_x: int = 5, n_bins_y: Optional[int] = None, standardize: bool = True
    ) -> None:
        """
        Args:
            n_bins_x: number of bins per each X dimension
            n_bins_y: number of bins per each Y dimension. Leave to None to use `n_bins_x`
            standardize: whether to standardize the data set
        """
        self._n_bins_x = n_bins_x
        self._n_bins_y = n_bins_y or n_bins_x
        self._standardize = standardize

    def estimate(self, x: ArrayLike, y: ArrayLike) -> float:
        """MI estimate."""
        space = ProductSpace(x=x, y=y, standardize=self._standardize)
        bins = [self._n_bins_x for _ in range(space.dim_x)] + [
            self._n_bins_y for _ in range(space.dim_y)
        ]

        histogram, _ = np.histogramdd(space.xy, bins=bins, density=False)
        range_x = self._n_bins_x**space.dim_x
        range_y = self._n_bins_y**space.dim_y

        flat_histogram = np.zeros((range_x, range_y), dtype=float)
        for i, x in enumerate(product(range(self._n_bins_x), repeat=space.dim_x)):
            for j, y in enumerate(product(range(self._n_bins_y), repeat=space.dim_y)):
                index = tuple(x) + tuple(y)
                flat_histogram[i, j] = histogram[tuple(index)]

        del i, j, space, histogram

        # Convert from counts to (empirical) densities
        p_xy = flat_histogram / np.sum(flat_histogram)
        # Calculate marginals by integrating out the other variable
        p_x = p_xy.sum(axis=1)
        p_y = p_xy.sum(axis=0)

        assert p_x.shape == (range_x,)
        assert p_y.shape == (range_y,)

        # Calculate MI
        mi = 0.0
        for i in range(range_x):
            for j in range(range_y):
                if p_xy[i, j] > 0:
                    mi += p_xy[i, j] * (np.log(p_xy[i, j]) - np.log(p_x[i]) - np.log(p_y[j]))

        return mi
