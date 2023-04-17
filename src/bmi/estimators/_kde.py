"""Estimation of mutual information via kernel density estimation
of the differential entropy."""
from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from sklearn.neighbors import KernelDensity

from bmi.interface import BaseModel, IMutualInformationPointEstimator
from bmi.utils import ProductSpace


def _differential_entropy(estimator: KernelDensity, samples: np.ndarray) -> float:
    """Estimates the differential entropy of a distribution by fitting
    a kernel density estimator and estimating the (negative) mean log-PDF
    from samples."""
    log_probs = estimator.score_samples(samples)
    return -np.mean(log_probs)


_AllowedBandwith = Union[float, Literal["scott", "silverman"]]
_AllowedKernel = Literal["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"]


class KDEParams(BaseModel):
    kernel_xy: _AllowedKernel
    kernel_x: _AllowedKernel
    kernel_y: _AllowedKernel
    bandwidth_xy: _AllowedBandwith
    bandwidth_x: _AllowedBandwith
    bandwidth_y: _AllowedBandwith
    standardize: bool


class DifferentialEntropies(BaseModel):
    entropy_x: float
    entropy_y: float
    entropy_xy: float
    mutual_information: float


class KDEMutualInformationEstimator(IMutualInformationPointEstimator):
    """The kernel density mutual information estimator based on

    .. math::

       I(X; Y) = h(X) + h(Y) - h(X, Y),

    where :math:`h(X)` is the differential entropy

    .. math::

       h(X) = -\\mathbb{E}[ \\log p(X) ].

    The logarithm of probability density function :math:`\\log p(X)`
    is estimated via a kernel density estimator (KDE) using SciKit-Learn.

    Note:
        This estimator is very sensitive to the choice of the bandwidth
        and the kernel. We suggest to treat it with caution.
    """

    def __init__(
        self,
        kernel_xy: _AllowedKernel = "tophat",
        kernel_x: Optional[_AllowedKernel] = None,
        kernel_y: Optional[_AllowedKernel] = None,
        bandwidth_xy: _AllowedBandwith = "scott",
        bandwidth_x: Optional[_AllowedBandwith] = None,
        bandwidth_y: Optional[_AllowedBandwith] = None,
        standardize: bool = True,
    ) -> None:
        """

        Args:
            kernel_xy: kernel to be used for joint distribution
              PDF :math:`p_{XY}` estimation.
              See SciKit-Learn's ``KernelDensity`` object for more information.
            kernel_x: kernel to be used for the :math:`p_X` estimation.
              If ``None`` (default), ``kernel_xy`` will be used.
            kernel_y: similarly to ``kernel_x``.
            bandwidth_xy: kernel bandwidth to be used for joint distribution
              estimation.
            bandwidth_x: kernel bandwidth to be used
              for :math:`p_X` estimation.
              If set to None (default), then ``bandwidth_xy`` is used.
            bandwidth_y: similar to ``bandwidth_x``
            standardize: whether to standardize the data points
        """
        bandwidth_x = bandwidth_xy if bandwidth_x is None else bandwidth_x
        bandwidth_y = bandwidth_xy if bandwidth_y is None else bandwidth_y

        kernel_x = kernel_x or kernel_xy
        kernel_y = kernel_y or kernel_xy

        self._kde_x = KernelDensity(bandwidth=bandwidth_x, kernel=kernel_x)
        self._kde_y = KernelDensity(bandwidth=bandwidth_y, kernel=kernel_y)
        self._kde_xy = KernelDensity(bandwidth=bandwidth_xy, kernel=kernel_xy)

        self._params = KDEParams(
            bandwidth_xy=bandwidth_xy,
            bandwidth_x=bandwidth_x,
            bandwidth_y=bandwidth_y,
            kernel_xy=kernel_xy,
            kernel_x=kernel_x,
            kernel_y=kernel_y,
            standardize=standardize,
        )

        self._standardize = standardize

    def _fit(self, space: ProductSpace) -> None:
        self._kde_x.fit(space.x)
        self._kde_y.fit(space.y)
        self._kde_xy.fit(space.xy)

    def estimate(self, x: ArrayLike, y: ArrayLike) -> float:
        return self.estimate_entropies(x, y).mutual_information

    def parameters(self) -> BaseModel:
        return self._params

    def estimate_entropies(self, x: ArrayLike, y: ArrayLike) -> DifferentialEntropies:
        """Calculates differential entropies.

        Note:
            Differential entropy is *not* invariant to standardization.
            In particular, if you want to estimate differential entropy
            of the original variables, you should use ``standardize=False``.
        """
        space = ProductSpace(x=x, y=y, standardize=self._standardize)
        self._fit(space)

        h_x = _differential_entropy(estimator=self._kde_x, samples=space.x)
        h_y = _differential_entropy(estimator=self._kde_y, samples=space.y)
        h_xy = _differential_entropy(estimator=self._kde_xy, samples=space.xy)

        mutual_information = h_x + h_y - h_xy

        return DifferentialEntropies(
            entropy_x=h_x,
            entropy_y=h_y,
            entropy_xy=h_xy,
            mutual_information=mutual_information,
        )
