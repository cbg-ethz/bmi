from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from scipy.special import digamma, gamma

import bmi.samplers.splitmultinormal as spl
from bmi.samplers.base import BaseSampler


def _differential_entropy(k: int, dof: int) -> float:
    """Differential entropy of a :math:`Student-t(0, I_k, dof)`.

    See Eq. (7) of
      R.B. Arellano-Valle, J.E. Contreras-Reyes, M.G. Genton,
      Shannon Entropy and Mutual Information for Multivariate
      Skew-Elliptical Distributions,
      Scandinavian Journal of Statistics, vol. 40, pp. 46-47, 2013
    """
    half_sum = 0.5 * (dof + k)
    digamma_term = half_sum * (digamma(half_sum) - digamma(0.5 * dof))

    log_term = -np.log(gamma(half_sum)) + np.log(gamma(0.5 * dof)) + 0.5 * k * np.log(dof * np.pi)

    return log_term + digamma_term


class SplitStudentT(BaseSampler):
    """Multivariate Student-t distribution.

    Sampling is based on Wikipedia:
      https://en.wikipedia.org/wiki/Multivariate_t-distribution

    Mutual information is based on:
      R.B. Arellano-Valle, J.E. Contreras-Reyes, M.G. Genton,
      Shannon Entropy and Mutual Information for Multivariate
      Skew-Elliptical Distributions,
      Scandinavian Journal of Statistics, vol. 40, pp. 46-47, 2013

    Note:
        The formula for the mutual information is wrong, but can be
        calculated using the expressions involving differential entropies.
    """

    def __init__(
        self,
        *,
        dim_x: int,
        dim_y: int,
        df: int,
        dispersion: ArrayLike,
        mean: Optional[ArrayLike] = None,
    ) -> None:
        """

        Args:
            dim_x: dimension of the X variable
            dim_y: dimension of the Y variable
            df: degrees of freedom, strictly positive. Use `np.inf` for a Gaussian
            dispersion: dispersion matrix, shape (dim_x + dim_y, dim_x + dim_y)
            mean: mean of the distribution, shape (dim_x + dim_y,). Default: zero vector

        Note:
            Dispersion is *not* the covariance matrix. To calculate the covariance matrix,
            use the `covariance` method.
        """
        super().__init__(dim_x=dim_x, dim_y=dim_y)

        if mean is None:
            mean = np.zeros(dim_x + dim_y)
        self._mean = np.asarray(mean)

        # Mutual information of multivariate Student-t contains
        # the term corresponding to the multivariate normal distribution.
        # Note that this will also validate all the dimensions
        # and check whether the dispersion matrix is positive-definite
        self._multinormal = spl.SplitMultinormal(
            dim_x=dim_x, dim_y=dim_y, mean=mean, covariance=dispersion
        )

        if df <= 0:
            raise ValueError("Degrees of freedom must be positive.")
        self._degrees_of_freedom = df

        self._dispersion = np.asarray(dispersion)

    def sample(self, n_points: int, rng: int) -> tuple[np.ndarray, np.ndarray]:
        """Sampling from multivariate Student distribution.

        Note:
            This function is based on SciPy's sampling.
        """
        xy = stats.multivariate_t.rvs(
            loc=self._mean,
            shape=self._dispersion,
            df=self._degrees_of_freedom,
            size=n_points,
            random_state=rng,
        )
        assert xy.shape == (
            n_points,
            self.dim_total,
        ), f"Wrong shape: {xy.shape} != {(n_points, self.dim_total)}."

        return xy[:, : self.dim_x], xy[:, self.dim_x :]  # noqa: E203 colon spacing conventions

    @property
    def df(self) -> int:
        """Degrees of freedom."""
        return self._degrees_of_freedom

    def covariance(self) -> np.ndarray:
        """Calculates the covariance matrix.

        Returns:
            array, shape (dim_x+dim_y, dim_x+dim_y)

        Raises:
            ValueError, if covariance is not defined (for `df` <= 2)
        """
        if self.df <= 2:
            raise ValueError(
                f"For degrees of freedom {self.df} <= 2 the covariance is not defined."
            )
        else:
            return self.df * self._dispersion / (self.df - 2.0)

    def mi_normal(self) -> float:
        """Part of the mutual information corresponding to a multivariate
        normal with covariance given by the dispersion matrix.
        """
        return self._multinormal.mutual_information()

    def mi_correction(self) -> float:
        """Correction term in MI calculation.

        This term depends only on the dimensionality of each variable
        and the degrees of freedom.
        (It does not depend on the dispersion matrix or mean vector used).
        """
        return self.mi_correction_function(df=self.df, dim_x=self.dim_x, dim_y=self.dim_y)

    def mutual_information(self) -> float:
        """Expression for MI taken from Arellano-Valle et al., p. 47.

        This expression consists of two terms:
            `mi_normal`, which is the MI of a multivariate normal distribution
              with covariance given by the dispersion
            `mi_correction`, which is a correction term which does not depend
              on the means or the dispersion
        """

        return self.mi_normal() + self.mi_correction()

    @staticmethod
    def mi_correction_function(df: int, dim_x: int, dim_y: int) -> float:
        """Mutual information between variables with joint Multivariate Student-t
        decomposes as the sum of mutual information of Gaussian variables
        and the correction term.
        """
        h_x = _differential_entropy(k=dim_x, dof=df)
        h_y = _differential_entropy(k=dim_y, dof=df)
        h_xy = _differential_entropy(k=dim_x + dim_y, dof=df)
        return h_x + h_y - h_xy
