import numpy as np
from numpy.typing import ArrayLike
from scipy.special import digamma, gamma

import bmi.samplers.splitmultinormal as spl
from bmi.samplers.base import BaseSampler


class SplitStudentT(BaseSampler):
    """Multivariate Student-t distribution.

    Sampling is based on Wikipedia:
      https://en.wikipedia.org/wiki/Multivariate_t-distribution

    Mutual information is based on:
      R.B. Arellano-Valle, J.E. Contreras-Reyes, M.G. Genton,
      Shannon Entropy and Mutual Information for Multivariate
      Skew-Elliptical Distributions,
      Scandinavian Journal of Statistics, vol. 40, pp. 46-47, 2013
    """

    def __init__(
        self, *, dim_x: int, dim_y: int, mean: ArrayLike, dispersion: ArrayLike, nu: int
    ) -> None:
        """

        Args:
            dim_x: dimension of the X variable
            dim_y: dimension of the Y variable
            mean: mean of the distribution, shape (dim_x + dim_y,)
            dispersion: dispersion matrix, shape (dim_x + dim_y, dim_x + dim_y)
            nu: degrees of freedom
        """
        super().__init__(dim_x=dim_x, dim_y=dim_y)
        # Mutual information of multivariate Student-t contains
        # the term corresponding to the multivariate normal distribution.
        # Note that this will also validate all the dimensions
        # and check whether the dispersion matrix is positive-definite
        self._multinormal = spl.SplitMultinormal(
            dim_x=dim_x, dim_y=dim_y, mean=np.zeros_like(mean), covariance=dispersion
        )
        self._nu = nu

        self._mean = np.asarray(mean)
        self._dispersion = np.asarray(dispersion)

    def sample(self, n_points: int, rng: int) -> tuple[np.ndarray, np.ndarray]:
        """Sampling from multivariate Student-t as described in

          https://en.wikipedia.org/wiki/Multivariate_t-distribution

        Note:
            This function is based on NumPy's sampling, rather than JAX.
            This is motivated by the use of the chi-square distribution.
        """
        generator = np.random.default_rng(rng)

        # Shape (n_points, 1)
        u = generator.chisquare(df=self._nu, size=(n_points, 1))

        # Shape (n_points, dim_total)
        y = generator.multivariate_normal(
            mean=np.zeros(self.dim_total), cov=self._dispersion, size=n_points
        )

        # Shape (n_points, dim_total)
        x = np.sqrt(self._nu / u) * y + self._mean

        return x[:, : self.dim_x], x[:, self.dim_x :]  # noqa: E203 colon spacing conventions

    def mutual_information(self) -> float:
        """Expression for MI taken from Arellano-Valle et al., p. 47."""
        # Auxiliary variables, to make the expression look nice.
        # They should be read as "H"alf of the sum of "variables"
        h_nu = 0.5 * self._nu
        h_nu_x = 0.5 * (self._nu + self._dim_x)
        h_nu_y = 0.5 * (self._nu + self._dim_y)
        h_nu_xy = 0.5 * (self._nu + self._dim_x + self._dim_y)

        mi_normal = self._multinormal.mutual_information()
        log_term = np.log(gamma(h_nu) * gamma(h_nu_xy)) - np.log(gamma(h_nu_x) * gamma(h_nu_y))
        subtract_term = h_nu_x * digamma(h_nu_x) + h_nu_y * digamma(h_nu_y)
        add_term = h_nu_xy * digamma(h_nu_xy) + h_nu * digamma(h_nu)

        return mi_normal + log_term - subtract_term + add_term
