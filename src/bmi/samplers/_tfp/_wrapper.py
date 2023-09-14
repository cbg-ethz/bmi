"""A wrapper from TFP distributions to BMI samplers."""
from typing import Optional, Union

import jax

from bmi.samplers._tfp._core import JointDistribution, monte_carlo_mi_estimate
from bmi.samplers.base import BaseSampler, KeyArray, cast_to_rng


class FineSampler(BaseSampler):
    """Wrapper around a fine distribution."""

    def __init__(
        self,
        dist: JointDistribution,
        mi: Optional[float] = None,
        mi_estimate_seed: Union[KeyArray, int] = 0,
        mi_estimate_sample: int = 200_000,
    ) -> None:
        """

        Args:
            dist: fine distribution to be wrapped
            mi: mutual information of the fine distribution, if already calculated.
                If not provided, it will be estimated via Monte Carlo sampling.
            mi_estimate_seed: seed for the Monte Carlo sampling
            mi_estimate_sample: number of samples for the Monte Carlo sampling
        """
        super().__init__(dim_x=dist.dim_x, dim_y=dist.dim_y)
        self._dist = dist

        if mi is None:
            rng = cast_to_rng(mi_estimate_seed)
            self._mi, self._mi_stderr = monte_carlo_mi_estimate(
                key=rng, dist=self._dist, n=mi_estimate_sample
            )
        else:
            self._mi = mi
            self._mi_stderr = None

    def sample(
        self, n_points: int, rng: Union[int, KeyArray]
    ) -> tuple[jax.numpy.ndarray, jax.numpy.ndarray]:
        key = cast_to_rng(rng)
        return self._dist.sample(n_points=n_points, key=key)

    def mutual_information(self) -> float:
        return self._mi
