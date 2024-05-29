"""A wrapper from TFP distributions to BMI samplers."""

from typing import Optional, Union

import jax

from bmi.samplers._tfp._core import JointDistribution, monte_carlo_mi_estimate
from bmi.samplers.base import BaseSampler, KeyArray, cast_to_rng


class FineSampler(BaseSampler):
    """Wraps a given Bend and Mix Model (BMM) into a sampler."""

    def __init__(
        self,
        dist: JointDistribution,
        mi: Optional[float] = None,
        mi_estimate_seed: Union[KeyArray, int] = 0,
        mi_estimate_sample: int = 200_000,
    ) -> None:
        """

        Args:
            dist: distribution represented by a BMM to be wrapped
            mi: mutual information of the distribution, if already calculated.
                If not provided, it will be estimated via Monte Carlo sampling.
            mi_estimate_seed: seed for the Monte Carlo sampling
            mi_estimate_sample: number of samples for the Monte Carlo sampling
        """
        super().__init__(dim_x=dist.dim_x, dim_y=dist.dim_y)
        self._dist = dist

        self._mi = mi
        self._mi_stderr = None

        self._mi_estimate_seed = mi_estimate_seed
        self._mi_estimate_sample = mi_estimate_sample
        if self._mi_estimate_sample < 1:
            raise ValueError(f"Provided too small sample size: {mi_estimate_sample}.")

    def sample(
        self, n_points: int, rng: Union[int, KeyArray]
    ) -> tuple[jax.numpy.ndarray, jax.numpy.ndarray]:
        key = cast_to_rng(rng)
        return self._dist.sample(n_points=n_points, key=key)

    def mutual_information(self) -> float:
        if self._mi is not None:
            return self._mi
        else:
            rng = cast_to_rng(self._mi_estimate_seed)
            self._mi, self._mi_stderr = monte_carlo_mi_estimate(
                key=rng, dist=self._dist, n=self._mi_estimate_sample
            )
            return self._mi
