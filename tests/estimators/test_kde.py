"""Tests of the kernel density estimator."""

import numpy as np
import pytest

from bmi.estimators import KDEMutualInformationEstimator
from bmi.samplers import BivariateNormalSampler


def test_kde_estimator(n_points: int = 4_000, corr: float = 0.8) -> None:
    sampler = BivariateNormalSampler(correlation=corr)
    x, y = sampler.sample(n_points=n_points, rng=0)

    # standardize=False, so that we have differential entropies of uniform(0, 1)=0
    estimator = KDEMutualInformationEstimator(
        standardize=False, bandwidth_xy="scott", kernel_xy="tophat"
    )
    result = estimator.estimate_entropies(x, y)

    # Differential entropy of a Gaussian N(0, 1)
    constant = 2 * np.pi * np.e
    entropy_normal = 0.5 * np.log(constant)
    # Differential entropy of the bivariate Gaussian
    entropy_joint = 0.5 * np.log(constant**2 * (1 - corr**2))

    assert result.entropy_x == pytest.approx(entropy_normal, rel=0.05)
    assert result.entropy_y == pytest.approx(entropy_normal, rel=0.05)
    assert result.entropy_xy == pytest.approx(entropy_joint, rel=0.05)

    assert result.mutual_information == pytest.approx(sampler.mutual_information(), abs=0.03)

    assert result.mutual_information == pytest.approx(estimator.estimate(x, y))
