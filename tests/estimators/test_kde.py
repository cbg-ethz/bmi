"""Tests of the kernel density estimator."""
import pytest

from bmi.estimators.api import KDEMutualInformationEstimator
from bmi.samplers.api import BivariateUniformMarginsSampler


def test_kde_estimator(n_points: int = 20_000, corr: float = 0.8) -> None:
    sampler = BivariateUniformMarginsSampler(gaussian_correlation=corr)
    x, y = sampler.sample(n_points=n_points, rng=0)

    estimator = KDEMutualInformationEstimator()
    estimate = estimator.estimate(x, y)

    assert estimate == pytest.approx(sampler.mutual_information())
