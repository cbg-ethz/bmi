"""Tests of the kernel density estimator."""
import pytest

from bmi.estimators.api import KDEMutualInformationEstimator
from bmi.samplers.api import BivariateUniformMarginsSampler


def test_kde_estimator(n_points: int = 10_000, corr: float = 0.9) -> None:
    sampler = BivariateUniformMarginsSampler(gaussian_correlation=corr)
    x, y = sampler.sample(n_points=n_points, rng=0)

    # standardize=False, so that we have differential entropies of uniform(0, 1)=0
    estimator = KDEMutualInformationEstimator(
        standardize=False, bandwidth_xy=0.1, kernel_xy="cosine"
    )
    result = estimator.estimate_entropies(x, y)

    assert result.entropy_x == pytest.approx(0)
    assert result.entropy_y == pytest.approx(0)

    assert result.mutual_information == pytest.approx(sampler.mutual_information())
