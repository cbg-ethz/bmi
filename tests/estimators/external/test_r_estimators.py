import pytest

import bmi.estimators.external.r_estimators as r_estimators
from bmi.samplers._splitmultinormal import BivariateNormalSampler

# Factories for the estimators.
# This is e.g., useful when testing on the server
# as the R files don't need to exist there.
# TODO(Pawel): Consider moving the R files into somewhere in the module
#   so at least the path always exists.
R_ESTIMATOR_FACTORIES = [
    lambda: r_estimators.RKSGEstimator(variant=1, neighbors=5),
    lambda: r_estimators.RKSGEstimator(variant=2, neighbors=7),
    # TODO(frdrc): enable once we add RLNNEstimator
    # lambda: r_estimators.RLNNEstimator(neighbors=8, truncation=20),
    # TODO(frdrc): enable once we add RBNSLEstimator
    # lambda: r_estimators.REstimatorBNSL(proc=0)
]


@pytest.mark.parametrize("n_samples", [300])
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("estimator_factory", R_ESTIMATOR_FACTORIES)
@pytest.mark.requires_r
def test_r_estimators(
    estimator_factory,
    n_samples: int,
    seed: int,
    correlation: float = 0.85,
) -> None:
    sampler = BivariateNormalSampler(correlation=correlation)
    samples_x, samples_y = sampler.sample(n_samples, seed)

    estimator = estimator_factory()

    mi_true = sampler.mutual_information()
    mi_estimate = estimator.estimate(samples_x, samples_y)

    # Check if the MI estimates are somewhat close to the true value.
    # We use a small sample size, so let's not be too strict.
    assert mi_estimate == pytest.approx(mi_true, rel=0.35, abs=0.15)
    assert abs(mi_estimate - mi_true) > 1e-5
