import warnings

import numpy as np
import pytest

import bmi.benchmark.api as benchmark
from bmi.samplers.splitmultinormal import BivariateNormalSampler, SplitMultinormal

# Factories for the estimators.
# This is e.g., useful when testing on the server
# as the R files don't need to exist there.
# TODO(Pawel): Consider moving the R files into somewhere in the module
#   so at least the path always exists.
R_ESTIMATOR_FACTORIES = [
    lambda: benchmark.REstimatorKSG(variant=1, neighbors=5),
    lambda: benchmark.REstimatorKSG(variant=2, neighbors=7),
    lambda: benchmark.REstimatorLNN(neighbors=8, truncation=20),
]


@pytest.mark.parametrize("n_samples", [300])
@pytest.mark.parametrize("estimator_factory", R_ESTIMATOR_FACTORIES)
@pytest.mark.requires_r
def test_r_estimators(
    tmp_path,
    estimator_factory,
    n_samples: int,
    correlation: float = 0.85,
    task_id: str = "created-test-task",
) -> None:
    task_dir = tmp_path / "taskdir"

    sampler = BivariateNormalSampler(correlation=correlation)

    task = benchmark.generate_task(
        seeds=[1, 2], task_id=task_id, sampler=sampler, n_samples=n_samples
    )

    task.save(task_dir)

    estimator = estimator_factory()
    print(estimator.estimator_id())

    result1 = estimator.estimate(task_dir, seed=1)
    result2 = estimator.estimate(task_dir, seed=2)

    # Check whether the task ID is right
    assert result1.task_id == task_id
    assert result2.task_id == task_id

    # Check whether the seeds are right
    assert result1.seed == 1
    assert result2.seed == 2

    # Check whether the estimator ID is the same
    assert result1.estimator_id == result2.estimator_id

    # Check if the MI estimates are somewhat close to the true value.
    # We use a small sample size, so let's not be too strict.
    assert result1.mi_estimate == pytest.approx(task.mi_true, rel=0.35, abs=0.15)
    assert result2.mi_estimate == pytest.approx(task.mi_true, rel=0.35, abs=0.15)

    # Check whether the estimates are somewhat different, as the variance
    # should not be zero
    assert abs(result1.mi_estimate - result2.mi_estimate) > 1e-5


@pytest.mark.parametrize("dim_x", [1, 2])
@pytest.mark.parametrize("dim_y", [1, 2])
@pytest.mark.parametrize("n_samples", [300])
@pytest.mark.requires_r
def test_bnsl(
    tmp_path,
    dim_x: int,
    dim_y: int,
    n_samples: int,
    corr: float = 0.95,
    task_id: str = "test-task-bnsl",
) -> None:
    task_dir = tmp_path / "taskdir"
    cov = np.eye(dim_x + dim_y)
    cov[0, dim_x] = corr
    cov[dim_x, 0] = corr

    sampler = SplitMultinormal(dim_x=dim_x, dim_y=dim_y, covariance=cov)
    task = benchmark.generate_task(
        seeds=[1], task_id=task_id, sampler=sampler, n_samples=n_samples
    )
    task.save(task_dir)

    estimator = benchmark.REstimatorBNSL(proc=1)

    result = estimator.estimate(task_dir, seed=1)

    # Check whether the task ID is right
    assert result.task_id == task_id

    # Check whether the seeds are right
    assert result.seed == 1

    # Check if the MI estimates are somewhat close to the true value.
    # We use a small sample size, so let's not be too strict.
    # BNSL seem to have positive bias as well, though...
    assert result.mi_estimate == pytest.approx(task.mi_true, rel=0.3, abs=0.7)


JULIA_ESTIMATOR_FACTORIES = [
    lambda: benchmark.JuliaEstimatorKSG(neighbors=10, variant=1),
    lambda: benchmark.JuliaEstimatorKSG(neighbors=7, variant=2),
    lambda: benchmark.JuliaEstimatorKernel(bandwidth=2.0),
    lambda: benchmark.JuliaEstimatorTransfer(bins=4),
    lambda: benchmark.JuliaEstimatorHistogram(bins=3),
]


@pytest.mark.parametrize("n_samples", [300])
@pytest.mark.parametrize("estimator_factory", JULIA_ESTIMATOR_FACTORIES)
@pytest.mark.requires_julia
def test_julia_estimators_smoke(
    tmp_path,
    estimator_factory,
    n_samples: int,
    correlation: float = 0.85,
    task_id: str = "created-test-task",
) -> None:
    """A simple test whether Julia estimators run.
    We don't compare the values to the ground truth as we don't have
    so many samples and even for 5,000 samples Julia's KSG estimators
    seem to have strong positive bias."""
    task_dir = tmp_path / "taskdir"

    sampler = BivariateNormalSampler(correlation=correlation)

    task = benchmark.generate_task(
        seeds=[1], task_id=task_id, sampler=sampler, n_samples=n_samples
    )

    task.save(task_dir)

    estimator = estimator_factory()
    result = estimator.estimate(task_dir, seed=1)

    # Check whether the task ID is right
    assert result.task_id == task_id

    # Check whether the seeds are right
    assert result.seed == 1

    if not result.mi_estimate == pytest.approx(task.mi_true, abs=0.1, rel=0.3):
        warnings.warn(
            f"Estimator {estimator.estimator_id()}: "
            f"{result.mi_estimate:.2f} 1= {task.mi_true:.2f}."
        )

    assert result.mi_estimate > 0
