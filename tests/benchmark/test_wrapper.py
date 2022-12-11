import pytest

import bmi.benchmark.api as benchmark
from bmi.samplers.splitmultinormal import BivariateNormalSampler

ESTIMATORS = [
    benchmark.REstimatorKSG(variant=1, neighbors=5),
    benchmark.REstimatorKSG(variant=2, neighbors=7),
    benchmark.REstimatorLNN(neighbors=8, truncation=20),
]


@pytest.mark.parametrize("n_samples", [300])
@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.requires_r
def test_ksg_r_runs(
    tmp_path,
    estimator,
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
