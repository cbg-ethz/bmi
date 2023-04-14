# JULIA_ESTIMATOR_FACTORIES = [
#    lambda: benchmark.JuliaEstimatorKSG(neighbors=10, variant=1),
#    lambda: benchmark.JuliaEstimatorKSG(neighbors=7, variant=2),
#    lambda: benchmark.JuliaEstimatorKernel(bandwidth=2.0),
#    lambda: benchmark.JuliaEstimatorTransfer(bins=4),
#    lambda: benchmark.JuliaEstimatorHistogram(bins=3),
# ]
#
#
# @pytest.mark.parametrize("n_samples", [300])
# @pytest.mark.parametrize("estimator_factory", JULIA_ESTIMATOR_FACTORIES)
# @pytest.mark.requires_julia
# def test_julia_estimators_smoke(
#    tmp_path,
#    estimator_factory,
#    n_samples: int,
#    correlation: float = 0.85,
#    task_id: str = "created-test-task",
# ) -> None:
#    """A simple test whether Julia estimators run.
#    We don't compare the values to the ground truth as we don't have
#    so many samples and even for 5,000 samples Julia's KSG estimators
#    seem to have strong positive bias."""
#    task_dir = tmp_path / "taskdir"
#
#    sampler = BivariateNormalSampler(correlation=correlation)
#
#    task = benchmark.generate_task(
#        seeds=[1], task_id=task_id, sampler=sampler, n_samples=n_samples
#    )
#
#    task.save(task_dir)
#
#    estimator = estimator_factory()
#    result = estimator.estimate(task_dir, seed=1)
#
#    # Check whether the task ID is right
#    assert result.task_id == task_id
#
#    # Check whether the seeds are right
#    assert result.seed == 1
#
#    if not result.mi_estimate == pytest.approx(task.mi_true, abs=0.1, rel=0.3):
#        warnings.warn(
#            f"Estimator {estimator.estimator_id()}: "
#            f"{result.mi_estimate:.2f} 1= {task.mi_true:.2f}."
#        )
#
#    assert result.mi_estimate > 0
