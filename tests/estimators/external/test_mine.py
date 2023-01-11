import numpy as np
import pytest

from bmi.interface import IMutualInformationPointEstimator
from bmi.samplers.api import SplitMultinormal


def get_module():
    import bmi.estimators.external.mine as mine

    return mine


def get_estimator() -> IMutualInformationPointEstimator:
    import torch  # pytype: disable=import-error

    mine = get_module()
    device = "gpu" if torch.cuda.is_available() else "cpu"
    return mine.MutualInformationNeuralEstimator(
        max_epochs=100,
        device=device,
        hidden_layers=(15, 10),
    )


@pytest.mark.requires_mine_pytorch
def test_estimate_mi_2d(n_points: int = 15_000, correlation: float = 0.8) -> None:
    covariance = np.array(
        [
            [1.0, correlation],
            [correlation, 1.0],
        ]
    )
    distribution = SplitMultinormal(
        dim_x=1,
        dim_y=1,
        mean=np.zeros(2),
        covariance=covariance,
    )
    points_x, points_y = distribution.sample(n_points, rng=19)

    estimator = get_estimator()

    mine = get_module()
    assert isinstance(estimator.parameters(), mine.AllMINEParams)

    true_mi = distribution.mutual_information()
    estimate = estimator.estimate(points_x, points_y)

    assert estimate == pytest.approx(true_mi, abs=0.03, rel=0.02)


@pytest.mark.requires_mine_pytorch
def test_params_can_be_created() -> None:
    import bmi.estimators.external.mine as mine

    params = mine.AllMINEParams()
    assert isinstance(params, mine.AllMINEParams)
