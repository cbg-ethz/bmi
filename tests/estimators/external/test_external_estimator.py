import numpy as np
import pytest

import bmi.estimators.external.external_estimator as external_estimator


@pytest.mark.parametrize("mi", [-0.1, 0.0, 12.3, float("nan"), 1e3])
def test_run_command_and_read_mi(mi: float):
    mi_read = external_estimator._run_command_and_read_mi(["echo", str(mi)])
    assert np.allclose(mi, mi_read, equal_nan=True)
