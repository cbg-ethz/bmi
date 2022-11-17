import warnings
from math import isclose

import numpy as np
import pytest
from jax import random
from sklearn.feature_selection import mutual_info_regression

import bmi.samplers.split_student_t as student


@pytest.mark.parametrize("x", (2, 4))
@pytest.mark.parametrize("y", (1, 2))
@pytest.mark.parametrize("df", (4, 9.5, 20))
@pytest.mark.parametrize("n_samples", (1000,))
@pytest.mark.parametrize("dispersion", (0.1, 0.5))
def test_samples_produced(x: int, y: int, n_samples: int, df: float, dispersion: float) -> None:
    """Tests whether the sampling returns the right shapes."""
    print(f"x={x} y={y} df={df} n_samples={n_samples}")

    rng = random.PRNGKey(111)
    mean = random.uniform(rng, shape=(x + y,))

    sampler = student.SplitStudentT(
        dim_x=x, dim_y=y, mean=mean, dispersion=dispersion * np.eye(x + y), df=df
    )
    x_sample, y_sample = sampler.sample(n_points=n_samples, rng=42)

    assert x_sample.shape == (
        n_samples,
        x,
    ), f"X sample shape: {x_sample.shape} != {(n_samples, x)}"
    assert y_sample.shape == (
        n_samples,
        y,
    ), f"Y sample shape: {y_sample.shape} != {(n_samples, y)}"

    xy_sample = np.hstack([x_sample, y_sample])
    assert xy_sample.shape == (
        n_samples,
        x + y,
    ), f"XxY sample shape: {xy_sample.shape} != {(n_samples, x+y)}"

    assert np.allclose(
        mean, xy_sample.mean(axis=0), rtol=0.12, atol=0.07
    ), f"Arrays different: {mean} != {xy_sample.mean(axis=0)}"


@pytest.mark.parametrize("pseudocorrelation", (0.2, 0.8))
@pytest.mark.parametrize(
    "df",
    (
        10,
        20,
        100,
    ),
)
def test_2d(
    pseudocorrelation: float,
    df: int,
    var_x: float = 1.0,
    var_y: float = 1.0,
    n_samples: int = 5000,
) -> None:
    # Note: var_x and var_y are *not* really the variances of the distribution,
    # but diagonal parameters of the dispersion matrix.
    cov_xy = pseudocorrelation * np.sqrt(var_x * var_y)

    dispersion = np.asarray(
        [
            [var_x, cov_xy],
            [cov_xy, var_y],
        ]
    )

    sampler = student.SplitStudentT(
        dim_x=1,
        dim_y=1,
        mean=np.zeros(2),
        dispersion=dispersion,
        df=df,
    )

    x, y = sampler.sample(n_points=n_samples, rng=10)
    # According to API, y must have shape (n,) rather than (n, 1)
    mi_estimate = mutual_info_regression(x, y.ravel(), random_state=5, n_neighbors=20)[0]
    true_mi = sampler.mutual_information()

    assert sampler.mutual_information() == pytest.approx(
        sampler.mi_normal() + sampler.mi_correction()
    )

    # We discovered that KSG is not very good at estimating MI of Student-t
    if not isclose(true_mi, mi_estimate, rel_tol=0.05, abs_tol=0.03):
        warnings.warn(f"True MI is {true_mi:.2f} and the estimate is {mi_estimate:.2f}.")


def get_mi_correction(dim_x: int, dim_y: int, df: int) -> float:
    return student.SplitStudentT(
        dim_x=dim_x, dim_y=dim_y, df=df, dispersion=np.eye(dim_x + dim_y)
    ).mi_correction()


@pytest.mark.parametrize("dim_x", (2, 3))
@pytest.mark.parametrize("dim_y", (3, 4))
@pytest.mark.parametrize("df", (50, 80))
def test_mi_correction_goes_to_zero(df: int, dim_x: int, dim_y: int) -> None:
    """The MI correction term should asymptotically behave like dim_x * dim_y / df
    for df -> infinity.
    """
    assert df * get_mi_correction(dim_x=dim_x, dim_y=dim_y, df=df) == pytest.approx(
        dim_x * dim_y, rel=0.1
    )
