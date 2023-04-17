"""Tests of the utilities."""
import numpy as np
import numpy.testing as nptest
import pytest

import bmi.utils as utils
from bmi.samplers import SplitMultinormal


def test_sample_save_and_read(tmp_path) -> None:
    """Tests if saving and loading the data to the disk doesn't change anything."""
    target_path = tmp_path / "my-test-task"

    sampler = SplitMultinormal(dim_x=1, dim_y=1, covariance=np.asarray([[1.0, 0.7], [0.7, 1.0]]))
    samples_x, samples_y = sampler.sample(n_points=1000, rng=0)

    utils.save_sample(target_path, samples_x, samples_y)

    dim_x, dim_y = utils.read_sample_dims(target_path)
    assert dim_x == 1
    assert dim_y == 1

    samples_x_read, samples_y_read = utils.read_sample(target_path)

    assert np.allclose(samples_x, samples_x_read)
    assert np.allclose(samples_y, samples_y_read)


class LoadSampleInWrongFormat:
    """Tests for loading corrupted samples."""

    @pytest.mark.skip("Not implemented!")
    def test_dataframe_in_wrong_format(self) -> None:
        """Checks if an informative error is raised when
        a data frame in wrong format is passed."""
        raise NotImplementedError


class TestProductSpace:
    """Tests of the ProductSpace class"""

    @pytest.mark.parametrize("standardize", [True, False])
    def test_standardization_is_pure(
        self, standardize: bool, n_points: int = 10, dim_x: int = 2, dim_y: int = 3
    ) -> None:
        """Check if standardization doesn't change the inputs."""
        rng = np.random.default_rng(12)

        sqrt_cov_x = rng.normal(0, 1, size=(dim_x, dim_x))
        sqrt_cov_y = rng.normal(0.5, 1, size=(dim_y, dim_y))

        x = rng.multivariate_normal(np.ones(dim_x), sqrt_cov_x.T @ sqrt_cov_x, size=n_points)
        y = rng.multivariate_normal(
            rng.normal(loc=1, scale=3.0, size=dim_y), sqrt_cov_y.T @ sqrt_cov_y, size=n_points
        )
        x_copy = x.copy()
        y_copy = y.copy()

        utils.ProductSpace(standardize=standardize, x=x, y=y)

        nptest.assert_array_almost_equal(x, x_copy)
        nptest.assert_array_almost_equal(y, y_copy)

    def test_standardization_works(self, n_points: int = 30_000) -> None:
        rng = np.random.default_rng(111)

        mean_x = np.asarray([0.1, 5.0, 3.0])
        mean_y = np.asarray([2.0, 2.3])

        cov_x = np.diag([1.0, 0.4, 2.0])
        cov_y = np.asarray(
            [
                [1.0, 0.4],
                [0.4, 0.5**2],
            ]
        )

        x = rng.multivariate_normal(mean_x, cov_x, size=n_points)
        y = rng.multivariate_normal(mean_y, cov_y, size=n_points)

        space = utils.ProductSpace(standardize=True, x=x, y=y)

        # Check if means are zero
        nptest.assert_array_almost_equal(np.mean(space.x, axis=0), np.zeros(3), decimal=3)
        nptest.assert_array_almost_equal(np.mean(space.y, axis=0), np.zeros(2), decimal=3)

        cov_x_ = np.einsum("ij,ik->jk", space.x, space.x) / n_points
        cov_y_ = np.einsum("ij,ik->jk", space.y, space.y) / n_points

        # Check if X is whitened
        nptest.assert_array_almost_equal(cov_x_, np.eye(3), decimal=2)
        # The correlation for Y should be preserved
        expected_cov_y = np.asarray(
            [
                [1.0, 0.8],
                [0.8, 1.0],
            ]
        )
        nptest.assert_array_almost_equal(cov_y_, expected_cov_y, decimal=2)
