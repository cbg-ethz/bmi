import numpy as np
import pytest

import bmi.transforms.rotate as rt


class TestSpiral:
    pass


class TestSkewSymmetrize:
    @pytest.mark.parametrize("n", (3, 4, 5))
    @pytest.mark.parametrize("k", (3, 8))
    @pytest.mark.parametrize("seed", range(3))
    def test_symmetric_zero(self, n: int, k: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        w = rng.normal(size=(n, k))
        a = w @ w.T

        s = rt.skew_symmetrize(a)
        assert s.shape == a.shape
        assert np.allclose(s, np.zeros_like(s))

    @pytest.mark.parametrize("n", (3, 4, 5))
    @pytest.mark.parametrize("seed", range(3))
    def test_fixed_point(self, n: int, seed: int) -> None:
        """Check whether s(s(A)) = s(A) for any matrix A"""
        rng = np.random.default_rng(seed)
        a = rng.normal(size=(n, n))

        s_a = rt.skew_symmetrize(a)
        s_s_a = rt.skew_symmetrize(s_a)
        assert np.allclose(s_a, s_s_a)

    @pytest.mark.parametrize("n", (2, 3, 7))
    @pytest.mark.parametrize("seed", range(2))
    def test_skew_symmetric(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        a = rng.normal(size=(n, n))
        s_a = rt.skew_symmetrize(a)

        assert np.allclose(s_a.T, -s_a)


class TestSOGenerator:
    pass
