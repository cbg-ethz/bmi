import warnings

import jax
import numpy as np
import pytest

import bmi.transforms.rotate as rt


class TestSpiral:
    warnings.warn("Tests for the Spiral are not ready.")

    @pytest.mark.skip("Tests for the Spiral are not ready.")
    def test_spiral(self) -> None:
        raise NotImplementedError


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

    def test_jittable(self) -> None:
        jax.jit(rt.skew_symmetrize)

    @pytest.mark.parametrize("dim", (2, 5))
    @pytest.mark.parametrize("n_matrices", (3, 4))
    def test_vmappable(self, dim: int, n_matrices: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        matrices = rng.normal(size=(n_matrices, dim, dim))
        s_matrices = jax.vmap(rt.skew_symmetrize)(matrices)
        assert np.allclose(s_matrices, np.asarray([rt.skew_symmetrize(m) for m in matrices]))


class TestSOGenerator:
    def test_2d_example(self) -> None:
        expected = np.asarray([[0, 1], [-1, 0]])
        assert np.allclose(rt.so_generator(2, 0, 1), expected)

    def test_3d_example(self) -> None:
        expected = np.asarray([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
        assert np.allclose(rt.so_generator(3, 0, 1), expected)

    @pytest.mark.parametrize("n", (2, 4))
    def test_indices_right_order(self, n: int) -> None:
        for i in range(n):
            for j in range(i + 1):
                with pytest.raises(ValueError):
                    rt.so_generator(n, i, j)

    @pytest.mark.parametrize("n", (2, 3, 5))
    def test_basic_statistic(self, n: int) -> None:
        """Checks if all entries are positive and sum up to 1."""
        for i in range(n):
            for j in range(i + 1, n):
                a = rt.so_generator(n, i, j)
                assert np.allclose(a, -a.T)

                assert np.min(a) == pytest.approx(-1)
                assert np.sum(a) == pytest.approx(0)
                assert np.max(a) == pytest.approx(1)

    def test_raises(self) -> None:
        with pytest.raises(ValueError):
            rt.so_generator(3, 1, 1)
        with pytest.raises(ValueError):
            rt.so_generator(3, 4, 1)
        with pytest.raises(ValueError):
            rt.so_generator(2, 0, 0)
