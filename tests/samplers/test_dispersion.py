from typing import Optional

import numpy as np
import numpy.testing as nptest
import pytest

import bmi.samplers._dispersion as dis


@pytest.mark.parametrize("dim_x", (1, 2))
@pytest.mark.parametrize("dim_y", (1, 3))
@pytest.mark.parametrize("n_interacting", (0, 1))
@pytest.mark.parametrize("beta_y", (None, 0.2))
@pytest.mark.parametrize("eta_y", (None, 0.3))
@pytest.mark.parametrize("epsilon_y", (None, 0.32))
def test_covariance_agrees_with_mixing(
    dim_x: int,
    dim_y: int,
    n_interacting: int,
    beta_y: Optional[float],
    eta_y: Optional[float],
    epsilon_y: Optional[float],
    eta_x: float = 0.2,
    lambd: float = 0.4,
    beta_x: float = 0.1,
    epsilon_x: float = 1.0,
    alpha: float = 0.1,
) -> None:
    params = dis.GaussianLVMParametrization(
        dim_x=dim_x,
        dim_y=dim_y,
        n_interacting=n_interacting,
        alpha=alpha,
        beta_x=beta_x,
        beta_y=beta_y,
        epsilon_x=epsilon_x,
        epsilon_y=epsilon_y,
        eta_x=eta_x,
        eta_y=eta_y,
        lambd=lambd,
    )

    mixing = params.mixing
    assert mixing.shape == (dim_x + dim_y, params.n_latent)
    cov = params.covariance
    assert cov.shape == (dim_x + dim_y, dim_x + dim_y)

    nptest.assert_allclose(cov, np.einsum("il,jl->ij", mixing, mixing))


@pytest.mark.parametrize("n_interacting", [3, 4, 5])
def test_n_interacting_small(n_interacting: int) -> None:
    with pytest.raises(ValueError):
        dis.GaussianLVMParametrization(
            dim_x=2,
            dim_y=3,
            n_interacting=n_interacting,
            alpha=1.0,
            lambd=2.0,
        )


@pytest.mark.parametrize("beta_y", (None, 0.0))
@pytest.mark.parametrize("eta_y", (None, 0.12))
@pytest.mark.parametrize("epsilon_y", (None, 0.4))
def test_mixing_manually(
    beta_y: Optional[float],
    eta_y: Optional[float],
    epsilon_y: Optional[float],
    dim_x: int = 2,
    dim_y: int = 2,
    n_interacting: int = 1,
    alpha: float = 0.1,
    beta_x: float = 0.2,
    epsilon_x: float = 0.2,
    eta_x: float = 0.32,
    lambd: float = 0.123,
) -> None:
    params = dis.GaussianLVMParametrization(
        dim_x=dim_x,
        dim_y=dim_y,
        n_interacting=n_interacting,
        alpha=alpha,
        lambd=lambd,
        epsilon_x=epsilon_x,
        epsilon_y=epsilon_y,
        eta_x=eta_x,
        eta_y=eta_y,
        beta_x=beta_x,
        beta_y=beta_y,
    )

    if beta_y is None:
        beta_y = beta_x
    if eta_y is None:
        eta_y = eta_x
    if epsilon_y is None:
        epsilon_y = epsilon_x

    # n_latent = 3 + 1 + 2 + 2 + 1 + 1 = 10
    expected_mixing = np.asarray(
        [
            [alpha, beta_x, 0, lambd, epsilon_x, 0, 0, 0, 0, 0],
            [alpha, beta_x, 0, 0, 0, epsilon_x, 0, 0, eta_x, 0],
            [alpha, 0, beta_y, lambd, 0, 0, epsilon_y, 0, 0, 0],
            [alpha, 0, beta_y, 0, 0, 0, 0, epsilon_y, 0, eta_y],
        ]
    )

    nptest.assert_allclose(expected_mixing, params.mixing)


@pytest.mark.parametrize("dim_x", (1, 3))
@pytest.mark.parametrize("dim_y", (1, 2))
def test_correlation(dim_x: int, dim_y: int) -> None:
    params = dis.GaussianLVMParametrization(
        dim_x=dim_x,
        dim_y=dim_y,
        alpha=0.1,
        n_interacting=1,
        lambd=1.0,
    )
    corr = params.correlation

    assert params.covariance.shape == corr.shape
    nptest.assert_allclose(np.diag(corr), np.ones(dim_x + dim_y))
    assert np.min(corr) >= 0
    assert np.max(corr) == pytest.approx(1.0)


@pytest.mark.parametrize("alpha", (-0.3, 0.2))
@pytest.mark.parametrize("epsilon", (1.0, 0.2))
def test_dense(alpha: float, epsilon: float) -> None:
    params = dis.DenseLVMParametrization(
        dim_x=2,
        dim_y=1,
        alpha=alpha,
        epsilon=epsilon,
    )

    corr = params.correlation
    value = params.correlation_strength

    assert sorted(np.unique(corr.ravel())) == pytest.approx([value, 1.0])


@pytest.mark.parametrize("beta", (0.2, 0.3))
@pytest.mark.parametrize("lambd", (1.0, 2.0))
@pytest.mark.parametrize("eta", (None, 0.1))
def test_sparse(beta: float, lambd: float, eta: Optional[float], epsilon: float = 0.3) -> None:
    params = dis.SparseLVMParametrization(
        dim_x=2,
        dim_y=1,
        n_interacting=1,
        lambd=lambd,
        beta=beta,
        epsilon=epsilon,
        eta=eta,
    )

    # We have strong interactions
    assert params.correlation[0, 1] < params.correlation_interacting

    assert params.correlation[0, 2] == pytest.approx(params.correlation_interacting)
    assert params.correlation[1, 2] == pytest.approx(0, abs=0.005)
