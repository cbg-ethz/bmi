from typing import Optional

import numpy as np
import numpy.testing as nptest
import pytest

import bmi.samplers._dispersion as dis


@pytest.mark.parametrize("dim_x", (1, 2))
@pytest.mark.parametrize("dim_y", (1, 3))
@pytest.mark.parametrize("n_interacting", (0, 1))
@pytest.mark.parametrize("beta_x", (0, 0.1))
@pytest.mark.parametrize("beta_y", (None, 0.2))
@pytest.mark.parametrize("eta_x", (0, 0.2))
@pytest.mark.parametrize("eta_y", (None, 0.3))
@pytest.mark.parametrize("lambd", (0, 1.0))
@pytest.mark.parametrize("epsilon_x", (1.0, 2.0))
@pytest.mark.parametrize("epsilon_y", (None, 0.32))
def test_mixing_and_covariance(
    dim_x: int,
    dim_y: int,
    n_interacting: int,
    beta_x: float,
    beta_y: Optional[float],
    eta_x: float,
    eta_y: Optional[float],
    lambd: float,
    epsilon_x: float,
    epsilon_y: Optional[float],
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

    mixing = params.mixing()
    assert mixing.shape == (dim_x + dim_y, params.n_latent)
    cov = params.covariance()
    assert cov.shape == (dim_x + dim_y, dim_x + dim_y)

    nptest.assert_allclose(cov, np.einsum("il,jl->ij", mixing, mixing))
