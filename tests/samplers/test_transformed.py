import jax.numpy as jnp
import numpy as np
import pytest

import bmi.samplers as samplers
import bmi.samplers._transformed as tr


@pytest.mark.parametrize("x", [2, 120, 5.0, "something"])
def test_identity_various_objects(x) -> None:
    assert x == tr.identity(x)


@pytest.mark.parametrize("x", [np.asarray([1.0, 4.0]), np.eye(4)])
def test_identity_array(x) -> None:
    assert np.allclose(x, tr.identity(x))


def get_gaussian_sampler(
    dim_x: int = 2, dim_y: int = 3, corr: float = 0.5
) -> samplers.SplitMultinormal:
    """Auxiliary function creating a base sampler."""
    cov = np.eye(dim_x + dim_y)
    cov[0, dim_x] = corr
    cov[dim_x, 0] = corr
    return samplers.SplitMultinormal(dim_x=dim_x, dim_y=dim_y, covariance=cov)


@pytest.mark.parametrize("dim_x", [3])
@pytest.mark.parametrize("dim_y", [2])
@pytest.mark.parametrize("corr", [0.3, 0.5])
@pytest.mark.parametrize("n_points", [10, 20])
def test_transformed_identity(
    dim_x: int, dim_y: int, corr: float, n_points: int, random_seed: int = 0
) -> None:
    base_sampler = get_gaussian_sampler(dim_x=dim_x, dim_y=dim_y, corr=corr)

    transformed_sampler = tr.TransformedSampler(base_sampler=base_sampler)

    assert transformed_sampler.dim_x == base_sampler.dim_x
    assert transformed_sampler.dim_y == base_sampler.dim_y
    assert transformed_sampler.mutual_information() == pytest.approx(
        base_sampler.mutual_information()
    )

    x_base, y_base = base_sampler.sample(n_points, rng=random_seed)
    x_transformed, y_transformed = transformed_sampler.sample(n_points, rng=random_seed)

    assert np.allclose(x_base, x_transformed)
    assert np.allclose(y_base, y_transformed)

    x_transformed_new, y_transformed_new = transformed_sampler.transform(x_base, y_base)
    assert np.allclose(x_transformed, x_transformed_new)
    assert np.allclose(y_transformed, y_transformed_new)


def cubic(x: np.ndarray) -> np.ndarray:
    return x**3


def test_transformed_cubic(
    dim_x: int = 5, dim_y: int = 3, corr: float = 0.4, n_points: int = 100, random_seed: int = 12
) -> None:
    base_sampler = get_gaussian_sampler(dim_x=dim_x, dim_y=dim_y, corr=corr)
    transformed_sampler = tr.TransformedSampler(base_sampler, transform_x=cubic)

    x_base, y_base = base_sampler.sample(n_points=n_points, rng=random_seed)
    x_transformed, y_transformed = transformed_sampler.sample(n_points, rng=random_seed)

    assert transformed_sampler.dim_x == base_sampler.dim_x
    assert transformed_sampler.dim_y == base_sampler.dim_y

    assert np.allclose(x_transformed, np.asarray([cubic(x) for x in x_base]))
    assert np.allclose(y_base, y_transformed)

    x_transformed_new, y_transformed_new = transformed_sampler.transform(x_base, y_base)
    assert np.allclose(x_transformed, x_transformed_new)
    assert np.allclose(y_transformed, y_transformed_new)


def embed(n: int):
    return lambda x: jnp.concatenate([x, jnp.zeros(n)])


def test_change_dimension(
    add_dim_y: int = 3,
    dim_x: int = 2,
    dim_y: int = 3,
    corr: float = 0.4,
    n_points: int = 50,
    random_seed: int = 12,
) -> None:
    base_sampler = get_gaussian_sampler(dim_x=dim_x, dim_y=dim_y, corr=corr)
    transformed_sampler = tr.TransformedSampler(
        base_sampler=base_sampler,
        transform_x=cubic,
        transform_y=embed(add_dim_y),
        add_dim_y=add_dim_y,
    )

    assert transformed_sampler.mutual_information() == base_sampler.mutual_information()

    assert transformed_sampler.dim_x == base_sampler.dim_x
    assert transformed_sampler.dim_y == base_sampler.dim_y + add_dim_y

    x_base, y_base = base_sampler.sample(n_points, rng=random_seed)
    x_transformed, y_transformed = transformed_sampler.transform(x_base, y_base)

    x_transformed_new, y_transformed_new = transformed_sampler.sample(n_points, rng=random_seed)
    assert np.allclose(x_transformed, x_transformed_new)
    assert np.allclose(y_transformed, y_transformed_new)

    assert x_transformed.shape == (n_points, transformed_sampler.dim_x)
    assert y_transformed.shape == (n_points, transformed_sampler.dim_y)

    assert np.allclose(x_transformed, np.asarray([cubic(x) for x in x_base]))
    assert np.allclose(y_transformed, np.asarray([embed(add_dim_y)(y) for y in y_base]))
