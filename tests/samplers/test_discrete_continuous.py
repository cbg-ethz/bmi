import jax.numpy as jnp
import pytest

import bmi
from bmi.utils import add_noise

# isort: off
from bmi.samplers._discrete_continuous import (
    DiscreteUniformMixtureSampler,
    MultivariateDiscreteUniformMixtureSampler,
    ZeroInflatedPoissonizationSampler,
)

# isort: on


@pytest.mark.parametrize("truncation", [100, 1000])
def test_truncation_is_robust(truncation: int) -> None:
    sampler = ZeroInflatedPoissonizationSampler()
    approximation_default = sampler.mutual_information()
    approximation_truncated = sampler.mutual_information(truncation=truncation)

    assert pytest.approx(approximation_default, abs=1e-3) == approximation_truncated


def get_samplers():
    yield DiscreteUniformMixtureSampler(n_discrete=5, use_discrete_x=True)
    yield DiscreteUniformMixtureSampler(n_discrete=3, use_discrete_x=False)
    yield MultivariateDiscreteUniformMixtureSampler(ns_discrete=[4, 5], use_discrete_x=True)
    yield ZeroInflatedPoissonizationSampler(p=0)
    yield ZeroInflatedPoissonizationSampler(p=0.15, use_discrete_y=True)
    yield ZeroInflatedPoissonizationSampler(p=0.7, use_discrete_y=False)


@pytest.mark.parametrize("sampler", get_samplers())
def test_mutual_information_is_right(sampler) -> None:
    xs, ys = sampler.sample(n_points=7_000, rng=0)

    xs = add_noise(xs, 1e-7)
    ys = add_noise(ys, 1e-7)

    mi_estimate = bmi.estimators.KSGEnsembleFirstEstimator(neighborhoods=(5,)).estimate(xs, ys)
    mi_ground_truth = sampler.mutual_information()

    assert pytest.approx(mi_estimate, abs=0.05, rel=0.05) == mi_ground_truth


def test_discrete_x():
    discrete_sampler = DiscreteUniformMixtureSampler(n_discrete=5, use_discrete_x=True)
    continuous_sampler = DiscreteUniformMixtureSampler(n_discrete=5, use_discrete_x=False)

    xs_discrete, _ = discrete_sampler.sample(n_points=10, rng=0)
    xs_continuous, _ = continuous_sampler.sample(n_points=10, rng=0)

    assert (xs_discrete == xs_continuous).all()
    assert xs_discrete.dtype == jnp.int32
    assert xs_continuous.dtype == jnp.float32


def test_discrete_y():
    discrete_sampler = ZeroInflatedPoissonizationSampler(p=0.15, use_discrete_y=True)
    continuous_sampler = ZeroInflatedPoissonizationSampler(p=0.15, use_discrete_y=False)

    _, ys_discrete = discrete_sampler.sample(n_points=10, rng=0)
    _, ys_continuous = continuous_sampler.sample(n_points=10, rng=0)

    assert (ys_discrete == ys_continuous).all()
    assert ys_discrete.dtype == jnp.int32
    assert ys_continuous.dtype == jnp.float32
