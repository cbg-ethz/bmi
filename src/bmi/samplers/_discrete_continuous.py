from typing import Optional, Sequence, Union

import jax.numpy as jnp
import numpy as np
from jax import random

from bmi.interface import KeyArray
from bmi.samplers._independent_coordinates import IndependentConcatenationSampler
from bmi.samplers.base import BaseSampler, cast_to_rng


def _cite_gao_2017() -> str:
    return (
        "@inproceedings{Gao2017-DiscreteContinuousMI,\n"
        + "  author = {Gao, Weihao and Kannan, Sreeram and Oh, Sewoong and Viswanath, Pramod},\n"
        + "  booktitle = {Advances in Neural Information Processing Systems},\n"
        + "  editor = {I. Guyon and U. Von Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},\n"  # noqa: E501
        + "  pages = {},\n"
        + "  publisher = {Curran Associates, Inc.},\n"
        + "  title = {Estimating Mutual Information for Discrete-Continuous Mixtures},\n"
        + "  url = {https://proceedings.neurips.cc/paper_files/paper/2017/file/ef72d53990bc4805684c9b61fa64a102-Paper.pdf},\n"  # noqa: E501
        + "  volume = {30},\n"
        + "  year = {2017}n"
        + "}"
    )


class DiscreteUniformMixtureSampler(BaseSampler):
    """Sampler from Gao et al. (2017) for the discrete-continuous mixture model.

    X ~ Categorical(1/m, ..., 1/m) is between {0, ..., m-1}
    Y | X ~ Uniform(X, X+2)
    """

    def __init__(self, *, n_discrete: int = 5, use_discrete_x: bool = True) -> None:
        """
        Args:
            n_discrete: number of discrete values to sample X from
            use_discrete_x: if True, X will be an integer. If False, X will be casted to a float.
        """
        super().__init__(dim_x=1, dim_y=1)

        if n_discrete <= 0:
            raise ValueError(f"n_discrete must be positive, was {n_discrete}.")

        self._n_discrete = n_discrete

        if use_discrete_x:
            self._x_factor = 1
        else:
            self._x_factor = 1.0

    def sample(self, n_points: int, rng: Union[int, KeyArray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        rng = cast_to_rng(rng)
        key_x, key_y = random.split(rng)
        xs = random.randint(key_x, shape=(n_points, 1), minval=0, maxval=self._n_discrete)
        uniforms = random.uniform(key_y, shape=(n_points, 1), minval=0.0, maxval=2.0)
        ys = xs + uniforms

        xs = xs * self._x_factor
        return xs, ys

    def mutual_information(self) -> float:
        m = self._n_discrete
        return jnp.log(m) - (m - 1) * jnp.log(2) / m

    @staticmethod
    def cite() -> str:
        return _cite_gao_2017()


class MultivariateDiscreteUniformMixtureSampler(IndependentConcatenationSampler):
    """Multivariate alternative for `DiscreteUniformMixtureSampler`,
    which is a concatenation of several independent samplers.

    Namely, the variables X = (X1, ..., Xk) and Y = (Y1, ..., Yk) are sampled as
      Xk ~ Categorical(1/m, ..., 1/m) is between {0, ..., m-1}
      Yk | Xk ~ Uniform(Xk, Xk + 2)

    See Also:
        IndependentConcatenationSampler
    """

    def __init__(self, *, ns_discrete: Sequence[int], use_discrete_x: bool = True) -> None:
        samplers = [
            DiscreteUniformMixtureSampler(n_discrete=n, use_discrete_x=use_discrete_x)
            for n in ns_discrete
        ]
        super().__init__(samplers=samplers)

    @staticmethod
    def cite() -> str:
        return _cite_gao_2017()


class ZeroInflatedPoissonizationSampler(BaseSampler):
    def __init__(self, p: float = 0.15) -> None:
        """
        Args:
            p: zero-inflation parameter. Must be in [0, 1).
        """
        if p < 0 or p >= 1:
            raise ValueError(f"p must be in [0, 1), was {p}.")
        self._p = p

    def sample(self, n_points: int, rng: Union[int, KeyArray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        rng = cast_to_rng(rng)
        key_x, key_zeros, key_poisson = random.split(rng, 3)
        xs = random.exponential(key_x, shape=(n_points, 1))
        # With probability p we have 0, with probability 1-p we have 1
        zeros = random.bernoulli(key_zeros, p=1 - self._p, shape=(n_points, 1))
        poissons = random.poisson(key_poisson, lam=xs)
        # Note that this corresponds to a mixture model: with probability p we sample
        # from Dirac delta at 0 and with probability 1-p we sample from Poisson
        ys = zeros * poissons

        return xs, ys

    def mutual_information(self, truncation: Optional[int] = None) -> float:
        """Ground-truth mutual information is equal to
        I(X; Y) = (1-p) (2 log 2 - gamma - S)
        where
        S = sum_{i=1}^{infinity} log(i) * 2^{-i},

        so that the approximation

        I(X; Y) = (1-p) * 0.3012

        holds.

        Args:
            truncation: if set to None, the above approximation will be used.
                Otherwise, the sum will be truncated at the given value.
        """
        assert truncation is None or truncation > 0

        if truncation is None:
            bracket = 0.3012
        else:
            i_arr = 1.0 * jnp.arange(1, truncation + 1)
            s = jnp.sum(jnp.log(i_arr) * jnp.exp2(-i_arr))
            bracket = 2 * jnp.log(2) - np.euler_gamma - s
            bracket = float(bracket)

        return (1 - self._p) * bracket

    @staticmethod
    def cite() -> str:
        return _cite_gao_2017()
