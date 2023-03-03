"""API of the neural estimators implemented in JAX."""

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import pydantic
from numpy.typing import ArrayLike

from bmi.estimators.neural import _backend_quadratic
from bmi.estimators.neural._interfaces import Critic, Point
from bmi.estimators.neural._nn import MLP, basic_fit
from bmi.interface import BaseModel, IMutualInformationPointEstimator


class NeuralEstimatorParams(BaseModel):
    mi_formula: str
    batch_size: pydantic.PositiveInt
    max_n_steps: pydantic.PositiveInt
    learning_rate: pydantic.PositiveFloat
    hidden_layers: list[int]
    seed: int


class NeuralEstimator(IMutualInformationPointEstimator):
    def __init__(
        self,
        mi_formula: Callable[[Critic, Point, Point], float],
        mi_formula_name: str,
        batch_size: int = 256,
        max_n_steps: int = 2_000,
        learning_rate: float = 0.1,
        hidden_layers: Sequence[int] = (5,),
        seed: int = 42,
    ):
        self._mi_formula = mi_formula
        self._params = NeuralEstimatorParams(
            mi_formula=mi_formula_name,
            batch_size=batch_size,
            max_n_steps=max_n_steps,
            learning_rate=learning_rate,
            hidden_layers=list(hidden_layers),
            seed=seed,
        )

    def parameters(self) -> NeuralEstimatorParams:
        return self._params

    def estimate(self, x: ArrayLike, y: ArrayLike) -> float:
        key = jax.random.PRNGKey(self._params.seed)
        key_mlp, key_fit = jax.random.split(key)

        xs = jnp.array(x)
        ys = jnp.array(y)

        critic = MLP(
            key=key_mlp,
            dim_x=xs.shape[-1],
            dim_y=ys.shape[-1],
            hidden_layers=self._params.hidden_layers,
        )

        # TODO(frdrc): allow for train/valid split
        train_history = basic_fit(
            rng=key_fit,
            critic=critic,
            mi_formula=self._mi_formula,
            xs=xs,
            ys=ys,
            batch_size=self._params.batch_size,
            max_n_steps=self._params.max_n_steps,
            learning_rate=self._params.learning_rate,
        )

        # TODO(frdrc): how about mean starting from best value?
        return train_history.final_mi


class InfoNceEstimator(NeuralEstimator):
    def __init__(
        self,
        batch_size: int = 256,
        max_n_steps: int = 2_000,
        learning_rate: float = 0.1,
        hidden_layers: Sequence[int] = (5,),
        seed: int = 42,
    ):
        super().__init__(
            _backend_quadratic.infonce,
            "InfoNCE",
            max_n_steps=max_n_steps,
            learning_rate=learning_rate,
            hidden_layers=hidden_layers,
            seed=seed,
        )


class NwjEstimator(NeuralEstimator):
    def __init__(
        self,
        batch_size: int = 256,
        max_n_steps: int = 2_000,
        learning_rate: float = 0.1,
        hidden_layers: Sequence[int] = (5,),
        seed: int = 42,
    ):
        super().__init__(
            _backend_quadratic.nwj,
            "NWJ",
            max_n_steps=max_n_steps,
            learning_rate=learning_rate,
            hidden_layers=hidden_layers,
            seed=seed,
        )


class DonsekVaradhanEstimator(NeuralEstimator):
    def __init__(
        self,
        batch_size: int = 256,
        max_n_steps: int = 2_000,
        learning_rate: float = 0.1,
        hidden_layers: Sequence[int] = (5,),
        seed: int = 42,
    ):
        super().__init__(
            _backend_quadratic.donsker_varadhan,
            "Donsker-Varadhan",
            max_n_steps=max_n_steps,
            learning_rate=learning_rate,
            hidden_layers=hidden_layers,
            seed=seed,
        )
