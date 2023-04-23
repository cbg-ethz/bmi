"""API of the neural estimators implemented in JAX."""
from typing import Any, Callable, Literal, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import pydantic
import sklearn.model_selection as msel
from numpy.typing import ArrayLike

from bmi.estimators.neural import _backend_linear_memory, _backend_quadratic_memory
from bmi.estimators.neural._basic_training import basic_training
from bmi.estimators.neural._critics import MLP
from bmi.estimators.neural._types import BatchedPoints, Critic, Point
from bmi.interface import BaseModel, EstimateResult, IMutualInformationPointEstimator
from bmi.utils import ProductSpace

_DEFAULT_BATCH_SIZE = 256
_DEFAULT_N_STEPS: int = 10_000
_DEFAULT_TRAIN_TEST_SPLIT: float = 0.5
_DEFAULT_TEST_EVERY_N: int = 250
_DEFAULT_HIDDEN_LAYERS: tuple[int, ...] = (16, 8)
_DEFAULT_LEARNING_RATE: float = 0.1
_DEFAULT_TRAIN_BACKEND: Literal["quadratic", "linear"] = "quadratic"
_DEFAULT_STANDARDIZE: bool = True
_DEFAULT_VERBOSE: bool = True
_DEFAULT_SEED: int = 42


class NeuralEstimatorParams(BaseModel):
    mi_formula: str
    batch_size: pydantic.PositiveInt
    max_n_steps: pydantic.PositiveInt
    train_test_split: Optional[float]
    test_every_n_steps: int
    learning_rate: pydantic.PositiveFloat
    standardize: bool
    seed: int
    critic_params: Optional[BaseModel]


def train_test_split(
    xs: BatchedPoints,
    ys: BatchedPoints,
    train_size: Optional[float],
    key: jax.random.PRNGKeyArray,
) -> tuple[BatchedPoints, BatchedPoints, BatchedPoints, BatchedPoints]:
    if train_size is None:
        return xs, xs, ys, ys

    else:
        # get random int from jax key
        random_state = int(jax.random.randint(key, (1,), 0, 1000))

        xs_train, xs_test, ys_train, ys_test = msel.train_test_split(
            xs,
            ys,
            train_size=train_size,
            random_state=random_state,
        )

        return xs_train, xs_test, ys_train, ys_test


class NeuralEstimatorBase(IMutualInformationPointEstimator):
    def __init__(
        self,
        mi_formula: Callable[[Critic, Point, Point], float],
        mi_formula_name: str,
        critic_factory: Callable[[Any, int, int], eqx.Module],
        critic_params: Optional[BaseModel] = None,
        mi_formula_test: Optional[Callable[[Critic, Point, Point], float]] = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        max_n_steps: int = _DEFAULT_N_STEPS,
        train_test_split: Optional[float] = _DEFAULT_TRAIN_TEST_SPLIT,
        test_every_n_steps: int = _DEFAULT_TEST_EVERY_N,
        learning_rate: float = _DEFAULT_LEARNING_RATE,
        standardize: bool = _DEFAULT_STANDARDIZE,
        verbose: bool = _DEFAULT_VERBOSE,
        seed: int = _DEFAULT_SEED,
    ) -> None:
        """
        Args:
            mi_formula: loss function to be used
            mi_formula_name: name of the loss used
            critic_factory: factory method for the critic function. Should take as arguments
              (JAX random key, dim_x, dim_y) and initialize a critic method
            critic_params: hyperparameters of the critic
        """
        self._mi_formula = mi_formula
        self._mi_formula_test = mi_formula_test or mi_formula
        self._verbose = verbose

        self._critic_factory = critic_factory

        self._params = NeuralEstimatorParams(
            mi_formula=mi_formula_name,
            batch_size=batch_size,
            max_n_steps=max_n_steps,
            train_test_split=train_test_split,
            test_every_n_steps=test_every_n_steps,
            learning_rate=learning_rate,
            standardize=standardize,
            seed=seed,
            critic_params=critic_params,
        )

    def parameters(self) -> NeuralEstimatorParams:
        return self._params

    def estimate_with_info(self, x: ArrayLike, y: ArrayLike) -> EstimateResult:
        key = jax.random.PRNGKey(self._params.seed)
        key_init, key_split, key_fit = jax.random.split(key, 3)

        # standardize the data, note we do so before splitting into train/test
        space = ProductSpace(x, y, standardize=self._params.standardize)
        xs, ys = jnp.asarray(space.x), jnp.asarray(space.y)

        # split
        xs_train, xs_test, ys_train, ys_test = train_test_split(
            xs, ys, train_size=self._params.train_test_split, key=key_split
        )

        # initialize critic
        critic = self._critic_factory(key_init, xs_train.shape[-1], ys_train.shape[-1])

        training_log = basic_training(
            rng=key_fit,
            critic=critic,
            mi_formula=self._mi_formula,
            xs=xs_train,
            ys=ys_train,
            mi_formula_test=self._mi_formula_test,
            xs_test=xs_test,
            ys_test=ys_test,
            batch_size=self._params.batch_size,
            test_every_n_steps=self._params.test_every_n_steps,
            max_n_steps=self._params.max_n_steps,
            learning_rate=self._params.learning_rate,
            verbose=self._verbose,
        )

        return EstimateResult(
            mi_estimate=training_log.final_mi,
            additional_information=training_log.additional_information,
        )

    def estimate(self, x: ArrayLike, y: ArrayLike) -> float:
        return self.estimate_with_info(x, y).mi_estimate


class MLPParams(BaseModel):
    hidden_layers: list[int]


def _mlp_init(hidden_layers: Sequence[int]) -> Callable[[Any, int, int], eqx.Module]:
    def factory(key, dim_x: int, dim_y: int) -> eqx.Module:
        return MLP(key=key, dim_x=dim_x, dim_y=dim_y, hidden_layers=hidden_layers)

    return factory


class InfoNCEEstimator(NeuralEstimatorBase):
    def __init__(
        self,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        max_n_steps: int = _DEFAULT_N_STEPS,
        train_test_split: Optional[float] = _DEFAULT_TRAIN_TEST_SPLIT,
        test_every_n_steps: int = _DEFAULT_TEST_EVERY_N,
        learning_rate: float = _DEFAULT_LEARNING_RATE,
        hidden_layers: Sequence[int] = _DEFAULT_HIDDEN_LAYERS,
        standardize: bool = _DEFAULT_STANDARDIZE,
        verbose: bool = _DEFAULT_VERBOSE,
        seed: int = _DEFAULT_SEED,
        _train_backend: Literal["quadratic", "linear"] = _DEFAULT_TRAIN_BACKEND,
    ) -> None:
        if _train_backend == "quadratic":
            mi_formula = _backend_quadratic_memory.infonce
        elif _train_backend == "linear":
            mi_formula = _backend_linear_memory.infonce
        else:
            raise ValueError(f"Backend {_train_backend} not known.")

        hidden_layers = list(hidden_layers)

        super().__init__(
            mi_formula=mi_formula,
            mi_formula_name="InfoNCE",
            critic_factory=_mlp_init(hidden_layers),
            critic_params=MLPParams(hidden_layers=hidden_layers),
            mi_formula_test=_backend_linear_memory.infonce,
            max_n_steps=max_n_steps,
            train_test_split=train_test_split,
            test_every_n_steps=test_every_n_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            standardize=standardize,
            verbose=verbose,
            seed=seed,
        )


class NWJEstimator(NeuralEstimatorBase):
    def __init__(
        self,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        max_n_steps: int = _DEFAULT_N_STEPS,
        train_test_split: Optional[float] = _DEFAULT_TRAIN_TEST_SPLIT,
        test_every_n_steps: int = _DEFAULT_TEST_EVERY_N,
        learning_rate: float = _DEFAULT_LEARNING_RATE,
        hidden_layers: Sequence[int] = _DEFAULT_HIDDEN_LAYERS,
        standardize: bool = _DEFAULT_STANDARDIZE,
        verbose: bool = _DEFAULT_VERBOSE,
        seed: int = _DEFAULT_SEED,
        _train_backend: Literal["quadratic", "linear"] = _DEFAULT_TRAIN_BACKEND,
    ) -> None:
        if _train_backend == "quadratic":
            mi_formula = _backend_quadratic_memory.nwj
        elif _train_backend == "linear":
            mi_formula = _backend_linear_memory.nwj
        else:
            raise ValueError(f"Backend {_train_backend} not known.")

        hidden_layers = list(hidden_layers)

        super().__init__(
            mi_formula=mi_formula,
            mi_formula_name="NWJ",
            critic_factory=_mlp_init(hidden_layers),
            critic_params=MLPParams(hidden_layers=hidden_layers),
            mi_formula_test=_backend_linear_memory.nwj,
            max_n_steps=max_n_steps,
            train_test_split=train_test_split,
            test_every_n_steps=test_every_n_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            standardize=standardize,
            verbose=verbose,
            seed=seed,
        )


class DonskerVaradhanEstimator(NeuralEstimatorBase):
    def __init__(
        self,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        max_n_steps: int = _DEFAULT_N_STEPS,
        train_test_split: Optional[float] = _DEFAULT_TRAIN_TEST_SPLIT,
        test_every_n_steps: int = _DEFAULT_TEST_EVERY_N,
        learning_rate: float = _DEFAULT_LEARNING_RATE,
        hidden_layers: Sequence[int] = _DEFAULT_HIDDEN_LAYERS,
        standardize: bool = _DEFAULT_STANDARDIZE,
        verbose: bool = _DEFAULT_VERBOSE,
        seed: int = _DEFAULT_SEED,
        _train_backend: Literal["quadratic", "linear"] = _DEFAULT_TRAIN_BACKEND,
    ) -> None:
        if _train_backend == "quadratic":
            mi_formula = _backend_quadratic_memory.donsker_varadhan
        elif _train_backend == "linear":
            mi_formula = _backend_linear_memory.donsker_varadhan
        else:
            raise ValueError(f"Backend {_train_backend} not known.")

        hidden_layers = list(hidden_layers)

        super().__init__(
            mi_formula=mi_formula,
            mi_formula_name="Donsker-Varadhan",
            mi_formula_test=_backend_linear_memory.donsker_varadhan,
            critic_factory=_mlp_init(hidden_layers),
            critic_params=MLPParams(hidden_layers=hidden_layers),
            max_n_steps=max_n_steps,
            train_test_split=train_test_split,
            test_every_n_steps=test_every_n_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            standardize=standardize,
            verbose=verbose,
            seed=seed,
        )


__all__ = [
    "NeuralEstimatorParams",
    "NeuralEstimatorBase",
    "InfoNCEEstimator",
    "NWJEstimator",
    "DonskerVaradhanEstimator",
]
