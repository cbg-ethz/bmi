"""API of the neural estimators implemented in JAX."""

from typing import Any, Callable, Literal, Optional, Sequence

import jax
import jax.numpy as jnp
import pydantic
import sklearn.model_selection as msel
from numpy.typing import ArrayLike

from bmi.estimators.neural import _backend_linear, _backend_quadratic
from bmi.estimators.neural._interfaces import Critic, Point
from bmi.estimators.neural._nn import MLP, basic_fit, mi_divergence_check
from bmi.interface import BaseModel, IMutualInformationPointEstimator
from bmi.utils import ProductSpace


class NeuralEstimatorParams(BaseModel):
    mi_formula: str
    batch_size: pydantic.PositiveInt
    max_n_steps: pydantic.PositiveInt
    train_test_split: Optional[float]
    test_every_n_steps: Optional[int]
    learning_rate: pydantic.PositiveFloat
    seed: int
    standardize: bool
    critic_params: Optional[BaseModel]


class _NeuralEstimator(IMutualInformationPointEstimator):
    def __init__(
        self,
        mi_formula: Callable[[Critic, Point, Point], float],
        mi_formula_name: str,
        critic_factory: Callable[[Any, int, int], Critic],
        critic_params: Optional[BaseModel] = None,
        mi_formula_test: Optional[Callable[[Critic, Point, Point], float]] = None,
        batch_size: int = 256,
        max_n_steps: int = 2_000,
        train_test_split: Optional[float] = 0.5,
        test_every_n_steps: Optional[int] = 250,
        learning_rate: float = 0.1,
        seed: int = 42,
        verbose: bool = False,
        standardize: bool = True,
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
        self._standardize = standardize

        self._critic_factory = critic_factory

        self._testing = train_test_split is not None and test_every_n_steps is not None

        if not self._testing:
            train_test_split = None
            test_every_n_steps = None

        self._params = NeuralEstimatorParams(
            mi_formula=mi_formula_name,
            batch_size=batch_size,
            max_n_steps=max_n_steps,
            train_test_split=train_test_split,
            test_every_n_steps=test_every_n_steps,
            learning_rate=learning_rate,
            seed=seed,
            standardize=standardize,
            critic_params=critic_params,
        )

    def parameters(self) -> NeuralEstimatorParams:
        return self._params

    def estimate(self, x: ArrayLike, y: ArrayLike) -> float:
        key = jax.random.PRNGKey(self._params.seed)
        key_init, key_split, key_fit = jax.random.split(key, 3)

        # Standardize the data if the right option is specified
        # Note that we standardize before splitting into train/test split
        space = ProductSpace(x, y, standardize=self._standardize)
        xs, ys = jnp.asarray(space.x), jnp.asarray(space.y)

        if self._testing:
            xs_train, xs_test, ys_train, ys_test = msel.train_test_split(
                xs, ys, train_size=self._params.train_test_split
            )
        else:
            xs_train, ys_train = xs, ys
            xs_test, ys_test = None, None

        critic = self._critic_factory(key_init, xs.shape[-1], ys.shape[-1])

        train_history = basic_fit(
            rng=key_fit,
            critic=critic,
            mi_formula=self._mi_formula,
            xs=xs_train,
            ys=ys_train,
            mi_formula_test=self._mi_formula_test,
            xs_test=xs_test,
            ys_test=ys_test,
            test_every_n_steps=self._params.test_every_n_steps,
            batch_size=self._params.batch_size,
            max_n_steps=self._params.max_n_steps,
            learning_rate=self._params.learning_rate,
            verbose=self._verbose,
        )

        # check for problems with train loss
        mi_train = -jnp.array(train_history.loss_history)
        if mi_train.max() == mi_train[-1]:
            print(
                "WARNING! Train MI reached maximum during last step. "
                "This might mean the network has not fully converged."
            )

        if self._testing:
            # check for problems with test loss
            mi_test = -jnp.array(train_history.test_history)
            test_check = mi_divergence_check(mi_test)
            if test_check:
                print(
                    f"WARNING! Test MI reached {test_check[0]:.3f}, "
                    f"but then dropped to {test_check[1]:.3f}. This might "
                    "indicate overfitting."
                )

            elif len(mi_test) > 1 and mi_test[-1] > 1.01 * mi_test[-2]:
                print(
                    "WARNING! Test MI was still increasing when training stopped. "
                    "This might mean the network has not fully converged."
                )

        return train_history.final_mi


class MLPParams(BaseModel):
    hidden_layers: list[int]


def _mlp_init(hidden_layers: Sequence[int]) -> Callable[[Any, int, int], Critic]:
    def factory(key, dim_x: int, dim_y: int) -> Critic:
        return MLP(key=key, dim_x=dim_x, dim_y=dim_y, hidden_layers=hidden_layers)

    return factory


class InfoNCEEstimator(_NeuralEstimator):
    def __init__(
        self,
        batch_size: int = 256,
        max_n_steps: int = 2_000,
        train_test_split: Optional[float] = 0.5,
        test_every_n_steps: Optional[int] = 250,
        learning_rate: float = 0.1,
        hidden_layers: Sequence[int] = (5,),
        seed: int = 42,
        verbose: bool = False,
        _train_backend: Literal["quadratic", "linear"] = "quadratic",
    ) -> None:
        if _train_backend == "quadratic":
            mi_formula = _backend_quadratic.infonce
        elif _train_backend == "linear":
            mi_formula = _backend_linear.infonce
        else:
            raise ValueError(f"Backend {_train_backend} not known.")

        super().__init__(
            mi_formula=mi_formula,
            mi_formula_name="InfoNCE",
            critic_factory=_mlp_init(hidden_layers),
            critic_params=MLPParams(hidden_layers=hidden_layers),
            mi_formula_test=_backend_linear.infonce,
            max_n_steps=max_n_steps,
            train_test_split=train_test_split,
            test_every_n_steps=test_every_n_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            verbose=verbose,
        )


class NWJEstimator(_NeuralEstimator):
    def __init__(
        self,
        batch_size: int = 256,
        max_n_steps: int = 2_000,
        train_test_split: Optional[float] = 0.5,
        test_every_n_steps: Optional[int] = 250,
        learning_rate: float = 0.1,
        hidden_layers: Sequence[int] = (5,),
        seed: int = 42,
        verbose: bool = False,
        _train_backend: Literal["quadratic", "linear"] = "quadratic",
    ) -> None:
        if _train_backend == "quadratic":
            mi_formula = _backend_quadratic.nwj
        elif _train_backend == "linear":
            mi_formula = _backend_linear.nwj
        else:
            raise ValueError(f"Backend {_train_backend} not known.")

        super().__init__(
            mi_formula=mi_formula,
            mi_formula_name="NWJ",
            critic_factory=_mlp_init(hidden_layers),
            critic_params=MLPParams(hidden_layers=hidden_layers),
            mi_formula_test=_backend_linear.nwj,
            max_n_steps=max_n_steps,
            train_test_split=train_test_split,
            test_every_n_steps=test_every_n_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            verbose=verbose,
        )


class DonskerVaradhanEstimator(_NeuralEstimator):
    def __init__(
        self,
        batch_size: int = 256,
        max_n_steps: int = 2_000,
        train_test_split: Optional[float] = 0.5,
        test_every_n_steps: Optional[int] = 250,
        learning_rate: float = 0.1,
        hidden_layers: Sequence[int] = (5,),
        seed: int = 42,
        verbose: bool = False,
        _train_backend: Literal["quadratic", "linear"] = "quadratic",
    ) -> None:
        if _train_backend == "quadratic":
            mi_formula = _backend_quadratic.donsker_varadhan
        elif _train_backend == "linear":
            mi_formula = _backend_linear.donsker_varadhan
        else:
            raise ValueError(f"Backend {_train_backend} not known.")

        super().__init__(
            mi_formula=mi_formula,
            mi_formula_name="Donsker-Varadhan",
            mi_formula_test=_backend_linear.donsker_varadhan,
            critic_factory=_mlp_init(hidden_layers),
            critic_params=MLPParams(hidden_layers=hidden_layers),
            max_n_steps=max_n_steps,
            train_test_split=train_test_split,
            test_every_n_steps=test_every_n_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            verbose=verbose,
        )


__all__ = [
    "NeuralEstimatorParams",
    "InfoNCEEstimator",
    "NWJEstimator",
    "DonskerVaradhanEstimator",
]
