"""Implementation of MINE:

M.I. Belghazi et al.,
MINE: Mutual Information Neural Estimation
https://arxiv.org/abs/1801.04062v5

The expression for the gradient
is given by Equation (12) in Section 3.2.
"""
from typing import Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pydantic
from jax.scipy.special import logsumexp
from numpy.typing import ArrayLike

import bmi.estimators.neural._estimators as _estimators
from bmi.estimators.neural._critics import MLP
from bmi.estimators.neural._training_log import TrainingLog
from bmi.estimators.neural._types import BatchedPoints, Critic
from bmi.interface import BaseModel, IMutualInformationPointEstimator
from bmi.utils import ProductSpace


def _mine_T_value_and_grad(
    f: Critic,
    xs: jnp.ndarray,
    ys: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    f_vmap = jax.vmap(f, in_axes=(0, 0))
    return jax.value_and_grad(f_vmap)(xs, ys)


def _mine_value(
    f: Critic,
    xs: jnp.ndarray,
    ys_paired: jnp.ndarray,
    ys_unpaired: jnp.ndarray,
):
    f_vmap = jax.vmap(f, in_axes=(0, 0))
    p_T = f_vmap(xs, ys_paired)
    u_T = f_vmap(xs, ys_unpaired)

    return jnp.mean(p_T) - logsumexp(u_T)


def _mine_value_neg_grad_log_denom(
    f: Critic,
    xs: jnp.ndarray,
    ys_paired: jnp.ndarray,
    ys_unpaired: jnp.ndarray,
    log_denom_prev: float,
    alpha: float,
):
    p_T, p_dT = _mine_T_value_and_grad(f, xs, ys_paired)
    u_T, u_dT = _mine_T_value_and_grad(f, xs, ys_unpaired)

    # compute denominator (with exponential smoothing)
    log_denom_batch = logsumexp(u_T)
    log_denom = jnp.logaddexp(
        jnp.log(alpha) + log_denom_prev,
        jnp.log(1.0 - alpha) + log_denom_batch,
    )

    # compute (negative) grad
    second_term = jnp.mean(u_dT * jnp.exp(u_T - log_denom))
    neg_grad = second_term - jnp.mean(p_dT)

    # compute value
    value = jnp.mean(p_T) - log_denom_batch

    return value, neg_grad, log_denom


def _sample_paired_unpaired(
    key: jax.random.PRNGKeyArray,
    xs: BatchedPoints,
    ys: BatchedPoints,
    batch_size: Optional[int],
) -> tuple[BatchedPoints, BatchedPoints, BatchedPoints]:
    assert len(xs) == len(ys), f"Length mismatch: {len(xs)} != {len(ys)}"
    key_paired, key_unpaired = jax.random.split(key)

    if batch_size is None:
        ys_unpaired = jax.random.shuffle(key_unpaired, ys)
        return xs, ys, ys_unpaired

    paired_indices = jax.random.choice(
        key_paired,
        len(xs),
        shape=(batch_size,),
        replace=False,
    )
    unpaired_indices = jax.random.choice(
        key_unpaired,
        len(xs),
        shape=(batch_size,),
        replace=False,
    )
    xs_paired = xs[paired_indices]
    ys_paired = ys[paired_indices]
    ys_unpaired = ys[unpaired_indices]

    return xs_paired, ys_paired, ys_unpaired


def mine_training(
    rng: jax.random.PRNGKeyArray,
    critic: eqx.Module,
    xs: BatchedPoints,
    ys: BatchedPoints,
    xs_test: Optional[BatchedPoints] = None,
    ys_test: Optional[BatchedPoints] = None,
    alpha: float = 0.9,
    batch_size: Optional[int] = 256,
    test_every_n_steps: int = 250,
    max_n_steps: int = 2_000,
    early_stopping: bool = True,
    learning_rate: float = 0.1,
    verbose: bool = False,
) -> TrainingLog:
    """Basic training loop for MINE.

    Args:
        rng: random key
        critic: critic to be trained
        xs: samples of X, shape (n_points, dim_x)
        ys: paired samples of Y, shape (n_points, dim_y)
        xs_test: samples of X used for computing test MI, shape (n_points_test, dim_x),
          if None will reuse xs
        ys_test: paired samples of Y used for computing test MI, shape (n_points_test, dim_y),
          if None will reuse ys
        alpha: parameter used in exponential smoothing of the gradient,
          in the open interval (0, 1). Values closer to 1 result in less smoothing
        batch_size: batch size
        test_every_n_steps: step intervals at which the training checkpoint should be saved
        max_n_steps: maximum number of steps
        early_stopping: whether training should stop early when test MI stops growing
        learning_rate: learning rate to be used
        verbose: print info during training
    """
    xs_test = xs_test if xs_test is not None else xs
    ys_test = ys_test if ys_test is not None else ys

    # initialize the optimizer
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(critic)

    # compile the training step
    @jax.jit
    def step(
        *,
        critic,
        opt_state,
        xs: BatchedPoints,
        ys_paired: BatchedPoints,
        ys_unpaired: BatchedPoints,
        log_denom_carry: float,
        alpha: float,
    ):
        value, neg_grad, log_denom_carry = _mine_value_neg_grad_log_denom(
            f=critic,
            xs=xs,
            ys_paired=ys_paired,
            ys_unpaired=ys_unpaired,
            log_denom_prev=log_denom_carry,
            alpha=alpha,
        )

        updates, opt_state = optimizer.update(neg_grad, opt_state, critic)
        critic = optax.apply_updates(critic, updates)
        return critic, opt_state, value, log_denom_carry

    # initialize log_denom_carry to be very small at the beginning,
    # so that the smoothing effect is negligible at the start
    log_denom_carry = -jnp.inf

    # main training loop
    training_log = TrainingLog(max_n_steps=max_n_steps, verbose=verbose)
    keys = jax.random.split(rng, max_n_steps)
    for n_step, key in enumerate(keys, start=1):
        key_sample, key_test = jax.random.split(key)
        # sample
        xs_batch, ys_batch_paired, ys_batch_unpaired = _sample_paired_unpaired(
            key_sample, xs=xs, ys=ys, batch_size=batch_size
        )

        # run step
        critic, opt_state, mi_train, log_denom_carry = step(
            critic=critic,
            opt_state=opt_state,
            xs=xs_batch,
            ys_paired=ys_batch_paired,
            ys_unpaired=ys_batch_unpaired,
            log_denom_carry=log_denom_carry,
            alpha=alpha,
        )

        # logging train
        training_log.log_train_mi(n_step, mi_train)

        # logging test
        if n_step % test_every_n_steps == 0:
            ys_test_unpaired = jax.random.shuffle(key_test, ys_test)
            mi_test = _mine_value(
                f=critic, xs=xs_test, ys_paired=ys_test, ys_unpaired=ys_test_unpaired
            )
            training_log.log_test_mi(n_step, mi_test)

        # early stop?
        if early_stopping and training_log.early_stop():
            break

    training_log.finish()

    if verbose:
        if early_stopping and not training_log.early_stop():
            print("WARNING: Early stopping enabled but max_n_steps reached.")

    return training_log


class MINEParams(BaseModel):
    batch_size: pydantic.PositiveInt
    max_n_steps: pydantic.PositiveInt
    train_test_split: Optional[float]
    test_every_n_steps: pydantic.PositiveInt
    learning_rate: pydantic.PositiveFloat
    smoothing_alpha: pydantic.confloat(gt=0, lt=1) = pydantic.Field(
        description="Alpha used for gradient smoothing. "
        "Values closer to 1 result in less smoothing."
    )
    standardize: bool
    seed: int
    hidden_layers: list[int]


class MINEEstimator(IMutualInformationPointEstimator):
    def __init__(
        self,
        batch_size: int = _estimators._DEFAULT_BATCH_SIZE,
        max_n_steps: int = _estimators._DEFAULT_N_STEPS,
        train_test_split: Optional[float] = _estimators._DEFAULT_TRAIN_TEST_SPLIT,
        test_every_n_steps: int = _estimators._DEFAULT_TEST_EVERY_N,
        learning_rate: float = _estimators._DEFAULT_LEARNING_RATE,
        hidden_layers: Sequence[int] = _estimators._DEFAULT_HIDDEN_LAYERS,
        smoothing_alpha: float = 0.9,
        standardize: bool = _estimators._DEFAULT_STANDARDIZE,
        verbose: bool = _estimators._DEFAULT_VERBOSE,
        seed: int = _estimators._DEFAULT_SEED,
    ) -> None:
        self._params = MINEParams(
            batch_size=batch_size,
            max_n_steps=max_n_steps,
            train_test_split=train_test_split,
            test_every_n_steps=test_every_n_steps,
            learning_rate=learning_rate,
            smoothing_alpha=smoothing_alpha,
            standardize=standardize,
            seed=seed,
            hidden_layers=list(hidden_layers),
        )
        self._verbose = verbose
        self._training_log: Optional[TrainingLog] = None

    def parameters(self) -> MINEParams:
        return self._params

    def _create_critic(self, dim_x: int, dim_y: int, key: jax.random.PRNGKeyArray) -> MLP:
        return MLP(dim_x=dim_x, dim_y=dim_y, key=key, hidden_layers=self._params.hidden_layers)

    def estimate(self, x: ArrayLike, y: ArrayLike) -> float:
        key = jax.random.PRNGKey(self._params.seed)
        key_init, key_split, key_fit = jax.random.split(key, 3)

        # standardize the data, note we do so before splitting into train/test
        space = ProductSpace(x, y, standardize=self._params.standardize)
        xs, ys = jnp.asarray(space.x), jnp.asarray(space.y)

        # split
        xs_train, xs_test, ys_train, ys_test = _estimators.train_test_split(
            xs, ys, train_size=self._params.train_test_split, key=key_split
        )

        # initialize critic
        critic = self._create_critic(dim_x=space.dim_x, dim_y=space.dim_y, key=key_init)

        training_log = mine_training(
            rng=key_fit,
            critic=critic,
            xs=xs_train,
            ys=ys_train,
            xs_test=xs_test,
            ys_test=ys_test,
            alpha=self._params.smoothing_alpha,
            batch_size=self._params.batch_size,
            test_every_n_steps=self._params.test_every_n_steps,
            max_n_steps=self._params.max_n_steps,
            learning_rate=self._params.learning_rate,
            verbose=self._verbose,
        )

        return training_log.final_mi
