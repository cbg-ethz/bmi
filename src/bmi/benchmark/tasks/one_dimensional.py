from typing import Callable, Iterable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erf

import bmi.samplers.api as samplers
from bmi.benchmark.core import Task, generate_task

# TODO(Frederic, Pawel): Otherwise the bimodal Gaussian mixture cannot be created.
#  We need to invert a strictly increasing function. As we do it via binsearch,
#  we need to have high-enough precision
jax.config.update("jax_enable_x64", True)


N_SAMPLES: int = 5000
DEFAULT_CORRELATION: float = (
    0.75  # For many tasks we will use Gaussian sampler with this correlation
)

_RealFunction = Callable[[float], float]

# *** Additive uniform tasks ***


ADDITIVE_UNIFORM_EPSILONS: Iterable[float] = [0.75, 0.1]


def generate_additive_uniform_task(
    epsilon: float, n_seeds: int, n_samples: int = N_SAMPLES
) -> Task:
    return generate_task(
        sampler=samplers.AdditiveUniformSampler(epsilon=epsilon),
        task_id=f"one-dimensional-uniform-additive-{epsilon:.4f}",
        n_samples=n_samples,
        task_params={"epsilon": epsilon},
        seeds=range(n_seeds),
    )


def _generate_additive_uniform_tasks(n_seeds: int, n_samples: int) -> Iterable[Task]:
    for epsilon in ADDITIVE_UNIFORM_EPSILONS:
        yield generate_additive_uniform_task(epsilon=epsilon, n_seeds=n_seeds, n_samples=n_samples)


# *** Bivariate Student-t task ***
def get_student_sampler(
    df: int = 5, strength: float = DEFAULT_CORRELATION
) -> samplers.SplitStudentT:
    return samplers.SplitStudentT(
        dim_x=1,
        dim_y=1,
        df=df,
        dispersion=np.asarray(
            [
                [1.0, strength],
                [strength, 1.0],
            ]
        ),
    )


def generate_student_task(
    *, n_samples: int, n_seeds: int, df: int = 5, strength: float = DEFAULT_CORRELATION
) -> Task:
    if strength >= 1 or strength <= -1:
        raise ValueError(f"Strength must be between (-1, 1), was {strength}.")
    sampler = get_student_sampler(df=df, strength=strength)
    task_id = f"one-dimensional-student-{df}_df-{strength:.4f}_strength"
    task_params = dict(degrees_of_freedom=df, strength=strength)
    return generate_task(
        sampler=sampler,
        n_samples=n_samples,
        seeds=range(n_seeds),
        task_id=task_id,
        task_params=task_params,
    )


# *** Uniform marginal distributions task ***


def normal_cdf(x: float) -> float:
    """The CDF of the standard normal distribution."""
    return 0.5 * (1 + erf(x / 2**0.5))


def get_marginal_uniform_sampler(
    gaussian_correlation: float = DEFAULT_CORRELATION,
) -> samplers.TransformedSampler:
    gaussian_sampler = samplers.BivariateNormalSampler(correlation=gaussian_correlation)
    return samplers.TransformedSampler(
        base_sampler=gaussian_sampler,
        transform_x=normal_cdf,
        transform_y=normal_cdf,
    )


def generate_marginal_uniform_task(
    *,
    n_samples: int,
    n_seeds: int,
    gaussian_correlation: float = DEFAULT_CORRELATION,
) -> Task:
    sampler = get_marginal_uniform_sampler(gaussian_correlation=gaussian_correlation)
    return generate_task(
        sampler=sampler,
        n_samples=n_samples,
        seeds=range(n_seeds),
        task_id=f"one-dimensional-uniform-margins-{gaussian_correlation:.4f}_corr",
        task_params=dict(gaussian_correlation=gaussian_correlation),
    )


# *** Half-cube (longer-tails) task ***


def half_cube(x: float) -> float:
    return x * jnp.abs(x) ** 0.5


def get_half_cube_sampler(
    gaussian_correlation: float = DEFAULT_CORRELATION,
) -> samplers.TransformedSampler:
    sampler = samplers.BivariateNormalSampler(correlation=gaussian_correlation)
    return samplers.TransformedSampler(
        base_sampler=sampler,
        transform_x=half_cube,
        transform_y=half_cube,
    )


def generate_half_cube_task(
    *, n_samples: int, n_seeds: int, gaussian_correlation: float = DEFAULT_CORRELATION
) -> Task:
    sampler = get_half_cube_sampler(gaussian_correlation=gaussian_correlation)
    return generate_task(
        sampler=sampler,
        n_samples=n_samples,
        seeds=range(n_seeds),
        task_id=f"one-dimensional-half-cube-{gaussian_correlation:.4f}_corr",
        task_params=dict(gaussian_correlation=gaussian_correlation),
    )


# *** Bimodal Gaussian mixture task ***


def inverse_mono(f: _RealFunction, x_min: float, x_max: float, delta: float = 1e-8):
    """Finds the inverse to a strictly increasing function `f`.

    Note:
        This function is somewhat experimental
    """
    x_min, x_max = float(x_min), float(x_max)

    def g(y):
        def g_step(xs):
            x0, x1 = xs
            xm = (x0 + x1) / 2
            return jax.lax.cond(
                f(xm) < y,
                lambda: (xm, x1),
                lambda: (x0, xm),
            )

        def g_cond(xs):
            x0, x1 = xs
            ok = abs(x1 - x0) > delta
            return ok

        # TODO(Frederic, Pawel): This while loop may not converge
        #   e.g., due to the numerical inaccuracies
        x0_fin, x1_fin = jax.lax.while_loop(g_cond, g_step, (x_min, x_max))
        return (x0_fin + x1_fin) / 2

    return g


def inverse_cdf(
    cdf: _RealFunction, x_min: float = -10, x_max: float = 10, delta: float = 1e-8
) -> _RealFunction:
    """Finds an approximate inverse CDF.

    Note:
        This function is somewhat experimental as it is based on the experimental `inverse_mono`.
    """
    _icdf_appx = jax.jit(inverse_mono(cdf, x_min=x_min, x_max=x_max, delta=delta))

    def icdf_appx(xs):
        return jnp.array([_icdf_appx(x[..., 0]) for x in xs]).reshape(-1, 1)

    return icdf_appx


icdf_x = inverse_cdf(lambda x: 0.3 * normal_cdf(x + 0) + 0.7 * normal_cdf(x - 5))
icdf_y = inverse_cdf(lambda x: 0.5 * normal_cdf(x + 1) + 0.5 * normal_cdf(x - 3))


def get_bimodal_sampler(
    gaussian_correlation: float = DEFAULT_CORRELATION,
) -> samplers.TransformedSampler:
    """
    Note:
        This sampler is somewhat experimental.
    """
    base_sampler = get_marginal_uniform_sampler(gaussian_correlation=gaussian_correlation)
    return samplers.TransformedSampler(
        base_sampler=base_sampler,
        transform_x=icdf_x,
        transform_y=icdf_y,
        vectorise=False,
    )


def get_bimodal_task(
    *, n_seeds: int, n_samples: int, gaussian_correlation: float = DEFAULT_CORRELATION
) -> Task:
    return generate_task(
        sampler=get_bimodal_sampler(gaussian_correlation=gaussian_correlation),
        n_samples=n_samples,
        seeds=range(n_seeds),
        task_id=f"one-dimensional-bimodal-{gaussian_correlation:.4f}_corr",
        task_params=dict(gaussian_correlation=gaussian_correlation),
    )


# *** Gaussian distribution transformed via wiggly diffeomorphisms ***


def default_wiggly_x(x: float) -> float:
    return x + 0.4 * jnp.sin(1.0 * x) + 0.2 * jnp.sin(1.7 * x + 1) + 0.03 * jnp.sin(3.3 * x - 2.5)


def default_wiggly_y(x: float) -> float:
    return (
        x - 0.4 * jnp.sin(0.4 * x) + 0.17 * jnp.sin(1.3 * x + 3.5) + 0.02 * jnp.sin(4.3 * x - 2.5)
    )


def get_wiggly_sampler(
    *,
    gaussian_correlation: float = DEFAULT_CORRELATION,
    wiggly_x: Optional[_RealFunction] = None,
    wiggly_y: Optional[_RealFunction] = None,
) -> samplers.TransformedSampler:
    wiggly_x = wiggly_x if wiggly_x is not None else default_wiggly_x
    wiggly_y = wiggly_y if wiggly_y is not None else default_wiggly_y

    base_sampler = samplers.BivariateNormalSampler(correlation=gaussian_correlation)
    return samplers.TransformedSampler(
        base_sampler=base_sampler,
        transform_x=wiggly_x,
        transform_y=wiggly_y,
    )


def generate_wiggly_task(
    *, gaussian_correlation: float = DEFAULT_CORRELATION, n_samples: int, n_seeds: int
) -> Task:
    return generate_task(
        sampler=get_wiggly_sampler(gaussian_correlation=gaussian_correlation),
        n_samples=n_samples,
        seeds=range(n_seeds),
        task_id=f"one-dimensional-wiggly-{gaussian_correlation:.4f}_corr",
        task_params=dict(gaussian_correlation=gaussian_correlation),
    )


# *** The main function generating the task suite ***


def generate_tasks(*, n_seeds: int = 10, n_samples: int = N_SAMPLES) -> Iterable[Task]:
    # Additive uniform tasks
    yield from _generate_additive_uniform_tasks(n_samples=n_samples, n_seeds=n_seeds)
    # Student-t task
    yield generate_student_task(n_samples=n_samples, n_seeds=n_seeds)
    # Uniform margins task
    yield generate_marginal_uniform_task(n_samples=n_samples, n_seeds=n_seeds)
    # Longer-tails (half-cube transform) tasks
    yield generate_half_cube_task(n_samples=n_samples, n_seeds=n_seeds)
    # Bimodal margins (Gaussian mixtures)
    yield get_bimodal_task(n_seeds=n_seeds, n_samples=n_samples)
    # The wiggly task
    yield generate_wiggly_task(n_samples=n_samples, n_seeds=n_seeds)
