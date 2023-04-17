from typing import Any, Callable

import jax
import jax.numpy as jnp

# Any should be float, but this leads to cumbersome type errors with arrays
_RealFunction = Callable[[Any], Any]


def invert_mono(
    f: _RealFunction, x_min: float, x_max: float, delta: float = 1e-12
) -> _RealFunction:
    """Finds the inverse to a strictly increasing function `f`."""
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
            xm = (x0 + x1) / 2
            ok = (abs(x1 - x0) > delta) * (x1 > xm) * (xm > x0)
            return ok

        x0_fin, x1_fin = jax.lax.while_loop(g_cond, g_step, (x_min, x_max))
        return (x0_fin + x1_fin) / 2

    return g


def invert_cdf(
    cdf: _RealFunction, x_min: float = -10.0, x_max: float = +10.0, delta: float = 1e-12
) -> _RealFunction:
    """Finds an approximate inverse CDF."""

    icdf_appx = jax.jit(jnp.vectorize(invert_mono(cdf, x_min=x_min, x_max=x_max, delta=delta)))

    return icdf_appx
