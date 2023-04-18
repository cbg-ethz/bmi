from typing import Callable

import jax.numpy as jnp

# A single point, i.e., a vector (dim_x,)
Point = jnp.ndarray

# A collection of points, i.e., a vector (batch_size, dim_x)
BatchedPoints = jnp.ndarray

# Critic is a real-valued (NOT (0,1)-valued) function taking a pair of points.
# We vmap over it inside the code, so it should assume arrays of shape (dim_x,) and (dim_y,).
Critic = Callable[[Point, Point], float]

__all__ = ["BatchedPoints", "Point", "Critic"]
