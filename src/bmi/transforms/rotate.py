from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import expm
from numpy.typing import ArrayLike


class Spiral(eqx.Module):
    """Represents the "spiraling" function
      x -> R(x) x,
    where R(x) is a matrix given by a product `initial` @ `rotation(x)`.
    `initial` can be an arbitrary invertible matrix
    and `rotation(x)` is an SO(n) element given by
      exp(generator * ||x||^2),
    where `generator` is an element of the so(n) Lie algebra, i.e., a skew-symmetric matrix.

    Example:
        >>> a = np.array([[0, -1], [1, 0]])
        >>> spiral = Spiral(a, speed=np.pi/2)
        >>> x = np.array([1, 0])
        >>> spiral(x)
        DeviceArray([0., 1.])
    """

    initial: jnp.ndarray
    generator: jnp.ndarray

    def __init__(
        self, generator: ArrayLike, initial: Optional[ArrayLike] = None, speed: float = 1.0
    ) -> None:
        """

        Args:
            generator: a skew-symmetric matrix, an element of so(n) Lie algebra. Shape (n, n)
            initial: an (n, n) matrix used to left-multiply the spiral.
              Default (None) corresponds to the identity.
            speed: for convenience, the passed `generator` will be scaled up by `speed` constant,
              which (for a given `generator`) controls how quickly the spiral will wind
        """
        self.generator = jnp.asarray(generator * speed)

        if len(self.generator.shape) != 2 or self.generator.shape[0] != self.generator.shape[1]:
            raise ValueError(f"Generator has wrong shape {self.generator.shape}.")

        if initial is None:
            self.initial = jnp.eye(self.generator.shape[0])
        else:
            initial = np.asarray(initial)
            if self.generator.shape != initial.shape:
                raise ValueError(
                    f"Initial point has shape {initial.shape} while "
                    f"the generator has {self.generator.shape}."
                )
            self.initial = jnp.asarray(initial)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: point in the Euclidean space, shape (n,)

        Returns:
            transformation applied to `x`, shape (n,)
        """
        r = jnp.einsum("i, i", x, x)  # We have r = ||x||^2
        return self.initial @ expm(self.generator * r) @ x


def so_generator(n: int, i: int = 0, j: int = 1) -> np.ndarray:
    """The (i,j)-th canonical generator of the so(n) Lie algebra.

    As so(n) Lie algebra is the vector space of all n x n
    skew-symmetric matrices, we have a canonical basis
    such that its (i,j)th vector is a matrix A such that
          A[i, j] = 1, A[j, i] = -1, i < j
    and all the other entries are 0.

    Note that there exist n(n-1)/2 such matrices.

    Args:
        n: we use the Lie algebra so(n)
        i: index in range {0, 1, ..., j-1}
        j: index in range {i+1, i+2, ..., n-1}

    Returns:
        array (n, n)
    """
    assert n >= 2
    assert 0 <= i < j < n

    a = np.zeros((n, n))
    a[i, j] = 1
    a[j, i] = -1
    return a


def skew_symmetrize(a: np.ndarray) -> np.ndarray:
    """The skew-symmetric part of a given matrix `a`.

    Args:
        a: array, shape (n, n)

    Returns:
        skew-symmetric part of `a`, shape (n, n)
    """
    return 0.5 * (a - a.T)
