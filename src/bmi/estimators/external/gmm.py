try:
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
except ImportError:
    numpyro, dist, MCMC, NUTS = [None] * 4

from typing import Optional

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike

from bmi.interface import BaseModel, IMutualInformationPointEstimator
from bmi.samplers import fine
from bmi.utils import ProductSpace


def model(
    data, K: int = 10, alpha: Optional[float] = None, jitter: float = 1e-3, lkj_conc: float = 1.0
) -> None:
    """Builds a Gaussian mixture model.

    Args:
        data: shape (n_points, n_dim)
        K: number of components
        alpha: concentration parameter for the Dirichlet prior
        jitter: jitter for the covariance matrix to assure that it is positive definite
        lkj_conc: concentration parameter for the LKJ prior. Use 1.0 for uniform prior.

    Note:
        To use sparse Dirichlet prior we advise setting alpha = 1/K.

    Sample attributes:
        "pi": mixing proportions, shape (K,)
        "mu": component means, shape (K, n_dim)
        "cov": component covariances, shape (K, n_dim, n_dim)
    """
    alpha = alpha or 1.0 / K

    n_points, n_dim = data.shape

    # Prior for mixing proportions
    pi = numpyro.sample("pi", dist.Dirichlet(concentration=alpha * jnp.ones(K)))

    with numpyro.plate("components", K):
        # Prior for component means
        mu = numpyro.sample(
            "mu", dist.MultivariateNormal(loc=jnp.zeros(n_dim), scale_tril=3 * jnp.eye(n_dim))
        )

        # Prior for component covariances using LKJ and HalfCauchy

        with numpyro.plate("dimensions", n_dim):
            tau_ = numpyro.sample("tau", dist.HalfCauchy(1.0))

        tau = tau_.T  # (components, dimensions)

        # (components, dimensions, dimensions)
        L_omega = numpyro.sample("L_omega", dist.LKJCholesky(n_dim, concentration=lkj_conc))
        L_scaled = tau[:, :, None] * L_omega + jitter * jnp.eye(n_dim)[None, :, :]

        cov_matrix = L_scaled @ jnp.transpose(L_scaled, (0, 2, 1))
        numpyro.deterministic("cov", cov_matrix)

    # Likelihood
    numpyro.sample(
        "obs",
        dist.MixtureSameFamily(
            mixing_distribution=dist.Categorical(probs=pi),
            component_distribution=dist.MultivariateNormal(mu, scale_tril=L_scaled),
        ),
        obs=data,
    )


def sample_into_fine_distribution(
    means: jnp.ndarray,
    covariances: jnp.ndarray,
    proportions: jnp.ndarray,
    dim_x: int,
    dim_y: int,
) -> fine.JointDistribution:
    """Builds a fine distribution from a Gaussian mixture model parameters."""
    # Check if the dimensions are right
    n_components = proportions.shape[0]
    n_dims = dim_x + dim_y
    assert means.shape == (n_components, n_dims)
    assert covariances.shape == (n_components, n_dims, n_dims)

    # Build components
    components = [
        fine.MultivariateNormalDistribution(
            dim_x=dim_x,
            dim_y=dim_y,
            mean=mean,
            covariance=cov,
        )
        for mean, cov in zip(means, covariances)
    ]

    # Build a mixture model
    return fine.mixture(proportions=proportions, components=components)


class GMMEstimatorParams(BaseModel):
    """
    Attrs:
        standardize: whether to standardize the data before fitting
        n_components: number of GMM components, note that some of them
            may be left empty
        alpha: sparsity parameter of the GMM model
        mcmc_num_warmup: number of warmup steps during the MCMC sampling
        mcmc_num_samples: number of MCMC samples to take
        n_thinned_samples: number of MCMC samples used at MI estimation,
            lower values can speed up MI estimation at the cost of higher variance
        mi_estimate_num_samples: number of Monte Carlo samples to estimate MI
            for every single MCMC-obtained distribution. Lower values can speed up
            MI estimation at the cost of higher variance
    """

    standardize: bool
    n_components: int
    alpha: Optional[float]
    mcmc_num_warmup: int
    mcmc_num_samples: int
    mi_estimate_num_samples: int
    n_thinned_samples: int


class GMMEstimator(IMutualInformationPointEstimator):
    def __init__(
        self,
        key: int = 0,
        n_components: int = 10,
        alpha: Optional[float] = None,
        standardize: bool = True,
        mcmc_num_warmup: int = 500,
        mcmc_num_samples: int = 500,
        mi_estimate_num_samples: int = 1_000,
        n_thinned_samples: Optional[int] = None,
    ) -> None:
        """
        Args:
            n_bins_x: number of bins per each X dimension
            n_bins_y: number of bins per each Y dimension. Leave to None to use `n_bins_x`
            standardize: whether to standardize the data set
        """
        self._params = GMMEstimatorParams(
            standardize=standardize,
            n_components=n_components,
            alpha=alpha,
            mcmc_num_warmup=mcmc_num_warmup,
            mcmc_num_samples=mcmc_num_samples,
            mi_estimate_num_samples=mi_estimate_num_samples,
            n_thinned_samples=mcmc_num_samples if n_thinned_samples is None else n_thinned_samples,
        )
        # TODO(Pawel): Allow for JAX key
        self.key = jax.random.PRNGKey(key)

        self._dim_x: int = -1
        self._dim_y: int = -1
        self._mcmc = None

    def parameters(self) -> GMMEstimatorParams:
        return self._params

    def run_mcmc(self, x: ArrayLike, y: ArrayLike):
        space = ProductSpace(x=x, y=y, standardize=self._params.standardize)

        self.key, subkey = jax.random.split(self.key)

        if NUTS is None or MCMC is None:
            raise ImportError("You need to install the NumPyro package to do MCMC sampling.")

        nuts_kernel = NUTS(model)
        mcmc = MCMC(
            nuts_kernel,
            num_samples=self._params.mcmc_num_samples,
            num_warmup=self._params.mcmc_num_warmup,
        )
        mcmc.run(subkey, data=space.xy, K=self._params.n_components)

        self._mcmc = mcmc
        self._dim_x = space.dim_x
        self._dim_y = space.dim_y

    def get_fine_distribution(self, idx: int) -> fine.JointDistribution:
        if self._mcmc is None:
            raise ValueError("You need to run MCMC first. See the `run_mcmc` method.")

        samples = self._mcmc.get_samples()
        return sample_into_fine_distribution(
            means=samples["mu"][idx],
            covariances=samples["cov"][idx],
            proportions=samples["pi"][idx],
            dim_x=self._dim_x,
            dim_y=self._dim_y,
        )

    def get_sample_mi(self, idx: int, mc_samples: Optional[int] = None, key=None) -> float:
        if mc_samples is None:
            mc_samples = self._params.mi_estimate_num_samples
        if key is None:
            self.key, key = jax.random.split(self.key)

        distribution = self.get_fine_distribution(idx)
        mi, _ = fine.monte_carlo_mi_estimate(key=key, dist=distribution, n=mc_samples)
        return mi

    def get_posterior_mi(
        self, x: ArrayLike, y: ArrayLike, n_thinned_samples: Optional[int] = None
    ) -> jax.Array:
        self.run_mcmc(x, y)

        if n_thinned_samples is None:
            n_thinned_samples = self._params.n_thinned_samples

        self.key, subkey = jax.random.split(self.key)
        indices = jax.random.choice(
            subkey, self._params.mcmc_num_samples, shape=(n_thinned_samples,), replace=False
        )

        mis = [self.get_sample_mi(idx) for idx in indices]
        return jnp.asarray(mis)

    def estimate(
        self, x: ArrayLike, y: ArrayLike, n_thinned_samples: Optional[int] = None
    ) -> float:
        """MI estimate."""
        return jnp.mean(self.get_posterior_mi(x=x, y=y, n_thinned_samples=n_thinned_samples))
