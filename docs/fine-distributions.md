# Fine distributions

In this tutorial we will take a closer look at the family of *fine distributions*, proposed in [The Mixtures and the Neural Critics](https://arxiv.org/abs/2310.10240) paper[@mixtures-neural-critics-2023].

A distribution $P_{XY}$ is fine, if it is possible to evaluate the densities $\log p_{XY}(x, y)$, $\log p_X(x)$, and $\log p_Y(y)$ for arbitrary points and to efficiently generate samples from $P_{XY}$.
In particular, one can evaluate [pointwise mutual information](https://en.wikipedia.org/wiki/Pointwise_mutual_information) $\mathrm{PMI}(x, y) = \log \frac{p_{XY}(x, y)}{p_X(x)p_Y(y)}$ and use a Monte Carlo approximation of the mutual information $I(X; Y) = \mathbb E_{(x, y)\sim P_{XY}}[\mathrm{PMI}(x, y)]$.

The fine distribution can therefore be implemented as a triple of [TensorFlow Probability on JAX](https://www.tensorflow.org/probability/examples/TensorFlow_Probability_on_JAX) distributions: the joint distribution $P_{XY}$ and marginals $P_X$ and $P_Y$.

For example, let's create a multivariate normal distribution:

```python
import jax.numpy as jnp
from bmi.samplers import fine

# Define a fine distribution
r = 0.8
cov = jnp.asarray([[1., r], [r, 1.]])
dist = fine.MultivariateNormalDistribution(dim_x=1, dim_y=1, covariance=cov)

# Evaluate the PMI at a specified point
print(dist.pmi(2.0, 3.5))  # 1.6219368
```

The distribution can be constructed.
To see how a fine distribution can be used in the benchmark see [this section](#connecting-fine-distributions-and-samplers).

## Basic operations supported by fine distributions

!!! todo

    Explain how to estimate MI and how to use PMI profiles.

### Combining and transforming fine distributions

One can construct new fine distributions from existing ones by three basic operations:

  - concatenation,
  - transformation by a diffeomorphism, and
  - constructing a mixture.

!!! todo
    
    Add an example on how to use all operations.

## Discrete variables

!!! todo

    Add an example on how to use discrete variables.

## Connecting fine distributions and samplers 
One of the main abstractions of the proposed package is the [`ISampler`](api/interfaces.md#bmi.interface.ISampler) class, which holds a joint probability distribution with known mutual information. As such, samplers can be [transformed with arbitrary continuous injective functions](api/samplers.md#bmi.samplers.TransformedSampler), [combined with other samplers](api/samplers.md#bmi.samplers.IndependentConcatenationSampler) or used to [define named benchmark tasks](api/tasks.md).

Fine distributions can be used to create samplers using the [`FineSampler`](api/fine-distributions.md#bmi.samplers._tfp.FineSampler) wrapper. For example, let's create a sampler for a bivariate normal distribution:

```python
import jax.numpy as jnp
from bmi.samplers import fine

# Define a fine distribution
cov = jnp.asarray([[1.0, 0.8], [0.8, 1.0]])
dist = fine.MultivariateNormalDistribution(dim_x=1, dim_y=1, covariance=cov)

# Wrap the distribution into a sampler.
# We will use 10 000 samples to estimate ground-truth mutual information.
# Generally, the more samples, the better.
sampler = fine.FineSampler(dist, mi_estimate_sample=10_000, mi_estimate_seed=42)

print(sampler.mutual_information())  # 0.5178049
# Created sampler can be used as usual (transformed, used to create tasks, combined, etc.)
```

## Estimating mutual information with fine distributions

!!! todo

    Add an example on how to use NumPyro to construct Bayesian samplers.

## FAQ

### Where is the API?
The API is [here](api/fine-distributions.md).

\bibliography