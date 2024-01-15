# Fine distributions

In this tutorial we will take a closer look at the family of *fine distributions*, proposed in [The Mixtures and the Neural Critics](https://arxiv.org/abs/2310.10240) paper[@mixtures-neural-critics-2023].

A distribution $P_{XY}$ is fine, if it is possible to evaluate the densities $\log p_{XY}(x, y)$, $\log p_X(x)$, and $\log p_Y(y)$ for arbitrary points and to efficiently generate samples from $P_{XY}$.
In particular, one can evaluate [pointwise mutual information](https://en.wikipedia.org/wiki/Pointwise_mutual_information) 

$\mathrm{PMI}(x, y) = \log \frac{p_{XY}(x, y)}{p_X(x)p_Y(y)}$

and use a Monte Carlo approximation of the mutual information $I(X; Y) = \mathbb E_{(x, y)\sim P_{XY}}[\mathrm{PMI}(x, y)]$.

The fine distribution can therefore be implemented as a triple of [TensorFlow Probability on JAX](https://www.tensorflow.org/probability/examples/TensorFlow_Probability_on_JAX) distributions: the joint distribution $P_{XY}$ and marginals $P_X$ and $P_Y$.

For example, let's create a multivariate normal distribution:

```python
import jax.numpy as jnp
from bmi.samplers import fine

# Define a fine distribution
cov = jnp.asarray([[1., 0.8], [0.8, 1.]])
dist = fine.MultivariateNormalDistribution(dim_x=1, dim_y=1, covariance=cov)
```

To see how fine distributions can be used in the benchmark, see [this section](#connecting-fine-distributions-and-samplers).

## Basic operations supported by fine distributions

A fine distribution $P_{XY}$ can be used to sample from $P_{XY}$ or evaluate the pointwise mutual information $\mathrm{PMI}(x, y)$ at any point.
The distribution of $\mathrm{PMI}(x, y)$, where $(x, y)\sim P_{XY}$ is called the *PMI profile* and can be approximated via the histograms from the samples.
Similarly, sample-based Monte Carlo approximation can be used to estimate the mutual information $I(X; Y)$, which is the mean of the PMI profile.

```python
import jax.numpy as jnp
from jax import random

from bmi.samplers import fine

# Define a fine distribution:
cov = jnp.asarray([[1., 0.8], [0.8, 1.]])
dist = fine.MultivariateNormalDistribution(dim_x=1, dim_y=1, covariance=cov)

# Sample 100_000 points from the distribution:
key = random.PRNGKey(42)
n_samples = 100_000
X, Y = dist.sample(1000, key=key)

# Calculate the PMI, obtaining the sample from the PMI profile:
pmis = dist.pmi(X, Y)

# If one wants only the samples from the PMI profile,
# one can use a helper function.
# We use the same key to get the same output as in the previous example.
pmis = fine.pmi_profile(key=key, dist=dist, n=1000)

# Estimate mutual information from samples:
print(jnp.mean(pmis))  # 0.51192063

# One can also estimate the mutual information
# and the associated Monte Carlo Standard Error (MCSE)
# using the helper function:
mi_estimate, mi_mcse = fine.estimate_mutual_information(key=key, dist=dist, n=n_samples)
print(mi_estimate)  # 0.51192063
print(mi_mcse)  # 0.00252177
```

### Combining and transforming fine distributions

One can construct new fine distributions from the existing ones by two basic operations: transformation by a diffeomorphism and constructing a mixture.

#### Transformation by a diffeomorphism 

A fine distribution $P_{XY}$ can be transformed by a diffeomorphism of the form $f\times g$ to obtain a new fine distribution $P_{f(X)g(Y)}$.
Any diffeomorphism supported by [TensorFlow Probability on JAX](https://www.tensorflow.org/probability/examples/TensorFlow_Probability_on_JAX#bijectors) can be used.

```python
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from bmi.samplers import fine, SparseLVMParametrization

# Define a normal distribution
cov = SparseLVMParametrization(dim_x=2, dim_y=3, n_interacting=1).covariance
normal = fine.MultivariateNormalDistribution(dim_x=2, dim_y=3, covariance=cov)

# Use a TensorFlow Probability bijector to transform the distribution
transformed_normal = fine.transform(dist=normal, x_transform=tfp.bijectors.Sigmoid(), y_transform=tfp.bijectors.Sigmoid())
```

Note that samplers can be transformed by arbitrary continuous injective functions, not only diffeomorphisms, which preserve mutual information. However, this comes at the cost of losing the ability to compute pointwise mutual information. To wrap a fine distribution into a sampler, see [this section](#connecting-fine-distributions-and-samplers).

#### Constructing a mixture

If $P_{X_1Y_1}, \dotsc, P_{X_KY_K}$ are arbitrary fine distributions defined on the same space $\mathcal X\times \mathcal Y$, and $w_1, \dotsc, w_K$ are positive numbers such that $w_1 + \dotsc + w_K=1$, then the mixture

$$P_{XY} = \sum_{k=1}^K w_k P_{X_kY_K}$$

is also a fine distribution. Note that even if the component distributions do not encode any information ($I(X_k; Y_k) = 0$), the mixture $P_{XY}$ can have $I(X; Y) > 0$ and be as large as $\log K$.

```python
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from bmi.samplers import fine, SparseLVMParametrization

# Define a normal distribution
cov1 = SparseLVMParametrization(dim_x=2, dim_y=3, n_interacting=1).covariance
normal1 = fine.MultivariateNormalDistribution(dim_x=2, dim_y=3, covariance=cov1)

cov2 = SparseLVMParametrization(dim_x=2, dim_y=3, n_interacting=2).covariance
normal2 = fine.MultivariateNormalDistribution(dim_x=2, dim_y=3, covariance=cov2)

mixture = fine.mixture(proportions=[0.8, 0.2], components=[normal1, normal2])
```

## Discrete variables

When both variables are discrete (and over an alphabet which is not too large), the joint distribution $P_{XY}$ can be represented by a probability table, and the mutual information can be computed exactly and the [`dit`](https://github.com/dit/dit) package is an excellent choice for these applications.

However, discrete distributions $P_{XY}$ are also fine and Monte Carlo approximation can be used. Moreover, the fine distributions also encompass some cases where one of the variables is discrete and the other is continuous.

Consider a family of distributions $P_{X_z}\otimes P_{Y_z}$, where each variable $X_z$ is a continuous variable and $Y_z$ is a discrete variable. Although we have $I(X_z; Y_z) = 0$, by mixing the distributions $P_{X_z}\otimes P_{Y_z}$ with different $z$ we can obtain a distribution with non-zero mutual information, represented by a Bayesian network $X \leftarrow Z\rightarrow Y$.

Below we will show an example: 

```python
import jax.numpy as jnp

from bmi.samplers import fine

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

def construct_bernoulli(p: float, dtype=jnp.float64) -> tfd.Distribution:
    """Constructs a Bernoulli distribution, as
    TensorFlow Probability disallows products of continuous
    and discrete distributions."""
    return tfd.Independent(
            tfd.Bernoulli(probs=jnp.asarray([p], dtype=dtype), dtype=dtype),
            reinterpreted_batch_ndims=1,
    )


# Define the distributions X_k. Each of them is a continuous distribution on R^2.
x1 = fine.construct_multivariate_student_distribution(mean=-jnp.ones(2), dispersion=0.2 * jnp.eye(2), df=8)
x2 = fine.construct_multivariate_normal_distribution(mean=jnp.zeros(2), covariance=jnp.asarray([[1.0, 0.8], [0.8, 1.0]]))
x3 = fine.construct_multivariate_student_distribution(mean=jnp.ones(2), dispersion=0.2 * jnp.eye(2), df=5)

# Define the distributions for Y_k. Each of them is a discrete distribution over the alphabet {0, 1}.
y1 = construct_bernoulli(0.95)
y2 = construct_bernoulli(0.5)
y3 = construct_bernoulli(0.05)

# Define the product distributions P(X_k, Y_k).
components = [fine.ProductDistribution(dist_x, dist_y) for dist_x, dist_y in zip([x1, x2, x3], [y1, y2, y3])]

# Construct the distribution P(X, Y) distribution.
# As this is a fine distribution, one can construct the PMI profile, approximate MI, etc.
joint_distribution = fine.mixture(proportions=[0.25, 0.5, 0.25], components=components)
```

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

Consider a parametric family of fine distributions $\{P_\theta \mid \theta \in \mathcal T\}$, where $P_{\theta}$ is a model for a joint distribution $P(X, Y\mid \theta)$.
For example, if $X$ and $Y$ are assumed to be jointly multivariate norrmal, each $P_\theta$ will be a multivariate norrmal distribution with $\theta$ representing the mean vector and covariance matrix.
As each $P_\theta$ is fine, one can calculate the mutual information between the idealised variables $X$ and $Y$ for different parameters $\theta$.

Hence, once a data set $(x_1, y_1), \dotsc, (x_N, y_N)$ is available, one can construct the Bayesian posterior on the parameters $P(\theta \mid (x_1, y_1),\dotsc, (x_N, y_N))$ and use it to obtain the posterior distribution on $I(X; Y)$.

Hence, the workflow for model-based estimation of mutual information is as follows:

1. Construct a parametric family of distributions in a probabilistic programming language (e.g., 
[TensorFlow Probability on JAX](https://www.tensorflow.org/probability/examples/TensorFlow_Probability_on_JAX) or [NumPyro](https://num.pyro.ai/)).
  
2. Ensure that for each parameter $\theta$, you can construct the distribution $P_\theta$ (e.g., by implementing a simple wrapper).
3. Use the data set to construct the posterior distribution $P(\theta \mid (x_1, y_1),\dotsc, (x_N, y_N))$.
4. Ensure that the model is well-specified and does not underfit or overfit the data.
5. Wrap the samples from the posterior in the fine distributions and use Monte Carlo approximation to estimate the mutual information to obtain the samples from the mutual information posterior.

As an example using NumPyro to fit a mixture of multivariate normal distributions, see [`workflows/Mixtures/fitting_gmm.smk`](https://github.com/cbg-ethz/bmi/blob/main/workflows/Mixtures/fitting_gmm.smk).

## FAQ

### Where is the API?
The API is [here](api/fine-distributions.md).

### What is the difference between a fine distribution and a sampler?
Samplers are more general than fine distributions. [This section](#connecting-fine-distributions-and-samplers) explains how to wrap a fine distribution into a sampler.

### How should I choose the number of samples for the mutual information estimation?
If variance of the PMI profile is finite and equal to $V$, then the Monte Carlo Standard Error (MCSE) of the mutual information estimate is equal to $\sqrt{V/n}$.
We suggest to obtain an estimate for e.g., 10,000 samples (which should be fast) and observing whether the MCSE is small enough.
Additionally, we suggest to repeat sampling several times and observing whether the MCSE is stable: it may be possible that for some problems the variance of the PMI profile may be infinite and MCSE is not a valid uncertainty estimate.

\bibliography