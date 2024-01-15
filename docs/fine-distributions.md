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

One can construct new fine distributions from the existing ones by two basic operations: transformation by a diffeomorphism and constructing a mixture.

#### Transformation by a diffeomorphism 

!!! todo
    
    Add an example on how to use all operations.

#### Constructing a mixture


!!! todo
    
    Add an example on how to use all operations.

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

\bibliography