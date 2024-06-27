# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

# +
key = jax.random.PRNGKey(42)

initial_distribution = tfd.Categorical(probs=jnp.array([0.5, 0.3, 0.2]))

observation_distribution = tfd.Normal(jnp.asarray([0., 0.5, 1.0]), 1.0)

num_steps = 5

transition_probs =jnp.asarray([[0.0, 0.5, 0.5],
                               [1.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0],
                               ])
transition_distribution = tfd.Categorical(probs=transition_probs)



dist_x = tfd.MarkovChain(
    initial_state_prior=initial_distribution,
    transition_fn=lambda _, x: tfd.Categorical(probs=transition_probs[x, :]),
    num_steps=num_steps,
)

dist_y = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=num_steps)

def dist_xy_log_prob(xs, ys):
    # xs, ys = xys
    log_prob_xs = dist_x.log_prob(xs)

    ys_dists = observation_distribution[xs]
    log_prob_ys_given_xs = jnp.sum(
        observation_distribution[xs].log_prob(ys),
        axis=-1,
    )

    return log_prob_xs + log_prob_ys_given_xs

def dist_xy_sample(n, key):
    key1, key2 = jax.random.split(key)
    xs = dist_x.sample(n, key)

    ys_dists = observation_distribution[xs]
    ys = ys_dists.sample((), key2)

    return xs, ys
    
# dist_xy = tfd.Distribution(
#     sample_fn=,
#     log_prob_fn=,
#     (


# +
dist_xy = tfd.JointDistributionSequential([
    dist_x,
    lambda xs: tfd.Independent(observation_distribution[xs]),
])

xys = dist_xy.sample(5_000, key)
xs, ys = xys



# +
from bmi.samplers._tfp._core import JointDistribution, monte_carlo_mi_estimate


our_dist = JointDistribution(
    dist_x=dist_x,
    dist_y=dist_y,
    dist_joint=dist_xy,
    dim_x=num_steps,
    dim_y=num_steps,
    unwrap=False,
)
# -

mi, std_err = monte_carlo_mi_estimate(key + 3, our_dist, 10_000)
mi

std_err

import bmi

estimator = bmi.estimators.CCAMutualInformationEstimator()
estimator.estimate(xs, ys)

estimator = bmi.estimators.NWJEstimator()
estimator.estimate(xs, ys)



# +
estimator = bmi.estimators.KSGEnsembleFirstEstimator(neighborhoods=(5,))

es = 1e-4 * jax.random.normal(key+3, shape=xs.shape)
estimator.estimate(xs + es, ys)
# -

estimator = bmi.estimators.MINEEstimator()
estimator.estimate(xs, ys)

estimator = bmi.estimators.InfoNCEEstimator()
estimator.estimate(xs, ys)

xs, ys = dist_xy
_maybe.sample(3, key)
xys = dist_xy_maybe.sample(3, key)

# +
# xys = jnp.stack([xs, ys], axis=0)
# -

dist_xy_maybe.log_prob(xys)

xs, ys = dist_xy_sample(3, key)

ys.shape

# +
num_samples = 3

sample_x = dist_x.sample(num_samples, key)
sample_y = dist_y.sample(num_samples, key)

dist_xy_log_prob(sample_x, sample_y)
# -


