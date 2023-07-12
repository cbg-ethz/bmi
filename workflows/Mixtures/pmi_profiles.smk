# This workflow generates PMI profiles of several distributions
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow_probability.substrates import jax as tfp

import jax
import jax.numpy as jnp

import bmi.samplers._tfp as bmi_tfp
from bmi.samplers import SparseLVMParametrization

mpl.use("Agg")

dispersion = SparseLVMParametrization(dim_x=2, dim_y=3, n_interacting=1).covariance
normal = bmi_tfp.MultivariateNormalDistribution(dim_x=2, dim_y=3, covariance=dispersion)
student = bmi_tfp.MultivariateStudentDistribution(dim_x=2, dim_y=3, dispersion=dispersion, df=5)
transformed_normal = bmi_tfp.transform(dist=normal, x_transform=tfp.bijectors.Sigmoid(), y_transform=tfp.bijectors.Sigmoid())

DISTRIBUTIONS = {
    "Normal-2-3": normal,
    "Student-2-3": student,
    "Transformed-Normal-2-3": transformed_normal,
    "Transformed-Student2-3": bmi_tfp.transform(dist=student, x_transform=tfp.bijectors.Sigmoid(), y_transform=tfp.bijectors.Identity()),
    "Mixture": bmi_tfp.mixture(
        proportions=[0.1, 0.4, 0.5],
        components=[
            bmi_tfp.MultivariateNormalDistribution(dim_x=2, dim_y=3, mean=jnp.zeros(5), covariance=jnp.eye(5)),
            bmi_tfp.MultivariateNormalDistribution(dim_x=2, dim_y=3, mean=jnp.ones(5), covariance=jnp.eye(5)),
            bmi_tfp.MultivariateNormalDistribution(dim_x=2, dim_y=3, mean=-1 * jnp.ones(5), covariance=2 * jnp.eye(5)),
        ]
    ),
}

# === WORKDIR ===
workdir: "generated/mixtures/pmi_profiles/"

rule all:
    input: expand("pmi-profile-{distribution}.pdf", distribution=DISTRIBUTIONS.keys())

rule generate_profile:
    output: "pmi-profile-{distribution}.pdf"
    run:
        dist_name = wildcards.distribution
        dist = DISTRIBUTIONS[dist_name]
        
        key = jax.random.PRNGKey(0)

        pmis = bmi_tfp.pmi_profile(key=key, dist=dist, n=10_000)
        fig, ax = plt.subplots()
        ax.hist(pmis, bins=100, density=True)
        fig.suptitle(f"PMI profile of {dist_name}. Mean (MI): {pmis.mean():.2f}")
        fig.tight_layout()
        fig.savefig(str(output))
