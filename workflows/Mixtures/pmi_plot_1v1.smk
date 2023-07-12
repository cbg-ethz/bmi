# This workflow plots PMI for (1+1)-dimensional distributions
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tensorflow_probability.substrates import jax as tfp

import jax
import jax.numpy as jnp

import bmi.samplers._tfp as bmi_tfp
from bmi.samplers import SparseLVMParametrization

mpl.use("Agg")

dispersion_05 = jnp.asarray([[1.0, 0.5], [0.5, 1.0]])
dispersion_09 = jnp.asarray([[1.0, 0.9], [0.9, 1.0]])

normal_05 = bmi_tfp.MultivariateNormalDistribution(dim_x=1, dim_y=1, covariance=dispersion_05)

DISTRIBUTIONS = {
    "Normal-0.5": normal_05,
    "Normal-0.9": bmi_tfp.MultivariateNormalDistribution(dim_x=1, dim_y=1, covariance=dispersion_09),
    # Student distributions
    "Student-0.5-3": bmi_tfp.MultivariateStudentDistribution(dim_x=1, dim_y=1, dispersion=dispersion_05, df=3),
    "Student-0.5-5": bmi_tfp.MultivariateStudentDistribution(dim_x=1, dim_y=1, dispersion=dispersion_05, df=5),
    # Transformed normal
    "Transformed-Normal-0.5": bmi_tfp.transform(dist=normal_05, x_transform=tfp.bijectors.Shift(3.0), y_transform=tfp.bijectors.Identity()),
    
    # Mixture distribution
    "Mixture": bmi_tfp.mixture(
        proportions=[0.25, 0.25, 0.25, 0.25],
        components=[
            bmi_tfp.MultivariateNormalDistribution(dim_x=1, dim_y=1, mean=jnp.asarray([-1., 1.]), covariance=2 * jnp.eye(2)),
            bmi_tfp.MultivariateNormalDistribution(dim_x=1, dim_y=1, mean=jnp.asarray([1., 1.]), covariance=2 * jnp.eye(2)),
            bmi_tfp.MultivariateNormalDistribution(dim_x=1, dim_y=1, mean=jnp.asarray([1., -1.]), covariance=2 * jnp.eye(2)),
            bmi_tfp.MultivariateNormalDistribution(dim_x=1, dim_y=1, mean=jnp.asarray([-1., -1.]), covariance=2 * jnp.eye(2)),
        ]
    ),
}

# === WORKDIR ===
workdir: "generated/mixtures/pmi_plots-1v1/"

rule all:
    input: expand("pmi-plot-{distribution}.pdf", distribution=DISTRIBUTIONS.keys())

rule plot_pmi:
    output: "pmi-plot-{distribution}.pdf"
    run:
        dist_name = wildcards.distribution
        dist = DISTRIBUTIONS[dist_name]

        x = np.linspace(-10, 10, 100)  # change resolution if necessary
        y = np.linspace(-10, 10, 100)  # change resolution if necessary
        X, Y = np.meshgrid(x, y)

        Z = dist.pmi(X.reshape(-1, 1), Y.reshape(-1, 1)).reshape(X.shape)

        key = jax.random.PRNGKey(0)
        fig, ax = plt.subplots()
        ax.imshow(Z, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='Purples')
        ax.set_title(f'PMI for {dist_name}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        fig.tight_layout()
        fig.savefig(str(output))


