import jax
import jax.numpy as jnp

import matplotlib
import matplotlib.pyplot as plt

import bmi
from bmi.plot_utils.subplots_from_axsize import subplots_from_axsize

import numpy as np
from scipy import stats


# === WORKDIR ===
workdir: "generated/figure_visualise_pointwise_mi/"

matplotlib.use("agg")


def student_logpdfratio(x, y, df: int, shape_matrix):
    joint = stats.multivariate_t(
        loc=np.zeros(2),
        shape=shape_matrix,
        df=df,
    ).pdf(np.hstack([x, y]))

    individual_x = stats.multivariate_t(
        loc=0.0,
        shape=shape_matrix[0, 0],
        df=df
    ).pdf(x)
    individual_y = stats.multivariate_t(
        loc=0.0,
        shape=shape_matrix[1, 1],
        df=df
    ).pdf(y)
    
    return np.log(joint) - np.log(individual_x) - np.log(individual_y)


def gaussian_logpdfratio(x, y, shape_matrix):
    joint = stats.multivariate_normal(
        np.zeros(2),
        shape_matrix,
    ).pdf(np.hstack([x, y]))

    individual_x = stats.multivariate_normal(
        0.0,
        shape_matrix[0, 0],
    ).pdf(x)
    individual_y = stats.multivariate_normal(
        0.0,
        shape_matrix[1, 1],
    ).pdf(y)
    
    return np.log(joint) - np.log(individual_x) - np.log(individual_y)

RADIUS = 5

def get_data(task, logratio_true, radius: float = RADIUS):
    estimator = bmi.estimators.InfoNCEEstimator()
    samples_x, samples_y = task.sample(10_000, 42)
    estimate = estimator.estimate(samples_x, samples_y)

    print(f"True MI: {task.mutual_information:.2f}. Estimate: {estimate:.2f}")
    
    critic = estimator.trained_critic

    xs = jnp.linspace(-radius, radius, 71)
    ys = jnp.linspace(-radius, radius, 71)

    mxs, mys = jnp.meshgrid(xs, ys)

    pmi_critic = jax.vmap(jax.vmap(critic))(mxs[..., None], mys[..., None])
    pmi_true = np.vectorize(logratio_true)(mxs[..., None], mys[..., None])[..., 0]
    
    return {
        "samples_x": samples_x[:1000],
        "samples_y": samples_y[:1000],
        "pmi_true": pmi_true,
        "pmi_critic": pmi_critic,
    }
    
    # ax = axs[0]

    # ax.scatter(samples_x[:1000], samples_y[:1000], s=1, color='k', alpha=0.2, rasterized=True)
    # ax

    # ax.set_title("Pointwise MI")
    # ax.set_xlabel("$X$")
    # ax.set_ylabel("$Y$")


    # ax = axs[1]

    # ax.scatter(samples_x[:1000], samples_y[:1000], s=1, color='k', alpha=0.2, rasterized=True)
    # ax.imshow(pmi_critic, extent=[-radius, radius, radius, -radius], cmap="Blues")


rule all:
    input: ["figures/plot_pmi.pdf"]

rule plot:
    output: "figures/plot_pmi.pdf"
    input: expand("arrays/Student-{df}.npz", df=[1, 2, 3, 5, 10]) + ["arrays/Gauss.npz"]
    run:
        fig, axs = subplots_from_axsize(
            axsize=(1.5, 1.5),
            nrows=2,
            ncols=6,
            top=0.5,
            left=0.6,
            right=0.15,
        )
        for i, inp in enumerate(input):
            arrays = np.load(inp)

            for ax in axs[:, i]:
                ax.scatter(arrays["samples_x"], arrays["samples_y"], rasterized=True, alpha=0.3, color='k', s=1)

            radius = RADIUS

            axs[0, i].imshow(arrays["pmi_true"], extent=[-radius, radius, radius, -radius], cmap="Purples")
            axs[1, i].imshow(arrays["pmi_critic"], extent=[-radius, radius, radius, -radius], cmap="Blues")
            
            # Custom title, making it shorter
            title = arrays["name"][0].strip("Bivariate ")
            title = title.capitalize()
            axs[0, i].set_title(title)
        
        axs[0, 0].set_ylabel("PMI")
        axs[1, 0].set_ylabel("Critic")

        fig.savefig(str(output))



rule generate_gauss_arrays:
    output: "arrays/Gauss.npz"
    run:
        task = bmi.benchmark.tasks.task_multinormal_dense(1, 1, off_diag=0.5)
        shape_matrix = np.asarray(task.params["covariance"])
        logratio_true = lambda x, y: gaussian_logpdfratio(x, y, shape_matrix=shape_matrix)
        np.savez(str(output), name=np.asarray(["Bivariate normal"]), **get_data(task, logratio_true))

rule generate_student_arrays:
    output: "arrays/Student-{df}.npz"
    run:
        df = int(wildcards.df)
        task = bmi.benchmark.tasks.task_student_dense(1, 1, off_diag=0.5, df=df)
        shape_matrix = np.asarray(task.params["dispersion"])
        logratio_true = lambda x, y: student_logpdfratio(x, y, df=task.params['dof'], shape_matrix=shape_matrix)
        np.savez(str(output), name=np.asarray([f"Bivariate Student, $\\nu={df}$"]), **get_data(task, logratio_true))
