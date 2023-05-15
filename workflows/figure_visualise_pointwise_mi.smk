import jax
import jax.numpy as jnp

import matplotlib
import matplotlib.pyplot as plt

import bmi

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


def visualise(
    task,
    logratio_true,
):
    estimator = bmi.estimators.InfoNCEEstimator()
    samples_x, samples_y = task.sample(10_000, 42)
    estimate = estimator.estimate(samples_x, samples_y)

    print(f"True MI: {task.mutual_information:.2f}. Estimate: {estimate:.2f}")
    
    critic = estimator.trained_critic
    radius = 3
    xs = jnp.linspace(-radius, radius, 51)
    ys = jnp.linspace(-radius, radius, 51)

    mxs, mys = jnp.meshgrid(xs, ys)

    pmi_critic = jax.vmap(jax.vmap(critic))(mxs[..., None], mys[..., None])


    fig, axs = plt.subplots(1, 2, figsize=(4.5, 2.3))

    pmi_true = np.vectorize(logratio_true)(mxs[..., None], mys[..., None])[..., 0]
    ax = axs[0]

    ax.scatter(samples_x[:1000], samples_y[:1000], s=1, color='k', alpha=0.2, rasterized=True)
    ax.imshow(pmi_true, extent=[-radius, radius, radius, -radius], cmap="Purples")

    ax.set_title("Pointwise MI")
    ax.set_xlabel("$X$")
    ax.set_ylabel("$Y$")


    ax = axs[1]

    ax.scatter(samples_x[:1000], samples_y[:1000], s=1, color='k', alpha=0.2, rasterized=True)
    ax.imshow(pmi_critic, extent=[-radius, radius, radius, -radius], cmap="Blues")

    ax.set_title("Critic")
    ax.set_xlabel("$X$")
    ax.set_ylabel("$Y$")

    fig.tight_layout()

    return fig


rule all:
    input: ["Gauss.pdf"] + expand("Student-{df}.pdf", df=[1, 2, 3, 5, 10])


rule generate_gauss:
    output: "Gauss.pdf"
    run:
        task = bmi.benchmark.tasks.task_multinormal_dense(1, 1, off_diag=0.5)
        shape_matrix = np.asarray(task.params["covariance"])
        logratio_true = lambda x, y: gaussian_logpdfratio(x, y, shape_matrix=shape_matrix)
        fig = visualise(task, logratio_true)
        fig.suptitle("Bivariate normal")
        fig.savefig(str(output), dpi=200)

rule generate_student:
    output: "Student-{df}.pdf"
    run:
        df = int(wildcards.df)
        task = bmi.benchmark.tasks.task_student_dense(1, 1, off_diag=0.5, df=df)
        shape_matrix = np.asarray(task.params["dispersion"])
        logratio_true = lambda x, y: student_logpdfratio(x, y, df=task.params['dof'], shape_matrix=shape_matrix)

        fig = visualise(task, logratio_true)
        fig.suptitle(f"Bivariate Student, $\\nu={df}$")
        fig.savefig(str(output), dpi=200)