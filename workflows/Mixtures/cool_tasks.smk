"""Demonstration of the capabilities of the fine distribution family."""
import bmi.estimators as estimators
from bmi.benchmark.tasks import transform_rescale
import resource
import yaml

import pandas as pd

from bmi.benchmark import run_estimator

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib
from subplots_from_axsize import subplots_from_axsize
matplotlib.use("agg")

import bmi
from bmi.samplers import fine

# --- Define samplers ---

# The X distribution
x_dist = fine.mixture(
    proportions=jnp.array([0.5, 0.5]),
    components=[
        fine.MultivariateNormalDistribution(
            covariance=0.3 * bmi.samplers.canonical_correlation([x * 0.9]),
            mean=jnp.zeros(2),
            dim_x=1, dim_y=1,
        ) for x in [-1, 1]
    ]
)
x_sampler = fine.FineSampler(x_dist)

# The fence distribution
n_components = 12

base_dist = fine.mixture(
    proportions=jnp.ones(n_components) / n_components,
    components=[
        fine.MultivariateNormalDistribution(
            covariance=jnp.diag(jnp.array([0.1, 1.0, 0.1])),
            mean=jnp.array([x, 0, x%4]) * 1.5,
            dim_x=2, dim_y=1,
        ) for x in range(n_components)
    ]
)
base_sampler = fine.FineSampler(base_dist)
fence_aux_sampler = bmi.samplers.TransformedSampler(
    base_sampler,
    transform_x=lambda x: x + jnp.array([5., 0.]) * jnp.sin(3 * x[1]),
)
fence_sampler = bmi.samplers.TransformedSampler(
    fence_aux_sampler,
    transform_x=lambda x: jnp.array([0.1 * x[0]-0.8, 0.5 * x[1]])
)

# The AI distribution
corr = 0.95
var_x = 0.04

ai_dist = fine.mixture(
    proportions=np.full(6, fill_value=1/6),
    components=[
        # I components
        fine.MultivariateNormalDistribution(
            dim_x=1, dim_y=1,
            mean=np.array([1., 0.]),
            covariance=np.diag([0.01, 0.2]),
        ),
        fine.MultivariateNormalDistribution(
            dim_x=1, dim_y=1,
            mean=np.array([1., 1]),
            covariance=np.diag([0.05, 0.001]),
        ),    
        fine.MultivariateNormalDistribution(
            dim_x=1, dim_y=1,
            mean=np.array([1., -1]),
            covariance=np.diag([0.05, 0.001]),
        ),   
        # A components
        fine.MultivariateNormalDistribution(
            dim_x=1, dim_y=1,
            mean=np.array([-0.8, -0.2]),
            covariance=np.diag([0.03, 0.001]),
        ),  
        fine.MultivariateNormalDistribution(
            dim_x=1, dim_y=1,
            mean=np.array([-1.2, 0.]),
            covariance=np.array([[var_x, np.sqrt(var_x * 0.2) * corr], [np.sqrt(var_x * 0.2) * corr, 0.2]]),
        ),
        fine.MultivariateNormalDistribution(
            dim_x=1, dim_y=1,
            mean=np.array([-0.4, 0.]),
            covariance=np.array([[var_x, -np.sqrt(var_x * 0.2) * corr], [-np.sqrt(var_x * 0.2) * corr, 0.2]]),
        ),
    ]
)
ai_sampler = fine.FineSampler(ai_dist)

# Balls mixed with spiral

balls_mixt = fine.mixture(
    proportions=jnp.array([0.5, 0.5]),
    components=[
        fine.MultivariateNormalDistribution(
            covariance=bmi.samplers.canonical_correlation([0.0], additional_y=1),
            mean=jnp.array([x, x, x]) * 1.5,
            dim_x=2, dim_y=1,
        ) for x in [-1, 1]
    ]
)

base_balls_sampler = fine.FineSampler(balls_mixt)
a = jnp.array([[0, -1], [1, 0]])
spiral = bmi.transforms.Spiral(a, speed=0.5)

sampler_balls_aux = bmi.samplers.TransformedSampler(
    base_balls_sampler,
    transform_x=spiral
)
sampler_balls_transformed = bmi.samplers.TransformedSampler(
    sampler_balls_aux,
    transform_x=lambda x: 0.3 * x,
)


N_SAMPLES = [1_000, 5_000 ]
SEEDS = list(range(10))

ESTIMATORS = {
    "MINE": bmi.estimators.MINEEstimator(verbose=False),
    "InfoNCE": bmi.estimators.InfoNCEEstimator(verbose=False),
    # "NWJ": bmi.estimators.NWJEstimator(verbose=False),
    # "Donsker-Varadhan": bmi.estimators.DonskerVaradhanEstimator(verbose=False),
    'KSG-10': bmi.estimators.KSGEnsembleFirstEstimator(neighborhoods=(10,)),
    'Hist-10': bmi.estimators.HistogramEstimator(n_bins_x=10),
    "CCA": bmi.estimators.CCAMutualInformationEstimator(),
}
ESTIMATOR_NAMES = {
    "MINE": "MINE",
    "InfoNCE": "InfoNCE",
    "KSG-10": "KSG",
    "Hist-10": "Histogram",
    "CCA": "CCA",
}
assert set(ESTIMATOR_NAMES.keys()) == set(ESTIMATORS.keys())

TASKS_UNSCALED = {
    "X": bmi.benchmark.Task(
        sampler=x_sampler,
        task_id="X",
        task_name="X",
    ),
    "AI": bmi.benchmark.Task(
        sampler=ai_sampler,
        task_id="AI",
        task_name="AI",
    ),
    "Fence": bmi.benchmark.Task(
        sampler=fence_sampler,
        task_id="Fence",
        task_name="Fence",
    ),
    "Balls": bmi.benchmark.Task(
        sampler=sampler_balls_transformed,
        task_id="Balls",
        task_name="Balls",
    ),
}

def scale_tasks(tasks: dict[str, bmi.Task]) -> dict[str, bmi.Task]:
    """Auxiliary method used to rescale (whiten) each task in the list,
    without changing its name nor id."""
    return {
        key: transform_rescale(
            base_task=base_task,
            task_name=base_task.name,
            task_id=base_task.id,
        )
        for key, base_task in tasks.items()
    }

TASKS = scale_tasks(TASKS_UNSCALED)


# === WORKDIR ===
workdir: "generated/mixtures/cool_tasks/"

rule all:
    input:
        'cool_tasks.pdf',
        'results.csv',
        'results.pdf'

rule plot_distributions:
    output: "cool_tasks.pdf"
    run:
        fig, axs = subplots_from_axsize(1, 4, axsize=(3, 3))

        # Plot the X distribution
        ax = axs[0]
        xs, ys = x_sampler.sample(1000, 0)

        ax.scatter(xs[:, 0], ys[:, 0], s=4**2, alpha=0.3, color="k", rasterized=True)
        ax.set_xlabel("$X$")
        ax.set_ylabel("$Y$")

        # Plot the AI distribution
        ax = axs[1]
        xs, ys = ai_sampler.sample(2000, 0)
        ax.scatter(xs[:, 0], ys[:, 0], s=4**2, alpha=0.3, color="k", rasterized=True)
        ax.set_xlabel("$X$")
        ax.set_ylabel("$Y$")

        # Plot the fence distribution
        ax = axs[2]
        xs, ys = fence_sampler.sample(2000, 0)

        ax.scatter(xs[:, 0], xs[:, 1], c=ys[:, 0], s=4**2, alpha=0.3, rasterized=True)
        ax.set_xlabel("$X_1$")
        ax.set_ylabel("$X_2$")

        # Plot transformed balls distribution
        ax = axs[3]
        xs, ys = sampler_balls_transformed.sample(2000, 0)
        ax.scatter(xs[:, 0], xs[:, 1], c=ys[:, 0], s=4**2, alpha=0.3, rasterized=True)
        ax.set_xlabel("$X_1$")
        ax.set_ylabel("$X_2$")

        for ax in axs:
            ax.set_xlim(-2., 2.)
            ax.set_ylim(-2., 2.)

        fig.savefig(str(output))

rule plot_results:
    output: 'results.pdf'
    input: 'results.csv'
    run:
        data = pd.read_csv(str(input))
        fig, ax = subplots_from_axsize(1, 1, (4, 3))

        data_5k = data[data['n_samples'] == 5000]
        tasks = ['X', 'AI', 'Fence', 'Balls']
        tasks_official = ['X', 'AI', 'Waves', 'Galaxy']

        for estimator_id, data_est in data_5k.groupby('estimator_id'):
            ax.scatter(
                data_est['task_id'].apply(lambda e: tasks.index(e)) + 0.05 * np.random.normal(size=len(data_est)),
                data_est['mi_estimate'],
                label=ESTIMATOR_NAMES[estimator_id],
                alpha=0.4, s=5**2,
            )
            
        for task_id, data_task in data_5k.groupby('task_id'):
            true_mi = data_task['mi_true'].mean()
            x = tasks.index(task_id)
            ax.plot([x - 0.2, x + 0.2], [true_mi, true_mi], ':k')

        ax.set_xticks(range(len(tasks)), tasks_official)
            
        ax.legend(frameon=False, loc='upper left')
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylim(-0.1, 1.4)
        ax.set_ylabel('Mutual information [nats]')
        fig.savefig(str(output))

rule results:
    output: 'results.csv'
    input:
        expand(
            'results/{estimator_id}/{task_id}/{n_samples}-{seed}.yaml',
            estimator_id=ESTIMATORS,
            task_id=TASKS,
            n_samples=N_SAMPLES,
            seed=SEEDS,
        )
    run:
        results = []
        for result_path in input:
            with open(result_path) as f:
                result = yaml.load(f, yaml.SafeLoader)
                task = TASKS[result['task_id']]
                result['mi_true'] = task.mutual_information
                result['task_params'] = task.params
                results.append(result)
        pd.DataFrame(results).to_csv(str(output), index=False)


# Sample task given a seed and number of samples.
rule sample_task:
    output: 'tasks/{task_id}/{n_samples}-{seed}.csv'
    run:
        task_id = wildcards.task_id
        n_samples = int(wildcards.n_samples)
        seed = int(wildcards.seed)
        task = TASKS[task_id]
        task.save_sample(str(output), n_samples, seed)


# Apply estimator to sample
rule apply_estimator:
    output: 'results/{estimator_id}/{task_id}/{n_samples}-{seed}.yaml'
    input: 'tasks/{task_id}/{n_samples}-{seed}.csv'
    run:
        estimator_id = wildcards.estimator_id
        estimator = ESTIMATORS[estimator_id]
        task_id = wildcards.task_id
        seed = int(wildcards.seed)
        
        # this should be about ~4GiB
        resource.setrlimit(resource.RLIMIT_DATA, (1<<33, 1<<33))

        result = run_estimator(
            estimator=estimator,
            estimator_id=estimator_id,
            sample_path=str(input),
            task_id=task_id,
            seed=seed,
        )
        result.dump(str(output))

