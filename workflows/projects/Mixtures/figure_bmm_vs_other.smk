# Figure comparing BMMs and other estimators on selected problems.
# Note: to run this workflow, you need to have the results.csv files from:
#     - the benchmark (version 2) in `generated/benchmark/v2/results.csv`
#     - the BMM minibenchmark in `generated/projects/Mixtures/gmm_benchmark/results.csv`  
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from subplots_from_axsize import subplots_from_axsize



rule all:
    input: "generated/projects/Mixtures/figure_bmm_vs_other.pdf"


class YScaler:
    def __init__(self, estimator_ids: list[str], eps: float = 0.1):
        self._estimator_ids = estimator_ids
        assert eps > 0
        self._eps = eps

    @property
    def n(self) -> int:
        return len(self._estimator_ids)

    @property
    def offset(self) -> float:
        return self._eps * 0.5 * 1 / self.n

    def get_y(self, estimator_id: str, n_points: int) -> np.ndarray:
        index = self._estimator_ids.index(estimator_id)
        y0 = index / self.n
        y1 = (index + 1) / self.n

        return np.linspace(y0 + self._eps, y1 - self._eps, n_points)

    def get_tick_locations(self) -> list[float]:
        return (np.arange(self.n, dtype=float) + 0.5) / self.n


@dataclass
class TaskConfig:
    name: str
    xlim: tuple[float, float]
    xticks: list[float] | tuple[float, ...]

@dataclass
class EstimatorConfig:
    id: str
    name: str
    color: str

TASKS = {# task_id: task_name,
    '1v1-AI': TaskConfig(name="AI", xlim=(0.5, 0.85), xticks=[0.6, 0.7, 0.8]),
    'mult-sparse-w-inliers-5-5-2-2.0-0.2': TaskConfig(name="Inliers (5-dim, 0.2)", xlim=(0.4, 0.8), xticks=[0.45, 0.55, 0.65, 0.75]),
    '5v1-concentric_gaussians-5': TaskConfig(name="Concentric (5-dim, 5)", xlim=(0.35, 0.75), xticks=[0.4, 0.5, 0.6, 0.7]),
    'multinormal-sparse-5-5-2-2.0': TaskConfig(name="Normal (5-dim, sparse)", xlim=(0.65, 1.15), xticks=[0.7, 0.8, 0.9, 1.0, 1.1]),
}


# NAMES = {
#     # one-dimensional
#     '1v1-additive-0.75': "Additive",
#     '1v1-AI': "AI",
#     '1v1-X-0.9': "X",
#     '2v1-galaxy-0.5-3.0': "Galaxy",
#     # Concentric
#     '3v1-concentric_gaussians-10': "Concentric (3-dim, 10)",
#     '3v1-concentric_gaussians-5': "Concentric (3-dim, 5)",
#     '5v1-concentric_gaussians-10': "Concentric (5-dim, 10)",
#     '5v1-concentric_gaussians-5': "Concentric (5-dim, 5)",
#     # Inliers
#     'mult-sparse-w-inliers-5-5-2-2.0-0.2': "Inliers (5-dim, 0.2)",
#     'mult-sparse-w-inliers-5-5-2-2.0-0.5': "Inliers (5-dim, 0.5)",
#     # Multivariate normal
#     'multinormal-dense-5-5-0.5': "Normal (5-dim, dense)",
#     'multinormal-sparse-5-5-2-2.0': "Normal (5-dim, sparse)",
#     # Student
#     'asinh-student-identity-1-1-1': "Student (1-dim)",
#     'asinh-student-identity-2-2-1': "Student (2-dim)",
#     'asinh-student-identity-3-3-2': "Student (3-dim)",
#     'asinh-student-identity-5-5-2': "Student (5-dim)",
# }

# TASKS = {
#     id_v: TaskConfig(name=name, xlim=(0.2, 1), xticks=[]) for id_v, name in NAMES.items()
# }


N_SAMPLES = 5_000
POINT_ESTIMATORS = [
    EstimatorConfig(id="KSG-10", name="KSG", color="green"),
    EstimatorConfig(id="InfoNCE", name="InfoNCE", color="magenta"),
]

DOT_SIZE = 7


rule generate_figure:
    output: "generated/projects/Mixtures/figure_bmm_vs_other.pdf"
    input:
        v2 = "generated/benchmark/v2/results.csv",
        bmm = "generated/projects/Mixtures/gmm_benchmark/results.csv"
    run:
        data_v2 = pd.read_csv(input.v2)
        data_bmm = pd.read_csv(input.bmm)

        fig, axs = subplots_from_axsize(1, len(TASKS), (2.3, 0.8), left=0.8, right=0.05, top=0.3, bottom=0.3, dpi=350, wspace=0.05)
        
        y_scaler = YScaler(estimator_ids=["BMM"] + [config.id for config in POINT_ESTIMATORS], eps=0.12)

        for ax, (task_id, task_config) in zip(axs.ravel(), TASKS.items()):
            ax.set_title(task_config.name)
            ax.set_xlim(*task_config.xlim)
            ax.set_xticks(task_config.xticks)
            ax.set_yticks([])
            ax.set_ylim(-0.05, 1.01)
            ax.spines[["top", "left", "right"]].set_visible(False)

            mi_true = data_v2.groupby("task_id")["mi_true"].mean()[task_id]
            ax.axvline(mi_true, linestyle=":", color="black", linewidth=2)

            # Plot credible intervals from the BMM
            bmm_subtable = data_bmm[(data_bmm["task_id"] == task_id)].copy() 
            bmm_subtable["errorbar_low"] = bmm_subtable["mi_mean"] - bmm_subtable["mi_q_low"]
            bmm_subtable["errorbar_high"] = bmm_subtable["mi_q_high"] - bmm_subtable["mi_mean"]

            y = y_scaler.get_y(estimator_id="BMM", n_points=len(bmm_subtable))
            ax.errorbar(x=bmm_subtable["mi_mean"].values, y=y, xerr=bmm_subtable[["errorbar_low", "errorbar_high"]].T, capsize=3, ls="none", color="darkblue")
            ax.scatter(x=bmm_subtable["mi_mean"].values, y=y, color="darkblue", s=DOT_SIZE)

            # Plot the scatterplot representing estimators
            for estimator_config in POINT_ESTIMATORS:
                estimator_id = estimator_config.id

                index = (data_v2["task_id"] == task_id) & (data_v2["estimator_id"] == estimator_id) & (data_v2["n_samples"] == N_SAMPLES)
                estimates = data_v2[index]["mi_estimate"].values
                y = y_scaler.get_y(estimator_id=estimator_id, n_points=len(estimates))
                ax.scatter(estimates, y, color=estimator_config.color, s=DOT_SIZE, alpha=0.4)

        ax = axs[0]
        ax.set_yticks(y_scaler.get_tick_locations(), ["BMM"] + [config.name for config in POINT_ESTIMATORS])
        ax.spines["left"].set_visible(True)
        
        fig.savefig(str(output))
