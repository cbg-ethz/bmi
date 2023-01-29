"""Plots the run results."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bmi.api as bmi

RENAME_DICT = {
    "Julia-Histograms": "Hist. (Julia)",
    "Python-CCA": "CCA (Python)",
    "Python-MINE": "MINE (Python)",
    "R-KSG-1": "KSG (R)",
    "R-LNN": "LNN (R)",
}


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("TASKS", type=Path, help="Directory with tasks.")
    parser.add_argument("RUN_RESULTS", type=Path, help="Directory with run results.")
    parser.add_argument("OUTPUT", type=str, help="Output file.")
    return parser


TASK_ID = str
ESTIMATOR_ID = str


def plot(
    mi_true: float, plotting_results: list[dict[str, float]], color_true: str = "k"
) -> plt.Figure:
    speed_list = sorted({res["speed"] for res in plotting_results})
    estimator_list = sorted({res["estimator"] for res in plotting_results})
    fig, ax = plt.subplots(figsize=(6, 3))

    ax.hlines(
        mi_true,
        xmin=min(speed_list),
        xmax=max(speed_list),
        linestyles="dashed",
        label="True",
        colors=color_true,
    )

    ax.set_ylabel("Mutual Information")
    ax.set_xlabel("Speed")

    for i, estimator in enumerate(estimator_list, 1):
        vals = sorted(
            [
                (res["speed"], res["mean"], res["std"])
                for res in plotting_results
                if res["estimator"] == estimator
            ]
        )
        vals = np.asarray(vals)

        ax.plot(vals[:, 0], vals[:, 1], label=estimator, color=f"C{i}")
        ax.fill_between(
            vals[:, 0], vals[:, 1] - vals[:, 2], vals[:, 1] + vals[:, 2], alpha=0.5, color=f"C{i}"
        )
    fig.subplots_adjust(left=0.1, right=0.67, bottom=0.2, top=0.97)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 0.9), ncol=1, fancybox=True, shadow=False)

    return fig


def main() -> None:
    args = create_parser().parse_args()

    tasks_dict = bmi.benchmark.LoadTaskMetadata.from_directory(args.TASKS)
    results_list = bmi.benchmark.SaveLoadRunResults.from_directory(args.RUN_RESULTS)

    mi_true_values = [metadata.mi_true for metadata in tasks_dict.values()]

    # There may be small differences with deserialization
    _THRESHOLD = 1e-3
    if max(mi_true_values) - min(mi_true_values) > _THRESHOLD:
        raise ValueError(f"The MI true value is not unique: {mi_true_values}.")
    mi_true = float(np.mean(mi_true_values))

    results_df = pd.DataFrame([result.dict() for result in results_list])

    plotting_results = []
    for (task_id, estimator_id), group_df in results_df.groupby(["task_id", "estimator_id"]):
        if estimator_id not in RENAME_DICT:
            continue
        estimator_name = RENAME_DICT[estimator_id]

        mean = np.mean(group_df["mi_estimate"].values)
        std = np.std(group_df["mi_estimate"].values, ddof=0)
        plotting_results.append(
            {
                "speed": tasks_dict[task_id].task_params["speed"],
                "estimator": estimator_name,
                "mean": mean,
                "std": std,
            }
        )

    fig = plot(plotting_results=plotting_results, mi_true=mi_true)
    # fig.tight_layout()
    fig.savefig(args.OUTPUT)


if __name__ == "__main__":
    main()
