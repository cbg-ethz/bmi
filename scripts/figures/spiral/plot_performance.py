"""Plots the run results."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bmi.api as bmi


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("TASKS", type=Path, help="Directory with tasks.")
    parser.add_argument("RUN_RESULTS", type=Path, help="Directory with run results.")
    parser.add_argument("OUTPUT", type=str, help="Output file.")
    return parser


TASK_ID = str
ESTIMATOR_ID = str


def read_tasks(path: Path) -> dict[TASK_ID, bmi.benchmark.TaskMetadata]:
    ret = {}
    for task_dir in path.iterdir():
        task = bmi.benchmark.Task.load(task_dir)
        ret[task.task_id] = task.metadata
    return ret


def read_speed(task_id: TASK_ID) -> float:
    # TODO(Pawel): Refactor this, we should keep it somewhere,
    #   rather than parse it.
    return float(task_id.split("speed")[1].split("_")[1].strip())


def read_results(path: Path) -> list[bmi.RunResult]:
    results = []
    for json_path in path.iterdir():
        with open(json_path) as f:
            result = bmi.RunResult(**json.load(f))
            results.append(result)
    return results


def plot(
    mi_true: float, plotting_results: list[dict[str, float]], color_true: str = "k"
) -> plt.Figure:
    speed_list = sorted({res["speed"] for res in plotting_results})
    estimator_list = sorted({res["estimator"] for res in plotting_results})
    fig, ax = plt.subplots(figsize=(4, 3))

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

    ax.legend()
    return fig


def main() -> None:
    args = create_parser().parse_args()

    tasks_dict = read_tasks(args.TASKS)
    results_list = read_results(args.RUN_RESULTS)

    mi_true_values = [metadata.mi_true for metadata in tasks_dict.values()]

    # There may be small differences with deserialization
    _THRESHOLD = 1e-3
    if max(mi_true_values) - min(mi_true_values) > _THRESHOLD:
        raise ValueError(f"The MI true value is not unique: {mi_true_values}.")
    mi_true = float(np.mean(mi_true_values))

    results_df = pd.DataFrame([result.dict() for result in results_list])

    plotting_results = []
    for (task_id, estimator_id), group_df in results_df.groupby(["task_id", "estimator_id"]):
        mean = np.mean(group_df["mi_estimate"].values)
        std = np.std(group_df["mi_estimate"].values, ddof=0)
        plotting_results.append(
            {
                "speed": read_speed(task_id),
                "estimator": estimator_id,
                "mean": mean,
                "std": std,
            }
        )

    fig = plot(plotting_results=plotting_results, mi_true=mi_true)
    fig.tight_layout()
    fig.savefig(args.OUTPUT)


if __name__ == "__main__":
    main()
