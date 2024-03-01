import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from subplots_from_axsize import subplots_from_axsize


def read_results(
    path: str,
    unpack_task_params: bool = True,
    unpack_additional_information: bool = False,
):
    results = pd.read_csv(path)

    # read dicts
    for col in ["task_params", "additional_information"]:
        results[col] = results[col].apply(lambda x: yaml.safe_load(x))

    # unpack
    rows = []
    for _, row in results.iterrows():
        row_dict = row.to_dict()

        if unpack_task_params:
            row_dict |= row_dict["task_params"]
            del row_dict["task_params"]

        if unpack_additional_information:
            row_dict |= row_dict["additional_information"]
            del row_dict["additional_information"]

        rows.append(row_dict)

    results = pd.DataFrame(rows)
    return results


def subplots_benchmark(results):
    n_tasks = len(results["task_id"].unique())
    n_estimators = len(results["estimator_id"].unique())
    fig, ax = subplots_from_axsize(
        axsize=(n_tasks * 0.3, n_estimators * 0.3),
        left=1.2,
        bottom=4.0,
    )
    return fig, ax


# TODO(frdrc): estimator order, task order, estimator names, task names
def plot_benchmark(ax, results, n_samples=None):
    if n_samples is None:
        n_samples = results["n_samples"].max()

    data = results[results["n_samples"] == n_samples]
    data = results[["estimator_id", "task_id", "mi_estimate", "mi_true"]].copy()

    # relative_error
    with np.errstate(all="ignore"):
        data["log_relative_error"] = np.log(data["mi_estimate"] / data["mi_true"])

    # TODO(frdrc): filter out convergence failures

    # mean over seeds
    data = (
        data.groupby(["estimator_id", "task_id"])[["mi_estimate", "log_relative_error"]]
        .mean()
        .reset_index()
    )

    # add "True MI" pseudo-estimator
    task_id_mi_true = results[["task_id", "mi_true"]].drop_duplicates(subset=["task_id"])
    data = pd.concat(
        [
            data,
            pd.DataFrame(
                {
                    "estimator_id": "True MI",
                    "task_id": task_id_mi_true["task_id"],
                    "mi_estimate": task_id_mi_true["mi_true"],
                    "mi_true": task_id_mi_true["mi_true"],
                    "log_relative_error": 0.0,
                }
            ),
        ]
    )

    # create table of results
    table_mi = data.pivot(index="task_id", columns="estimator_id", values="mi_estimate")
    table_err = data.pivot(index="task_id", columns="estimator_id", values="log_relative_error")

    sns.heatmap(
        table_err.transpose(),
        annot=table_mi.transpose(),
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        ax=ax,
        square=False,
        fmt=".1f",
        cbar=False,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
