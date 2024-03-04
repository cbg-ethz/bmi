import numpy as np
import pandas as pd
import yaml


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


def create_benchmark_table(results, n_samples=None):
    if n_samples is None:
        n_samples = results["n_samples"].max()

    data = results[results["n_samples"] == n_samples]
    data = results[["estimator_id", "task_id", "mi_estimate", "mi_true"]].copy()

    # relative_error
    with np.errstate(all="ignore"):
        data["log_relative_error"] = np.log2(data["mi_estimate"] / data["mi_true"])

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

    def make_pretty(styler):
        styler.format(lambda x: f"{x:.2f}")
        styler.set_table_styles(
            [{"selector": "td", "props": "text-align: center; min-width: 5em;"}]
        )
        styler.background_gradient(
            vmin=-1.0,
            vmax=+1.0,
            cmap="coolwarm",
            gmap=table_err,
            axis=None,
        )
        return styler

    table_pretty = table_mi.style.pipe(make_pretty)
    table_pretty.index.name = None
    table_pretty.columns.name = None

    return table_pretty
