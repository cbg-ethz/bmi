import common_estimators as ce
import matplotlib
import pandas as pd
import yaml

from bmi.plot_utils.subplots_from_axsize import subplots_from_axsize

matplotlib.use("agg")


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


def prepare_fig_axs(
    axx=2.0,
    axy=1.5,
    **kwargs,
):
    fig, axs = subplots_from_axsize(axsize=(axx, axy), **kwargs)

    # formatting
    for ax in axs.reshape(-1):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    return fig, axs


def plot_mi(ax, results, x_col):
    for estimator_id, data_estimator in results.groupby("estimator_id"):
        data_mean = data_estimator.groupby(x_col)[["mi_estimate"]].mean().reset_index()
        ax.plot(
            data_mean[x_col],
            data_mean["mi_estimate"],
            color=ce.ESTIMATOR_COLORS[estimator_id],
            label=ce.ESTIMATOR_NAMES[estimator_id],
        )

    ax.legend()
    ax.set_xlabel(x_col)
    ax.set_ylabel("MI estimate [nats]")
