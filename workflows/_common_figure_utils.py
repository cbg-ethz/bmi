import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

import bmi
import bmi.estimators
import bmi.estimators.external.julia_estimators as julia_estimators
import bmi.estimators.external.r_estimators as r_estimators
from bmi.benchmark.tasks import transform_rescale

matplotlib.use("agg")


ESTIMATORS = {
    "MINE": bmi.estimators.MINEEstimator(verbose=False),
    "InfoNCE": bmi.estimators.InfoNCEEstimator(verbose=False),
    "NWJ": bmi.estimators.NWJEstimator(verbose=False),
    "Donsker-Varadhan": bmi.estimators.DonskerVaradhanEstimator(verbose=False),
    # 'KSG-5': bmi.estimators.KSGEnsembleFirstEstimator(neighborhoods=(5,)),
    # 'KSG-10': bmi.estimators.KSGEnsembleFirstEstimator(neighborhoods=(10,)),
    # 'Hist-10': bmi.estimators.HistogramEstimator(n_bins_x=10),
    "R-KSG-I-10": r_estimators.RKSGEstimator(variant=1, neighbors=5),
    # 'R-KSG-I-10': r_estimators.RKSGEstimator(variant=1, neighbors=10),
    # 'R-KSG-II-5': r_estimators.RKSGEstimator(variant=2, neighbors=5),
    # 'R-KSG-II-10': r_estimators.RKSGEstimator(variant=2, neighbors=10),
    # 'R-BNSL': r_estimators.RBNSLEstimator(),
    "R-LNN": r_estimators.RLNNEstimator(),
    "Julia-Hist-10": julia_estimators.JuliaHistogramEstimator(bins=10),
    # 'Julia-Kernel': julia_estimators.JuliaKernelEstimator(),
    # 'Julia-Transfer-30': julia_estimators.JuliaTransferEstimator(bins=30),
    # 'Julia-KSG-I-5': julia_estimators.JuliaKSGEstimator(variant=1, neighbors=5),
}

ESTIMATOR_COLORS = {
    "MINE": "purple",
    "InfoNCE": "magenta",
    "Donsker-Varadhan": "red",
    "NWJ": "orangered",
    "R-KSG-I-10": "mediumblue",
    "R-LNN": "goldenrod",
    "Julia-Hist-10": "limegreen",
}
assert set(ESTIMATORS.keys()) <= set(ESTIMATOR_COLORS.keys())

# shown on plots
ESTIMATOR_NAMES = {
    "MINE": "MINE",
    "InfoNCE": "InfoNCE",
    "Donsker-Varadhan": "D-V",
    "NWJ": "NWJ",
    "KSG-10": "KSG I (Python)",
    "R-KSG-I-10": "KSG I (R)",
    "R-LNN": "LNN (R)",
    "R-BNSL": "BNSL (R)",
    "Julia-Hist-10": "Hist. (Julia)",
    "Julia-Transfer-30": "Transfer (Julia)",
    "Julia-Kernel": "Kernel (Julia)",
}
assert set(ESTIMATORS.keys()) <= set(ESTIMATOR_NAMES.keys())


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


def format_axs(axs):
    for ax in axs.reshape(-1):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


# BASIC MI PLOT


def plot_mi(
    ax, results, x_col, estimator_colors=ESTIMATOR_COLORS, estimator_names=ESTIMATOR_NAMES
):
    for estimator_id, data_estimator in results.groupby("estimator_id"):
        data_mean = data_estimator.groupby(x_col)[["mi_estimate"]].mean().reset_index()
        ax.plot(
            data_mean[x_col],
            data_mean["mi_estimate"],
            color=estimator_colors[estimator_id],
            label=estimator_names[estimator_id],
        )

    data_mean = results.groupby(x_col)[["mi_true"]].mean().reset_index()
    ax.plot(
        data_mean[x_col],
        data_mean["mi_true"],
        linestyle=":",
        color="black",
        label="True MI",
    )

    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(x_col)
    ax.set_ylabel("MI estimate [nats]")


# BENCHMARK


def scale_tasks(tasks: dict[str, bmi.Task]) -> dict[str, bmi.Task]:
    """Auxiliary method used to rescale (whiten) each task in the list,
    without changing its name."""
    return {
        key: transform_rescale(
            base_task=base_task,
            task_name=base_task.name,
            task_id=base_task.id,
        )
        for key, base_task in tasks.items()
    }


def preprocess_benchmark_results(results, estimators=ESTIMATORS):
    data = results.copy()

    # detect neural estimators
    neural_ids = {
        estimator_id
        for estimator_id, estimator in estimators.items()
        if (
            isinstance(estimator, bmi.estimators.NeuralEstimatorBase)
            or isinstance(estimator, bmi.estimators.MINEEstimator)
        )
    }
    data["neural_estimator"] = data["estimator_id"].isin(neural_ids)

    # detect neural convergence issues
    # TODO(frdrc): we can do better than a flat 0.1
    data["neural_fail"] = (data["mi_estimate"] < 0.1) & data["neural_estimator"]

    # relative_error
    with np.errstate(all="ignore"):
        data["log_relative_error"] = np.log(data["mi_estimate"] / data["mi_true"])

    return data


def make_benchmark_table(data, values, estimators, tasks, estimator_names):
    table = data.pivot(index="task_id", columns="estimator_id", values=values)

    # reorder
    task_order = [task_id for task_id in tasks.keys() if task_id in set(data["task_id"])]
    table = table.reindex(task_order)
    estimator_order = [
        estimator_id
        for estimator_id in estimators.keys()
        if estimator_id in set(data["estimator_id"])
    ]
    table = table[estimator_order]

    # pretty names
    table.rename(
        inplace=True,
        index=lambda i: tasks[i].name,
        columns=lambda e: estimator_names[e],
    )

    return table


def plot_benchmark_mi_estimate(ax, results, estimators, tasks, estimator_names={}):
    estimator_names = ESTIMATOR_NAMES | estimator_names

    data = preprocess_benchmark_results(results, estimators)
    data = data[data["n_samples"] == data["n_samples"].max()]
    data = data[~data["neural_fail"]]

    # mean over seeds
    data = (
        data.groupby(["estimator_id", "task_id", "n_samples"])[
            ["mi_estimate", "log_relative_error"]
        ]
        .mean()
        .reset_index()
    )

    table_estimate = make_benchmark_table(data, "mi_estimate", estimators, tasks, estimator_names)
    table_estimate["True MI"] = [task.mutual_information for task in tasks.values()]

    table_log_rel_err = make_benchmark_table(
        data, "log_relative_error", estimators, tasks, estimator_names
    )
    table_log_rel_err["True MI"] = 0.0

    sns.heatmap(
        table_log_rel_err.transpose(),
        annot=table_estimate.transpose(),
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


def n_samples_needed(data_estimator_task):
    data_solved = data_estimator_task[
        np.abs(data_estimator_task["log_relative_error"]) < np.log(1.5)
    ]

    if len(data_solved):
        return data_solved["n_samples"].min()
    else:
        return np.inf


def n_samples_annotator(max_n_samples):
    def annot(val):
        if np.isnan(val):
            return ""

        if np.isinf(val):
            return f">{int(max_n_samples / 1000)}k"

        return f"{int(val / 1000)}k"

    return annot


def plot_benchmark_n_samples(ax, results, estimators, tasks, estimator_names={}):
    estimator_names = ESTIMATOR_NAMES | estimator_names

    max_n_samples = results["n_samples"].max()

    data = preprocess_benchmark_results(results, estimators)
    data = data[~data["neural_fail"]]
    data = (
        data.groupby(["estimator_id", "task_id", "n_samples"])[
            ["mi_estimate", "log_relative_error"]
        ]
        .mean()
        .reset_index()
    )

    data_samples = pd.DataFrame()
    data_samples["n_samples"] = data.groupby(["estimator_id", "task_id"]).apply(n_samples_needed)
    data_samples = data_samples.reset_index()

    table_samples = make_benchmark_table(
        data_samples, "n_samples", estimators, tasks, estimator_names
    )
    table_annot = table_samples.applymap(n_samples_annotator(max_n_samples))

    sns.heatmap(
        np.clip(table_samples.transpose(), a_min=0.0, a_max=100 * max_n_samples),
        annot=table_annot.transpose(),
        cmap="Spectral_r",
        vmin=0,
        vmax=max_n_samples * 1.2,
        ax=ax,
        square=False,
        fmt="",
        cbar=False,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")


def plot_benchmark_neural_fails(ax, results, estimators, tasks, estimator_names={}):
    estimator_names = ESTIMATOR_NAMES | estimator_names

    data = preprocess_benchmark_results(results, estimators)
    data = data[data["n_samples"] == data["n_samples"].max()]
    data = data[data["neural_estimator"]]
    data = (
        data.groupby(["estimator_id", "task_id", "n_samples"])[["neural_fail"]].sum().reset_index()
    )

    table = make_benchmark_table(data, "neural_fail", estimators, tasks, estimator_names)

    sns.heatmap(
        table.transpose(),
        annot=True,
        cmap="gray_r",
        ax=ax,
        square=False,
        fmt="",
        cbar=False,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
