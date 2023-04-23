import matplotlib
import pandas as pd
import yaml

import bmi.estimators as estimators
import bmi.estimators.external.julia_estimators as julia_estimators
import bmi.estimators.external.r_estimators as r_estimators

matplotlib.use("agg")


ESTIMATORS = {
    "MINE": estimators.MINEEstimator(verbose=False),
    "InfoNCE": estimators.InfoNCEEstimator(verbose=False),
    "NWJ": estimators.NWJEstimator(verbose=False),
    "Donsker-Varadhan": estimators.DonskerVaradhanEstimator(verbose=False),
    # 'KSG-5': estimators.KSGEnsembleFirstEstimator(neighborhoods=(5,)),
    # 'KSG-10': estimators.KSGEnsembleFirstEstimator(neighborhoods=(10,)),
    # 'Hist-10': estimators.HistogramEstimator(n_bins_x=10),
    "R-KSG-I-5": r_estimators.RKSGEstimator(variant=1, neighbors=5),
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
    "R-KSG-I-5": "mediumblue",
    "R-LNN": "goldenrod",
    "Julia-Hist-10": "limegreen",
}

# shown on plots
ESTIMATOR_NAMES = {
    "MINE": "MINE",
    "InfoNCE": "InfoNCE",
    "Donsker-Varadhan": "D-V",
    "NWJ": "NWJ",
    "R-KSG-I-5": "KSG I (R)",
    "R-LNN": "LNN (R)",
    "Julia-Hist-10": "Hist. (Julia)",
}


assert set(ESTIMATORS.keys()) == set(ESTIMATOR_COLORS.keys())
assert set(ESTIMATORS.keys()) == set(ESTIMATOR_NAMES.keys())


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


def plot_mi(ax, results, x_col):
    for estimator_id, data_estimator in results.groupby("estimator_id"):
        data_mean = data_estimator.groupby(x_col)[["mi_estimate"]].mean().reset_index()
        ax.plot(
            data_mean[x_col],
            data_mean["mi_estimate"],
            color=ESTIMATOR_COLORS[estimator_id],
            label=ESTIMATOR_NAMES[estimator_id],
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
