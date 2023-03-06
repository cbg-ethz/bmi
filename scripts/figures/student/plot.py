import argparse
from pathlib import Path
from typing import Callable, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bmi.api as bmi
from bmi.plot_utils.subplots_from_axsize import subplots_from_axsize

RENAME_DICT = {
    "Julia-Histograms": "Hist. (Julia)",
    "Python-CCA": "CCA (Python)",
    "Python-MINE": "MINE (Python)",
    "R-KSG-1": "KSG (R)",
    "R-LNN": "LNN (R)",
}

ESTIMATORS = {
    "Julia-Histograms": "Hist. (Julia)",
    "Julia-KSG-1": "KSG (Julia)",
    "Python-CCA": "CCA (Python)",
    "Python-MINE": "MINE (Python)",
    "R-KSG-1": "KSG (R)",
    "R-LNN": "LNN (R)",
}


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("RESULTS", type=Path, help="Path to the results directory.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to the output figure.",
        default=Path("student-mi-dof-plot.pdf"),
    )
    parser.add_argument("--strength", type=float, default=0.8, help="Strength of the interaction.")
    return parser


def result_to_dict(result):
    return dict(
        task_id=result.task_id,
        estimator_id=result.estimator_id,
        mi_estimate=result.mi_estimate,
        degrees_of_freedom=result.task_params["degrees_of_freedom"],
    )


def true_mi_factory(strength: float) -> tuple[float, Callable[[int], float]]:
    """For given interaction strength it returns:
    MI for corresponding Gaussian distribution
    function degrees of freedom (int)  ---> mutual information (float) to
      calculate the true Student MI
    """
    dispersion = np.eye(3)
    dispersion[0, 2] = dispersion[2, 0] = strength
    normal_mi = bmi.samplers.SplitStudentT(
        dim_x=2, dim_y=1, dispersion=dispersion, df=3
    ).mi_normal()

    def _true_mi_function(df: int) -> float:
        return (
            bmi.samplers.SplitStudentT.mi_correction_function(dim_x=2, dim_y=1, df=df) + normal_mi
        )

    return normal_mi, _true_mi_function


def read_results(results_dir: Path) -> pd.DataFrame:
    key = ["task_id", "estimator_id"]

    results = bmi.benchmark.SaveLoadRunResults.from_directory(results_dir)
    dataframe = pd.DataFrame(map(result_to_dict, results))

    # Compute SAMPLE mean and SAMPLE standard deviation
    mean_df = (
        dataframe.groupby(key)
        .mean()
        .reset_index()
        .rename({"mi_estimate": "estimate_mean"}, axis=1)
    )
    std_df = (
        dataframe.groupby(key)
        .std(ddof=0)
        .reset_index()
        .rename({"mi_estimate": "estimate_std"}, axis=1)
    )

    merged = pd.merge(
        mean_df,
        std_df,
        on=key,
    )
    return merged


def main() -> None:
    args = create_parser().parse_args()
    rng = np.random.default_rng(42)

    fig, ax = subplots_from_axsize(axsize=(3, 2))
    ax = cast(plt.Axes, ax)

    merged = read_results(args.RESULTS)

    x_values = set()

    for i, (estimator_id, mini_df) in enumerate(merged.groupby("estimator_id")):
        if estimator_id not in RENAME_DICT:
            continue
        estimator_name = RENAME_DICT[estimator_id]

        # Small multiplicative offset for the X axis, so that the points don't
        # look squished
        if estimator_id not in ESTIMATORS:
            print(f"Skipping estimator {estimator_id}...")
            continue
        estimator_name = ESTIMATORS[estimator_id]

        offset = 1 + rng.uniform() / 25 * (-1) ** i

        x_values = x_values.union(mini_df["degrees_of_freedom_x"].values)

        x = mini_df["degrees_of_freedom_x"].values * offset
        y = mini_df["estimate_mean"].values
        err_y = mini_df["estimate_std"].values

        ax.errorbar(x, y, yerr=err_y, fmt="o", capsize=5, alpha=0.5, label=estimator_name)

    x_values = sorted(map(int, x_values))
    normal_mi, true_mi = true_mi_factory(args.strength)
    x = list(range(1, max(x_values) + 1))
    ax.plot(
        x,
        [true_mi(_) for _ in x],
        c="k",
        linestyle="--",
        label="True MI",
    )

    ax.spines[["right", "top"]].set_visible(False)
    ax.set_xscale("log")
    ax.set_xticks(x_values)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    ax.plot(
        x,
        [normal_mi for _ in x],
        c="y",
        linestyle="-.",
        label="Gaussian MI",
    )

    ax.set_ylabel("Mutual Information")
    ax.set_xlabel("Degrees of freedom")

    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(args.output)


if __name__ == "__main__":
    main()
