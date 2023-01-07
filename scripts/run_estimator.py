"""This script loads a given estimator and runs it on a given task."""
import argparse
from enum import Enum
from pathlib import Path

import bmi.api as bmi


class Estimator(Enum):
    KSG_10 = "KSG-10"
    KSG_5 = "KSG-5"
    R_KSG_10 = "R-KSG-10"
    R_KSG_5 = "R-KSG-5"
    R_LNN_10 = "R-LNN-10"
    R_LNN_5 = "R-LNN-5"
    MINE = "MINE"
    HISTOGRAM_3 = "Histogram-3"
    HISTOGRAM_5 = "Histogram-5"
    CCA = "CCA"


def _load_mine() -> bmi.ITaskEstimator:
    import torch

    import bmi.estimators.external.mine as mine

    device = "gpu" if torch.cuda.is_available() else "cpu"

    return bmi.benchmark.WrappedEstimator(
        estimator_id="MINE",
        estimator=mine.MutualInformationNeuralEstimator(device=device),
    )


def create_estimator(estimator: Estimator) -> bmi.ITaskEstimator:  # noqa: C901
    # Silence the C901 linting error saying that this function is too complex.
    # It is indeed quite long and complex, but what else can we do?
    if estimator == Estimator.KSG_10:
        return bmi.benchmark.WrappedEstimator(
            estimator=bmi.estimators.KSGEnsembleFirstEstimator(neighborhoods=(10,)),
            estimator_id="KSG-10",
        )
    elif estimator == Estimator.KSG_5:
        return bmi.benchmark.WrappedEstimator(
            estimator=bmi.estimators.KSGEnsembleFirstEstimator(neighborhoods=(5,)),
            estimator_id="KSG-5",
        )
    elif estimator == Estimator.R_KSG_5:
        return bmi.benchmark.REstimatorKSG(neighbors=5)
    elif estimator == Estimator.R_KSG_10:
        return bmi.benchmark.REstimatorKSG(neighbors=10)
    elif estimator == Estimator.R_LNN_10:
        return bmi.benchmark.REstimatorLNN(neighbors=10)
    elif estimator == Estimator.R_LNN_5:
        return bmi.benchmark.REstimatorLNN(neighbors=5)
    elif estimator == Estimator.MINE:
        return _load_mine()
    elif estimator == Estimator.HISTOGRAM_3:
        return bmi.benchmark.WrappedEstimator(
            estimator_id="Histogram-3", estimator=bmi.estimators.HistogramEstimator(n_bins_x=3)
        )
    elif estimator == Estimator.HISTOGRAM_5:
        return bmi.benchmark.WrappedEstimator(
            estimator_id="Histogram-3", estimator=bmi.estimators.HistogramEstimator(n_bins_x=3)
        )
    elif estimator == Estimator.CCA:
        return bmi.benchmark.WrappedEstimator(
            estimator_id="CCA",
            estimator=bmi.estimators.CCAMutualInformationEstimator(),
        )
    else:
        raise ValueError(f"Estimator {estimator} not recognized.")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, help="Path to the task directory.")
    parser.add_argument(
        "--output", type=Path, help="Path to which the results YAML will be dumped."
    )
    parser.add_argument("--seed", type=int, help="Seed to load the right sample from the task.")

    def _to_estimator_enum(s: str) -> Estimator:
        for est in Estimator:
            if s == est.value:
                return est
        else:
            raise ValueError(f"Estimator {s} not recognized.")

    estimators_allowed = [est.value for est in Estimator]
    parser.add_argument(
        "--estimator",
        type=_to_estimator_enum,
        action="store",
        help=f"The estimator to be run. Available: {' '.join(estimators_allowed)}",
    )
    return parser


def main() -> None:
    args = create_parser().parse_args()

    estimator = create_estimator(args.estimator)
    result = estimator.estimate(task_path=args.task, seed=args.seed)
    bmi.benchmark.SaveLoadRunResults.dump(result, path=args.output)


if __name__ == "__main__":
    main()
