"""This script loads a given estimator and runs it on a given task."""
import argparse
from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Protocol, cast

import bmi.api as bmi


class EstimatorType(Enum):
    KSG = "KSG"
    R_KSG = "R-KSG"
    R_LNN = "R-LNN"
    MINE = "MINE"
    HISTOGRAM = "HISTOGRAM"
    CCA = "CCA"


def _load_mine(
    device: Literal["cpu", "gpu", "auto"], estimator_id: Optional[str], max_epochs: int
) -> bmi.ITaskEstimator:
    import torch

    import bmi.estimators.external.mine as mine

    if device == "auto":
        device = "gpu" if torch.cuda.is_available() else "cpu"

    return bmi.benchmark.WrappedEstimator(
        estimator=mine.MutualInformationNeuralEstimator(device=device, max_epochs=max_epochs),
        estimator_id=estimator_id,
    )


class Args(Protocol):
    estimator: EstimatorType
    estimator_id: Optional[str]
    neighbors: int  # For kNN-based methods
    truncation: int  # A parameter for LNN
    metric: Literal[
        "euclidean", "manhattan", "chebyshev"
    ]  # Metric for Python implementation of KSG
    bins_x: int  # Bins per X dimension for histogram
    bins_y: Optional[int]  # Bins per Y dimension for histogram. If None, defaults to bins_x
    variant: Literal[1, 2]
    device: Literal["cpu", "gpu", "auto"]
    max_epochs: int


def create_estimator(args: Args) -> bmi.ITaskEstimator:  # noqa: C901
    # Silence the C901 linting error saying that this function is too complex.
    # It is indeed quite long and complex, but what else can we do?
    estimator: EstimatorType = args.estimator
    if estimator == EstimatorType.KSG:
        ksg_estimator = bmi.estimators.KSGEnsembleFirstEstimator(
            neighborhoods=(args.neighbors,),
            metric_x=args.metric,
            metric_y=args.metric,
        )
        return bmi.benchmark.WrappedEstimator(
            estimator=ksg_estimator, estimator_id=args.estimator_id
        )
    elif estimator == EstimatorType.R_KSG:
        return bmi.benchmark.REstimatorKSG(
            neighbors=args.neighbors, variant=args.variant, estimator_id=args.estimator_id
        )
    elif estimator == EstimatorType.R_LNN:
        return bmi.benchmark.REstimatorLNN(
            neighbors=args.neighbors,
            truncation=args.truncation,
            estimator_id=args.estimator_id,
        )
    elif estimator == EstimatorType.MINE:
        return _load_mine(
            device=args.device, estimator_id=args.estimator_id, max_epochs=args.max_epochs
        )
    elif estimator == EstimatorType.HISTOGRAM:
        histogram_estimator = bmi.estimators.HistogramEstimator(
            n_bins_x=args.bins_x,
            n_bins_y=args.bins_y,
        )
        return bmi.benchmark.WrappedEstimator(
            estimator=histogram_estimator,
            estimator_id=args.estimator_id,
        )
    elif estimator == EstimatorType.CCA:
        return bmi.benchmark.WrappedEstimator(
            estimator=bmi.estimators.CCAMutualInformationEstimator(),
            estimator_id=args.estimator_id,
        )
    else:
        raise ValueError(f"Estimator {estimator} not recognized.")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--TASK", type=Path, help="Path to the task directory.")
    parser.add_argument(
        "--OUTPUT", type=Path, help="Path to which the results YAML will be dumped."
    )
    parser.add_argument("--SEED", type=int, help="Seed to load the right sample from the task.")

    def _to_estimator_enum(s: str) -> EstimatorType:
        for est in EstimatorType:
            if s == est.value:
                return est
        else:
            raise ValueError(f"Estimator {s} not recognized.")

    estimators_allowed = [est.value for est in EstimatorType]

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If present, an estimator will be created, but not run.",
    )

    # Arguments for kNN-based estimators
    parser.add_argument(
        "--estimator",
        type=_to_estimator_enum,
        action="store",
        help=f"The estimator to be run. Available: {' '.join(estimators_allowed)}",
        default=EstimatorType.KSG,
    )
    parser.add_argument(
        "--estimator-id",
        type=str,
        help="Custom `estimator_id`. If left unspecified, will be generated automatically.",
        default=None,
    )

    parser.add_argument(
        "--neighbors",
        type=int,
        default=5,
        help="Number of neighbors for kNN graph methods: " "both versions of KSG and LNN.",
    )
    parser.add_argument(
        "--variant",
        type=int,
        choices=[1, 2],
        help="The variant of the R KSG estimator to be used.",
    )
    parser.add_argument("--truncation", type=int, default=15, help="Truncation parameter of LNN.")
    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "manhattan", "chebyshev"],
        help="For the Python implementation of KSG we can specify the metric "
        "to be used (on both spaces).",
    )

    # Arguments for histogram-based estimators
    parser.add_argument(
        "--bins-x", type=int, default=5, help="Number of bins on each dimension of the X variable."
    )
    parser.add_argument(
        "--bins-y",
        type=int,
        default=None,
        help="Number of bins on each dimension of the Y variable." "Defaults to `--bins-x`.",
    )

    # Arguments for MINE
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu", "auto"],
        default="auto",
        help="Device to be used for MINE training. "
        "Defaults to 'auto', which will try to use CUDA if it's available.",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=300, help="Maximum number of epochs of MINE training."
    )

    return parser


def _check_seed_in_task(task_path: Path, seed: int) -> None:
    """Checks if the specified seed is in the task.
    If not, raises an exception."""
    task = bmi.Task.load(task_path)
    if seed not in task.keys():
        raise ValueError(f"Seed {seed} was not found in task at {task_path}.")


def main() -> None:
    args = create_parser().parse_args()

    estimator = create_estimator(cast(Args, args))

    # If it's a dry run, we will just check if there's the right seed in the task
    if args.dry_run:
        _check_seed_in_task(task_path=args.TASK, seed=args.SEED)
    # If it's not a dry run, we will run the actual estimator and save the result
    else:
        result = estimator.estimate(task_path=args.TASK, seed=args.SEED)
        bmi.benchmark.SaveLoadRunResults.dump(result, path=args.OUTPUT)


if __name__ == "__main__":
    main()
