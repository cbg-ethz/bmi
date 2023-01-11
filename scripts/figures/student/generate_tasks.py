"""Script used to generate multivariate Student-t tasks
with varying degrees of freedom."""
import argparse
from pathlib import Path

import numpy as np

import bmi.api as bmi


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("DIRECTORY", type=Path, help="Path to which the tasks will be dumped.")
    parser.add_argument("--n", type=int, default=5000, help="Number of points.")
    parser.add_argument(
        "--strength",
        type=float,
        default=0.8,
        help="Strength of the Gaussian interaction between X and Y.",
    )
    parser.add_argument(
        "--dof",
        metavar="DoF",
        type=int,
        nargs="+",
        help="List of degrees of freedom parameters.",
        default=[1, 2, 3, 5, 10, 30],
    )
    parser.add_argument("--seed", type=int, default=10, help="Number of seeds.")
    return parser


def generate_task(
    strength: float,
    df: int,
    n_samples: int,
    n_seeds: int,
) -> bmi.Task:
    """We have dim_x = 2, dim_y = 1
    and
        dispersion(X1, Y) = strength
        dispersion(X2, Y) = 0
        dispersion(X1, X2) = 0
    """
    dispersion = np.eye(3)
    dispersion[0, 2] = strength
    dispersion[2, 0] = strength

    sampler = bmi.samplers.SplitStudentT(
        dim_x=2,
        dim_y=1,
        dispersion=dispersion,
        df=df,
    )
    task_params = {
        "mi_gaussian": sampler.mi_normal(),
        "mi_correction": sampler.mi_correction(),
        "mi_true": sampler.mutual_information(),
        "degrees_of_freedom": df,
        "strength": strength,
    }
    task_id = f"student-df_{df}-strength_{strength:.4}"

    return bmi.benchmark.generate_task(
        sampler=sampler,
        n_samples=n_samples,
        seeds=range(n_seeds),
        task_id=task_id,
        task_params=task_params,
    )


def generate_task_suite(
    strength: float,
    n_samples: int,
    n_seeds: int,
    dfs: list[int],
):
    for df in dfs:
        yield generate_task(strength=strength, n_samples=n_samples, n_seeds=n_seeds, df=df)


def main() -> None:
    args = create_parser().parse_args()
    bmi.benchmark.save_benchmark_tasks(
        tasks_dir=args.DIRECTORY,
        tasks=generate_task_suite(
            strength=args.strength,
            n_samples=args.n,
            n_seeds=args.seed,
            dfs=args.dof,
        ),
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
