import argparse
from pathlib import Path
from typing import Sequence

import bmi.api as bmi

default_n_points = [500, 1000, 5000]
default_correlations = [0.0, 0.5, 0.8, 0.9, 0.99]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("DIRECTORY", type=Path, help="Path to which the tasks will be dumped.")
    parser.add_argument(
        "--n",
        metavar="N1 N2",
        type=float,
        nargs="+",
        help="List of number of points.",
        default=default_n_points,
    )
    parser.add_argument(
        "--correlation",
        metavar="CORR1 CORR2",
        type=float,
        nargs="+",
        help="List of correlations.",
        default=default_correlations,
    )
    parser.add_argument("--seed", type=int, default=10, help="Number of seeds.")
    return parser


def generate_task(correlation: float, n_points: int, n_seeds: int) -> bmi.Task:
    sampler = bmi.samplers.BivariateNormalSampler(correlation=correlation)
    return bmi.benchmark.generate_task(
        sampler=sampler,
        n_samples=n_points,
        seeds=range(n_seeds),
        task_id=f"high_mi-corr_{correlation:.4f}-points_{n_points}",
        task_params={
            "correlation": correlation,
            "points": n_points,
        },
    )


def generate_tasks(
    correlations: Sequence[float],
    n_points: Sequence[int],
    n_seeds: int,
) -> list[bmi.Task]:
    task_list = []
    for corr in correlations:
        for n in n_points:
            task = generate_task(correlation=corr, n_points=n, n_seeds=n_seeds)
            task_list.append(task)
    return task_list


def main() -> None:
    args = create_parser().parse_args()
    print(f"Script running with arguments: {args.__dict__}")

    tasks = generate_tasks(
        correlations=args.correlation,
        n_points=args.n,
        n_seeds=args.seed,
    )
    print(f"Generated {len(tasks)}:")
    for task in tasks:
        print(f"  - {task.task_id}")
    print(f"Saving the tasks to {args.DIRECTORY}...")
    bmi.benchmark.save_benchmark_tasks(
        tasks_dir=args.DIRECTORY,
        tasks=tasks,
        exist_ok=True,
    )
    print("Run finished!")


if __name__ == "__main__":
    main()
