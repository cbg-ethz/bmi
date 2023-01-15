"""Generates tasks and saves them in the specified location."""
import argparse
from pathlib import Path
from typing import Iterable, Sequence

import bmi.api as bmi


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("DIRECTORY", type=Path, help="Path to which the tasks will be dumped.")
    parser.add_argument("--n", type=int, default=5000, help="Number of points.")
    parser.add_argument("--correlation", type=float, default=0.8, help="Correlation.")
    parser.add_argument(
        "--speed",
        metavar="SPEED",
        type=float,
        nargs="+",
        help="List of speed parameters for the spiral diffeomorphisms.",
        default=[0, 0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 2.5, 3],
    )
    parser.add_argument("--seed", type=int, default=5, help="Number of seeds.")
    return parser


def generate_spiral_invariance_task_suite(
    correlation: float,
    n_points: int,
    n_seeds: int,
    speed_list: Iterable[float],
) -> Sequence[bmi.Task]:
    return [
        bmi.benchmark.generate_spiral_invariance_task(
            correlation=correlation,
            n_points=n_points,
            speed=speed,
            n_seeds=n_seeds,
            task_id=f"spiral-speed_{speed:.4f}",
        )
        for speed in speed_list
    ]


def main() -> None:
    args = create_parser().parse_args()
    tasks = generate_spiral_invariance_task_suite(
        correlation=args.correlation,
        n_points=args.n,
        speed_list=args.speed,
        n_seeds=args.seed,
    )
    bmi.benchmark.save_benchmark_tasks(
        tasks_dir=args.DIRECTORY,
        tasks=tasks,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
