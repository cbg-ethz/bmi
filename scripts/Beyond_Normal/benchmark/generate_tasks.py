import argparse
from pathlib import Path

import bmi.api as bmi


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("TASKS", type=Path, help="Directory where the tasks will be dumped to.")
    parser.add_argument(
        "--version",
        type=int,
        help="Version of the benchmark to be dumped.",
        choices=[1],
        default=1,
    )
    parser.add_argument(
        "--n-seeds", type=int, help="The default number of seeds per task.", default=10
    )
    return parser


def main() -> None:
    args = create_parser().parse_args()

    tasks_dir = args.TASKS

    # Generating benchmark tasks
    print("Generating benchmark tasks...")
    tasks = bmi.benchmark.generate_benchmark(version=args.version, n_seeds=args.n_seeds)
    bmi.benchmark.save_benchmark_tasks(
        tasks=tasks,
        tasks_dir=tasks_dir,
        exist_ok=True,
    )
    print(f"Tasks generated and saved to {args.TASKS}.")


if __name__ == "__main__":
    main()
