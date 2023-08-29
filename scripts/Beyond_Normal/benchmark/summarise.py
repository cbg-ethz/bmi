import argparse
from pathlib import Path

import pandas as pd

import bmi.api as bmi


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("TASKS", type=Path, help="Directory with the benchmark tasks.")
    parser.add_argument("RESULTS", type=Path, help="Directory with the run results.")
    parser.add_argument(
        "--results-csv",
        type=Path,
        help="Path to which resulting CSV will be dumped.",
        default=Path("benchmark-results.csv"),
    )
    parser.add_argument(
        "--stats-csv",
        type=Path,
        help="Path to which summary CSV will be dumped.",
        default=Path("benchmark-stats.csv"),
    )
    parser.add_argument(
        "--tasks-csv",
        type=Path,
        help="Path to which the tasks will be dumped",
        default=Path("benchmark-tasks.csv"),
    )
    return parser


def pydantic_to_dict(x) -> dict:
    return x.dict()


def main() -> None:
    args = create_parser().parse_args()

    # Dictionary indexed by task_id
    # Can be queried for task metadata, as true MI
    task_metadata: dict[str, bmi.TaskMetadata] = bmi.benchmark.LoadTaskMetadata.from_directory(
        args.TASKS
    )

    # Save the CSV
    pd.DataFrame(map(pydantic_to_dict, task_metadata.values())).to_csv(args.tasks_csv, index=False)

    # List of RunResults
    results = bmi.benchmark.SaveLoadRunResults.from_directory(args.RESULTS)

    results_df = pd.DataFrame(map(pydantic_to_dict, results))
    results_df.to_csv(args.results_csv, index=False)

    # Now do Pandas magic to have a nice table
    # TODO(Frederic): add column true MI!!!
    # TODO(Frederic, Pawel): idea we can have results.csv
    #  (detailed, not reduced) and results.html (pretty plots and means)
    interesting_cols = ["mi_estimate", "time_in_seconds"]
    means = (
        results_df.groupby(["task_id", "estimator_id"])[interesting_cols]
        .mean(numeric_only=True)
        .rename(columns=lambda x: x + "_mean")
    )

    stds = (
        results_df.groupby(["task_id", "estimator_id"])[interesting_cols]
        .std(numeric_only=True)
        .rename(columns=lambda x: x + "_std")
    )
    stats = means.join(stds)
    stats.to_csv(args.stats_csv)
    print(stats)


if __name__ == "__main__":
    main()
