import bmi.api as bmi


def main() -> None:
    args = ["julia", "external/mi_estimator.jl"]
    taskdir = "testdir"

    output = bmi.benchmark.run_external_estimator(
        taskdir, seed=1, command_args=args, estimator_id="julia-kraskov-1"
    )

    print(output)


if __name__ == "__main__":
    main()
