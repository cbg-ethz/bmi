import bmi.api as bmi


def run_julia() -> None:
    args = ["julia", "external/mi_estimator.jl"]
    taskdir = "testdir"

    output = bmi.benchmark.run_external_estimator(
        taskdir, seed=1, command_args=args, estimator_id="julia-kraskov-1"
    )

    print(output)


def run_r_ksg() -> None:
    taskdir = "testdir"

    estimator = bmi.benchmark.REstimatorKSG()
    output = estimator.estimate(taskdir, seed=1)
    print(output)


def main() -> None:
    run_r_ksg()


if __name__ == "__main__":
    main()
