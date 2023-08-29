import bmi.api as bmi


def run_julia() -> None:
    args = ["julia", "external/mi_estimator.jl"]
    taskdir = "testdir"

    output = bmi.benchmark.run_external_estimator(
        taskdir, seed=1, command_args=args, estimator_id="julia-kraskov-1"
    )

    print(output)


def run_estimators(estimators, taskdir: str, seed: int = 1) -> None:
    for estimator in estimators:
        output = estimator.estimate(taskdir, seed=seed)
        print(output)


def main() -> None:
    external_estimators = [
        bmi.benchmark.REstimatorKSG(variant=1),
        bmi.benchmark.REstimatorKSG(variant=2),
        bmi.benchmark.REstimatorLNN(neighbors=5),
    ]

    run_estimators(external_estimators, taskdir="testdir", seed=1)


if __name__ == "__main__":
    main()
