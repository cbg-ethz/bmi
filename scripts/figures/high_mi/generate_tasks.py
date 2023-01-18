import argparse
from typing import Callable, Iterable

import numpy as np

import bmi.api as bmi
from bmi.benchmark.tasks.one_dimensional import half_cube


def binsearch(
    func: Callable[[float], float],
    target: float,
    min_value: float,
    max_value: float,
    delta: float = 1e-4,
    max_iter: int = 5_000,
    seed: int = 0,
) -> float:
    """Solves func(x) = target for x assuming that func is strictly increasing
    and using a mixture of binary search and random sampling."""
    rng = np.random.default_rng(seed)

    min_val = min_value
    max_val = max_value
    mid = 0.5 * (min_val + max_val)
    value = 0

    for _ in range(max_iter):
        try:
            value = func(mid)

            if abs(value - target) < delta:
                break
            elif value > target:
                max_val = mid
            else:
                min_val = mid

            mid = 0.5 * (max_val + min_val)
        except ValueError:
            mid = rng.uniform(min_value, max_value)
    if abs(value - target) < delta:
        return mid
    else:
        raise Exception("The desired accuracy has not been reached.")


# *** Sparse Gaussian task factory ***
def get_sparse_gaussian_sampler(correlation: float, dim: int, noise: float = 0.1) -> bmi.ISampler:
    """Generates the sparse Gaussian sampler we will use."""
    covariance = bmi.samplers.parametrised_correlation_matrix(
        dim_x=dim,
        dim_y=dim,
        k=2,
        correlation=correlation,
        correlation_x=noise,
        correlation_y=noise,
    )
    return bmi.samplers.SplitMultinormal(
        dim_x=dim,
        dim_y=dim,
        covariance=covariance,
    )


def mi_sparse_gaussian(correlation: float, dim: int, noise: float = 0.1) -> float:
    """The mutual information of the sparse Gaussian sampler."""
    sampler = get_sparse_gaussian_sampler(correlation=correlation, dim=dim, noise=noise)
    return sampler.mutual_information()


def generate_sparse_gaussian_task(
    mi: float,
    dim: int,
    noise: float = 0.1,
    n_seeds: int = 5,
    n_samples: int = 5000,
    prefix: str = "",
) -> bmi.Task:
    corr = binsearch(
        lambda c: mi_sparse_gaussian(c, dim=dim, noise=noise),
        target=mi,
        min_value=0,
        max_value=1,
    )

    sampler = get_sparse_gaussian_sampler(correlation=corr, dim=dim, noise=noise)

    return bmi.benchmark.generate_task(
        sampler=sampler,
        task_params=dict(
            correlation=corr,
            desired_mi=mi,
            actual_mi=sampler.mutual_information(),
            dim=dim,
            noise=noise,
        ),
        n_samples=n_samples,
        seeds=range(n_seeds),
        task_id=f"{prefix}sparse-gaussian-{dim}_dim-{mi}_mi-{n_samples}_samples",
    )


# *** Half-cubed Gaussian sampler ***


def generate_half_cube_task(
    mi: float, dim: int, noise: float = 0.1, n_seeds: int = 5, n_samples: int = 5000
) -> bmi.Task:
    corr = binsearch(
        lambda c: mi_sparse_gaussian(c, dim=dim, noise=noise),
        target=mi,
        min_value=0,
        max_value=1,
    )

    base_sampler = get_sparse_gaussian_sampler(correlation=corr, dim=dim, noise=noise)
    sampler = bmi.samplers.TransformedSampler(
        base_sampler=base_sampler,
        transform_x=half_cube,
        transform_y=half_cube,
        vectorise=True,
    )

    return bmi.benchmark.generate_task(
        sampler=sampler,
        task_params=dict(
            correlation=corr,
            desired_mi=mi,
            actual_mi=sampler.mutual_information(),
            dim=dim,
            noise=noise,
        ),
        n_samples=n_samples,
        seeds=range(n_seeds),
        task_id=f"half-cubed-{dim}_dim-{mi}_mi-{n_samples}_samples",
    )


# *** Spiraled Gaussian sampler ***


def generate_spiral_task(
    mi: float,
    dim: int,
    noise: float = 0.1,
    n_seeds: int = 5,
    n_samples: int = 5000,
    speed: float = 1 / 3,
) -> bmi.Task:
    corr = binsearch(
        lambda c: mi_sparse_gaussian(c, dim=dim, noise=noise),
        target=mi,
        min_value=0,
        max_value=1,
    )

    base_sampler = get_sparse_gaussian_sampler(correlation=corr, dim=dim, noise=noise)
    sampler = bmi.benchmark.tasks.diffeo.create_spiral_sampler(
        dim=dim, base_sampler=base_sampler, speed=speed
    )

    return bmi.benchmark.generate_task(
        sampler=sampler,
        task_params=dict(
            correlation=corr,
            desired_mi=mi,
            actual_mi=sampler.mutual_information(),
            dim=dim,
            noise=noise,
            speed=speed,
        ),
        n_samples=n_samples,
        seeds=range(n_seeds),
        task_id=f"spiral-{dim}_dim-{mi}_mi-{n_samples}_samples",
    )


# *** Sparse Student-t sampler ***


def get_sparse_student_sampler(
    signal: float, dim: int, df: int = 5, noise: float = 0.1
) -> bmi.ISampler:
    dispersion = bmi.samplers.parametrised_correlation_matrix(
        dim_x=dim,
        dim_y=dim,
        k=2,
        correlation=signal,
        correlation_x=noise,
        correlation_y=noise,
    )

    return bmi.samplers.SplitStudentT(
        dim_x=dim,
        dim_y=dim,
        dispersion=dispersion,
        df=df,
    )


def mi_sparse_student(signal: float, dim: int, df: int = 5, noise: float = 0.1) -> float:
    sampler = get_sparse_student_sampler(signal=signal, dim=dim, df=df, noise=noise)
    return sampler.mutual_information()


def generate_sparse_student_task(
    mi: float,
    dim: int,
    df: int,
    noise: float = 0.1,
    n_seeds: int = 5,
    n_samples: int = 5_000,
) -> bmi.Task:
    signal = binsearch(
        lambda c: mi_sparse_student(c, df=df, dim=dim, noise=noise),
        target=mi,
        min_value=0,
        max_value=1,
    )

    sampler = get_sparse_student_sampler(signal=signal, df=df, dim=dim, noise=noise)

    return bmi.benchmark.generate_task(
        sampler=sampler,
        task_params=dict(
            signal=signal,
            desired_mi=mi,
            actual_mi=sampler.mutual_information(),
            dim=dim,
            noise=noise,
            df=df,
        ),
        n_samples=n_samples,
        seeds=range(n_seeds),
        task_id=f"sparse-student-{dim}_dim-{mi}_mi-{df}_df-{n_samples}_samples",
    )


# *** The main function ***


MI_DESIRED = (0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)


def generate_tasks_mi(  # noqa: C901
    n_seeds: int, n_samples: int, mi_desired: Iterable[float] = MI_DESIRED
) -> Iterable[bmi.Task]:
    for mi in mi_desired:
        try:
            yield generate_sparse_gaussian_task(mi=mi, n_samples=n_samples, n_seeds=n_seeds, dim=3)
        except Exception as e:
            print(
                f"Exception in the sparse Gaussian task with "
                f"{mi =}, {n_samples =}, dim=3 due to {e}."
            )
        try:
            yield generate_half_cube_task(mi=mi, n_samples=n_samples, n_seeds=n_seeds, dim=3)
        except Exception as e:
            print(f"Exception in the half-cubed " f"{mi =}, {n_samples =}, dim=3 due to {e}.")
        try:
            yield generate_sparse_student_task(
                mi=mi, n_samples=n_samples, n_seeds=n_seeds, df=7, dim=3
            )
        except Exception as e:
            print(
                f"Exception in the sparse Student task "
                f"with {mi =}, {n_samples =}, df=7, dim=3 due to {e}."
            )
        try:
            yield generate_spiral_task(mi=mi, n_samples=n_samples, n_seeds=n_seeds, dim=3)
        except Exception as e:
            print(
                f"Exception in the spiral task " f"with {mi =}, {n_samples =}, dim=3 due to {e}."
            )


N_SAMPLES = (500, 1000, 2000, 5000, 10_000)


def generate_tasks_samples(  # noqa: C901
    n_seeds: int, mi: float, n_samples: Iterable[int] = N_SAMPLES
) -> Iterable[bmi.Task]:
    for n in n_samples:
        try:
            yield generate_sparse_gaussian_task(mi=mi, n_samples=n, n_seeds=n_seeds, dim=3)
        except Exception as e:
            print(
                f"Exception in the sparse Gaussian task "
                f"with {n =}, {n_samples =}, dim=3 due to {e}."
            )
        try:
            yield generate_half_cube_task(mi=mi, n_samples=n, n_seeds=n_seeds, dim=3)
        except Exception as e:
            print(f"Exception in the half-cubed " f"{mi =}, {n =}, dim=3 due to {e}.")
        try:
            yield generate_sparse_student_task(mi=mi, n_samples=n, n_seeds=n_seeds, df=7, dim=3)
        except Exception as e:
            print(
                f"Exception in the sparse Student task "
                f"with {mi =}, {n =}, df=7, dim=3 due to {e}."
            )
        try:
            yield generate_spiral_task(mi=mi, n_samples=n, n_seeds=n_seeds, dim=3)
        except Exception as e:
            print(f"Exception in the spiral task " f"with {mi =}, {n =}, dim=3 due to {e}.")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("TASKS", help="Where the tasks should be saved to.")
    parser.add_argument("--CHANGE", type=str, choices=["MI", "SAMPLES"], help="Whether ")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds.")
    parser.add_argument(
        "--samples",
        type=int,
        default=5_000,
        help="Number of samples in each task (unless is changing).",
    )
    parser.add_argument("--mi", type=float, help="Desired MI (unless is changing).", default=1.5)
    return parser


def main() -> None:
    args = create_parser().parse_args()

    if args.CHANGE == "MI":
        tasks = generate_tasks_mi(n_seeds=args.seeds, n_samples=args.samples)
    elif args.CHANGE == "SAMPLES":
        tasks = generate_tasks_samples(n_seeds=args.seeds, mi=args.mi)
    else:
        raise ValueError(f"{args.CHANGE} not recognized.")

    bmi.benchmark.save_benchmark_tasks(
        tasks_dir=args.TASKS,
        tasks=tasks,
    )


if __name__ == "__main__":
    main()
