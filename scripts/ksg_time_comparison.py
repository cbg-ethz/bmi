"""Simple comparison of the speed of our KSG estimators."""
import argparse
import time

import numpy as np
from jax import random

import bmi.api as bmi


class TimeIt:
    """Context manager used to time a given procedure.

    Example:
        The code:
        ```
        import time
        with TimeIt(message_start="Starting the sleep procedure...", message_end="It took {}."):
            time.sleep(10)
        ```
        will produce to standard output
        ```
        Starting the sleep procedure...
        It took 10 seconds.
        ```
    """

    def __init__(self, message_start: str = "", message_end: str = "{}") -> None:
        """

        Args:
            message_start: message to print at the beginning. Set to empty (default) for no message
            message_end: format string to be printed at the end of the procedure.
              Should contain "{}" which will be filled with the time.
        """
        if message_start:
            print(message_start)
        self.message_end = message_end
        self.t0 = time.time()

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        delta_t = time.time() - self.t0
        message_fill = "{:.2f} seconds".format(delta_t)
        print(self.message_end.format(message_fill))


SLOW = "SLOW"
NUMPY = "NUMPY"

KSG_ESTIMATORS = {
    SLOW: bmi.estimators.KSGEnsembleFirstEstimatorSlow,
    NUMPY: bmi.estimators.KSGEnsembleFirstEstimator,
}


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ksg-type",
        default="NUMPY",
        choices=KSG_ESTIMATORS.keys(),
        help="Which KSG estimator to use.",
    )
    parser.add_argument("--n-samples", default=1000, type=int, help="Number of samples.")
    parser.add_argument("--dim-x", default=2, type=int, help="Dimension of the X variable.")
    parser.add_argument(
        "--dim-y",
        default=None,
        type=int,
        help="Dimension of the Y variable. "
        "By default it is equal to the dimension of the X variable.",
    )
    parser.add_argument("--neighborhood", default=5, type=int, help="Neighborhood size.")
    parser.add_argument("--n-jobs", default=1, type=int, help="Number of launched jobs.")
    parser.add_argument(
        "--chunk-size", default=10, type=int, help="Chunk size for the NumPy-based KSG estimator."
    )

    return parser


def get_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = parser.parse_args()

    assert args.neighborhood >= 1
    assert args.n_samples >= 1
    assert args.chunk_size >= 1

    assert args.dim_x >= 1
    # By default, we set dim_y to dim_x
    args.dim_y = args.dim_y or args.dim_x

    return args


def get_estimator(args: argparse.Namespace):
    if args.ksg_type == SLOW:
        return bmi.estimators.KSGEnsembleFirstEstimatorSlow(
            neighborhoods=(args.neighborhood,),
            n_jobs=args.n_jobs,
        )
    elif args.ksg_type == NUMPY:
        return bmi.estimators.KSGEnsembleFirstEstimator(
            neighborhoods=(args.neighborhood,),
            n_jobs=args.n_jobs,
            chunk_size=args.chunk_size,
        )
    else:
        raise ValueError(f"KSG estimator type {args.ksg_type} not known.")


def main() -> None:
    parser = create_parser()
    args = get_args(parser)

    print("Settings:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")

    dim_total = args.dim_x + args.dim_y

    with TimeIt(message_start="Creating samples...", message_end="\tTook {}."):
        sampler = bmi.samplers.SplitMultinormal(
            dim_x=args.dim_x,
            dim_y=args.dim_y,
            mean=np.zeros(dim_total),
            covariance=np.eye(dim_total),
        )
        x, y = sampler.sample(n_points=args.n_samples, rng=random.PRNGKey(10))

    estimator = get_estimator(args)

    with TimeIt(message_start="Estimating MI...", message_end="\tTook {}."):
        mi = estimator.estimate(x, y)

    print(f"Calculated MI: {mi}")
    print(f"True MI: {sampler.mutual_information()}")


if __name__ == "__main__":
    main()
