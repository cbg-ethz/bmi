"""Tests whether KSG estimator is invariant to the "spiral" diffeomorphism."""
import argparse

import numpy as np

import bmi.api as bmi


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment with applying the spiral diffeomorphism."
    )
    parser.add_argument("--dim-x", type=int, default=3, help="Dimension of the X variable.")
    parser.add_argument("--dim-y", type=int, default=2, help="Dimension of the Y variable.")
    parser.add_argument("--rho", type=float, default=0.8, help="Correlation, between -1 and 1.")
    parser.add_argument(
        "--n-points", type=int, default=5000, help="Number of points to be generated."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser


def generate_covariance(correlation: float, dim_x: int, dim_y: int) -> np.ndarray:
    """The correlation between the first dimension of X and the first dimension of Y is fixed.

    The rest of the covariance entries are zero,
    of course except for variance of each dimension (the diagonal), which is 1.
    """
    covariance = np.eye(dim_x + dim_y)
    covariance[0, dim_x] = correlation
    covariance[dim_x, 0] = correlation
    return covariance


def create_base_sampler(dim_x: int, dim_y: int, rho: float) -> bmi.samplers.SplitMultinormal:
    assert -1 <= rho < 1

    covariance = generate_covariance(dim_x=dim_x, dim_y=dim_y, correlation=rho)

    return bmi.samplers.SplitMultinormal(
        dim_x=dim_x,
        dim_y=dim_y,
        covariance=covariance,
    )


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    print(f"Settings:\n{args}")

    base_sampler = create_base_sampler(dim_x=args.dim_x, dim_y=args.dim_y, rho=args.rho)

    x_normal, y_normal = base_sampler.sample(args.n_points, rng=args.seed)
    mi_true = base_sampler.mutual_information()

    mi_estimate_normal = bmi.estimators.KSGEnsembleFirstEstimator().estimate(x_normal, y_normal)

    print(f"True MI: {mi_true:.3f}")
    print(f"KSG(X; Y) without distortion: {mi_estimate_normal:.3f}")

    print("-------------------")
    print("speed\tKSG(spiral(X); Y)")

    generator = bmi.transforms.so_generator(args.dim_x, i=0, j=1)

    for speed in [0.0, 0.02, 0.1, 0.5, 1.0, 10.0]:
        transform_x = bmi.transforms.Spiral(generator=generator, speed=speed)
        transformed_sampler = bmi.samplers.TransformedSampler(
            base_sampler=base_sampler, transform_x=transform_x
        )

        x_transformed, y_transformed = transformed_sampler.transform(x_normal, y_normal)

        mi_estimate_transformed = bmi.estimators.KSGEnsembleFirstEstimator().estimate(
            x_transformed, y_transformed
        )

        print(f"{speed:.2f}\t {mi_estimate_transformed:.3f}")


if __name__ == "__main__":
    main()
