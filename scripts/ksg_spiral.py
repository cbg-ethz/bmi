import numpy as np

import bmi.api as bmi
import bmi.transforms.rotate as rot


def generate_covariance(correlation: float, dim_x: int, dim_y: int) -> np.ndarray:
    """The correlation between the first dimension of X and the first dimension of Y is fixed.

    The rest of the covariance entries are zero.
    """
    covariance = np.eye(dim_x + dim_y)
    covariance[0, dim_x] = correlation
    covariance[dim_x, 0] = correlation
    return covariance


def main() -> None:
    dim_x = 5
    dim_y = 2
    rho = 0.5
    n_points = 5000
    seed = 42

    assert 0 <= rho < 1

    covariance = generate_covariance(dim_x=dim_x, dim_y=dim_y, correlation=rho)

    base_sampler = bmi.samplers.SplitMultinormal(
        dim_x=dim_x,
        dim_y=dim_y,
        covariance=covariance,
    )

    x_normal, y_normal = base_sampler.sample(n_points, rng=seed)
    mi_true = base_sampler.mutual_information()

    mi_estimate_normal = bmi.estimators.KSGEnsembleFirstEstimator().estimate(x_normal, y_normal)

    print(f"{mi_true = :.3f}\t{mi_estimate_normal = :.3f}")

    generator = rot.so_generator(dim_x, i=0, j=1)

    for speed in [0.0, 0.02, 0.1, 0.5, 1.0, 10.0]:
        transform_x = rot.Spiral(generator=generator, speed=speed)
        transformed_sampler = bmi.samplers.TransformedSampler(
            base_sampler=base_sampler, transform_x=transform_x
        )

        x_transformed, y_transformed = transformed_sampler.transform(x_normal, y_normal)

        mi_estimate_transformed = bmi.estimators.KSGEnsembleFirstEstimator().estimate(
            x_transformed, y_transformed
        )

        print(f"{speed = :.2f}\t {mi_estimate_transformed = :.3f}")


if __name__ == "__main__":
    main()
