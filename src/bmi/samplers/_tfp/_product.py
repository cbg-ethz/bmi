from tensorflow_probability.substrates import jax as tfp

from bmi.samplers._tfp._core import JointDistribution

jtf = tfp.tf2jax
tfd = tfp.distributions


class ProductDistribution(JointDistribution):
    """From distributions P_X and P_Y creates a distribution

        P_{XY} = P_X x P_Y

    in which the variables X and Y are independent.

    In particular,

        I(X; Y) = 0

    under this distribution.
    """

    def __init__(self, dist_x: tfd.Distribution, dist_y: tfd.Distribution) -> None:
        dims_x = dist_x.event_shape_tensor()
        dims_y = dist_y.event_shape_tensor()

        assert len(dims_x) == 1
        assert len(dims_y) == 1

        dim_x = int(dims_x[0])
        dim_y = int(dims_y[0])

        dist_joint = tfd.Blockwise([dist_x, dist_y])

        super().__init__(
            dim_x=dim_x,
            dim_y=dim_y,
            dist_joint=dist_joint,
            dist_x=dist_x,
            dist_y=dist_y,
            analytic_mi=0.0,
        )
