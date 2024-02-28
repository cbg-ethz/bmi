"""Subpackage with different samplers, which can be used to define benchmark tasks
or used directly to sample from distributions with known mutual information."""

from bmi.samplers._additive_uniform import AdditiveUniformSampler

# isort: off
from bmi.samplers._matrix_utils import (
    parametrised_correlation_matrix,
    canonical_correlation,
    DenseLVMParametrization,
    SparseLVMParametrization,
    GaussianLVMParametrization,
)

# isort: on
import bmi.samplers._tfp as fine
from bmi.samplers._independent_coordinates import IndependentConcatenationSampler
from bmi.samplers._split_student_t import SplitStudentT
from bmi.samplers._splitmultinormal import BivariateNormalSampler, SplitMultinormal
from bmi.samplers._transformed import TransformedSampler
from bmi.samplers.base import BaseSampler

# isort: off
from bmi.samplers._discrete_continuous import (
    DiscreteUniformMixtureSampler,
    MultivariateDiscreteUniformMixtureSampler,
    ZeroInflatedPoissonizationSampler,
)

# isort: on

__all__ = [
    "AdditiveUniformSampler",
    "BaseSampler",
    "canonical_correlation",
    "fine",
    "parametrised_correlation_matrix",
    "BivariateNormalSampler",
    "SplitMultinormal",
    "SplitStudentT",
    "TransformedSampler",
    "DenseLVMParametrization",
    "SparseLVMParametrization",
    "GaussianLVMParametrization",
    "IndependentConcatenationSampler",
    "DiscreteUniformMixtureSampler",
    "MultivariateDiscreteUniformMixtureSampler",
    "ZeroInflatedPoissonizationSampler",
]
