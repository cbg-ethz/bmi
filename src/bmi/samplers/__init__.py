"""Subpackage with different samplers, which can be used to define benchmark tasks
or used directly to sample from distributions with known mutual information."""
from bmi.samplers._additive_uniform import AdditiveUniformSampler

# isort: off
from bmi.samplers._matrix_utils import (
    parametrised_correlation_matrix,
    DenseLVMParametrization,
    SparseLVMParametrization,
    GaussianLVMParametrization,
)

# isort: on
from bmi.samplers._split_student_t import SplitStudentT
from bmi.samplers._splitmultinormal import BivariateNormalSampler, SplitMultinormal
from bmi.samplers._transformed import TransformedSampler
from bmi.samplers.base import BaseSampler

__all__ = [
    "AdditiveUniformSampler",
    "BaseSampler",
    "parametrised_correlation_matrix",
    "BivariateNormalSampler",
    "SplitMultinormal",
    "SplitStudentT",
    "TransformedSampler",
    "DenseLVMParametrization",
    "SparseLVMParametrization",
    "GaussianLVMParametrization",
]
