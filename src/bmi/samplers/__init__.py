"""Subpackage with different samplers, which can be used to define benchmark tasks
or used directly to sample from distributions with known mutual information."""
from bmi.samplers._additive_uniform import AdditiveUniformSampler
from bmi.samplers.dispersion import parametrised_correlation_matrix
from bmi.samplers.split_student_t import SplitStudentT
from bmi.samplers.splitmultinormal import BivariateNormalSampler, SplitMultinormal
from bmi.samplers.transformed import TransformedSampler

__all__ = [
    "AdditiveUniformSampler",
    "parametrised_correlation_matrix",
    "BivariateNormalSampler",
    "SplitMultinormal",
    "SplitStudentT",
    "TransformedSampler",
]
