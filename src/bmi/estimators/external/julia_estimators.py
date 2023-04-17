from pathlib import Path
from typing import Literal

import pydantic

from bmi.estimators.external.external_estimator import ExternalEstimator
from bmi.interface import BaseModel, Pathlike

# TODO(frdrc): figure out how to ship external code in installed packages
JULIA_CODE_DIR = Path(__file__).parent.parent.parent.parent.parent / "external"


class KSGParams(BaseModel):
    variant: Literal[1, 2]
    neighbors: pydantic.PositiveInt


class JuliaKSGEstimator(ExternalEstimator):
    """
    The KSG estimators implemented in the `TransferEntropy` package in julia.
    WARNING: gives suspicious results.
    """

    def __init__(self, variant: Literal[1, 2] = 1, neighbors: int = 10) -> None:
        """
        Args:
            variant: 1 corresponds to KSG1, 2 corresponds to KSG2
            neighbors: number of neighbors (k) to be used
        """
        self._params = KSGParams(
            variant=variant,
            neighbors=neighbors,
        )

    def parameters(self) -> KSGParams:
        return self._params

    def _build_command(self, path: Pathlike, dim_x: int, dim_y: int):
        sample_path_abs = str(Path(path).absolute())
        estimator_julia_path_abs = str((JULIA_CODE_DIR / "mi_estimator.jl").absolute())
        return [
            "julia",
            estimator_julia_path_abs,
            sample_path_abs,
            str(dim_x),
            str(dim_y),
            "--estimator",
            f"KSG{self._params.variant}",
            "--neighbors",
            str(self._params.neighbors),
        ]


class HistogramParams(BaseModel):
    bins: pydantic.PositiveInt


class JuliaHistogramEstimator(ExternalEstimator):
    """
    The VisitationFrequency estimator implemented in the `TransferEntropy` package in julia.
    """

    def __init__(self, bins: int = 10) -> None:
        if bins < 2:
            raise ValueError(f"Bins {bins} must be at least 2.")

        self._params = HistogramParams(
            bins=bins,
        )

    def parameters(self) -> HistogramParams:
        return self._params

    def _build_command(self, path: Pathlike, dim_x: int, dim_y: int):
        sample_path_abs = str(Path(path).absolute())
        estimator_julia_path_abs = str((JULIA_CODE_DIR / "mi_estimator.jl").absolute())
        return [
            "julia",
            estimator_julia_path_abs,
            sample_path_abs,
            str(dim_x),
            str(dim_y),
            "--estimator",
            "Histogram",
            "--bins",
            str(self._params.bins),
        ]


class TransferParams(BaseModel):
    bins: pydantic.PositiveInt


class JuliaTransferEstimator(ExternalEstimator):
    """
    The TransferOperator estimator implemented in the `TransferEntropy` package in julia.
    """

    def __init__(self, bins: int = 10) -> None:
        if bins < 2:
            raise ValueError(f"Bins {bins} must be at least 2.")

        self._params = TransferParams(
            bins=bins,
        )

    def parameters(self) -> TransferParams:
        return self._params

    def _build_command(self, path: Pathlike, dim_x: int, dim_y: int):
        sample_path_abs = str(Path(path).absolute())
        estimator_julia_path_abs = str((JULIA_CODE_DIR / "mi_estimator.jl").absolute())
        return [
            "julia",
            estimator_julia_path_abs,
            sample_path_abs,
            str(dim_x),
            str(dim_y),
            "--estimator",
            "Transfer",
            "--bins",
            str(self._params.bins),
        ]


class KernelParams(BaseModel):
    bandwidth: pydantic.PositiveFloat


class JuliaKernelEstimator(ExternalEstimator):
    """
    The NaiveKernel estimator implemented in the `TransferEntropy` package in julia.
    WARNING: gives suspicious results.
    """

    def __init__(self, bandwidth: float = 1.0) -> None:
        if bandwidth <= 0.0:
            raise ValueError(f"Bandwidth {bandwidth} should be positive.")

        self._params = KernelParams(
            bandwidth=bandwidth,
        )

    def parameters(self) -> KernelParams:
        return self._params

    def _build_command(self, path: Pathlike, dim_x: int, dim_y: int):
        sample_path_abs = str(Path(path).absolute())
        estimator_julia_path_abs = str((JULIA_CODE_DIR / "mi_estimator.jl").absolute())
        return [
            "julia",
            estimator_julia_path_abs,
            sample_path_abs,
            str(dim_x),
            str(dim_y),
            "--estimator",
            "Kernel",
            "--bandwidth",
            str(self._params.bandwidth),
        ]
