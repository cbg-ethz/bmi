from pathlib import Path
from typing import Literal

import pydantic

from bmi.estimators.external.external_estimator import ExternalEstimator
from bmi.interface import BaseModel, Pathlike

# TODO(frdrc): figure out how to ship external code in installed packages
R_CODE_DIR = Path(__file__).parent.parent.parent.parent.parent / "external"


class KSGParams(BaseModel):
    variant: Literal[1, 2]
    neighbors: pydantic.PositiveInt


class RKSGEstimator(ExternalEstimator):
    """The KSG estimators implemented in the `rmi` package in R."""

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
        estimator_r_path_abs = str((R_CODE_DIR / "rmi.R").absolute())
        return [
            "Rscript",
            estimator_r_path_abs,
            sample_path_abs,
            str(dim_x),
            str(dim_y),
            "--method",
            f"KSG{self._params.variant}",
            "--neighbors",
            str(self._params.neighbors),
        ]


class LNNParams(BaseModel):
    k: pydantic.PositiveInt
    truncation: pydantic.PositiveInt


class RLNNEstimator(ExternalEstimator):
    """The LNN estimator implemented in the `rmi` package in R."""

    def __init__(self, k: int = 5, truncation: int = 30) -> None:
        """
        Args:
            k: order of local kNN bandwidth estimation
            truncation: number of neighbors to include in the local density estimation
        """
        self._params = LNNParams(
            k=k,
            truncation=truncation,
        )

    def parameters(self) -> LNNParams:
        return self._params

    def _build_command(self, path: Pathlike, dim_x: int, dim_y: int):
        sample_path_abs = str(Path(path).absolute())
        estimator_r_path_abs = str((R_CODE_DIR / "rmi.R").absolute())
        return [
            "Rscript",
            estimator_r_path_abs,
            sample_path_abs,
            str(dim_x),
            str(dim_y),
            "--method",
            "LNN",
            "--neighbors",
            str(self._params.k),
            "--truncation",
            str(self._params.truncation),
        ]


class BNSLParams(BaseModel):
    proc: pydantic.PositiveInt


class RBNSLEstimator(ExternalEstimator):
    """The MI estimator implemented in the `BNSL` package in R."""

    def __init__(self, proc: int = 10) -> None:
        """
        Args:
            proc: the estimation is based on Jeffrey's prior, the MDL principle,
                  and BDeu for proc=0,1,2, respectively. If one of the two is
                  continuous, proc=10 should be chosen. If the argument proc is
                  missing, proc=0 (Jeffreys') is assumed.
        """
        self._params = BNSLParams(
            proc=proc,
        )

    def parameters(self) -> BNSLParams:
        return self._params

    def _build_command(self, path: Pathlike, dim_x: int, dim_y: int):
        sample_path_abs = str(Path(path).absolute())
        estimator_r_path_abs = str((R_CODE_DIR / "rmi.R").absolute())
        return [
            "Rscript",
            estimator_r_path_abs,
            sample_path_abs,
            str(dim_x),
            str(dim_y),
            "--method",
            "BNSL",
            "--proc",
            str(self._params.proc),
        ]
