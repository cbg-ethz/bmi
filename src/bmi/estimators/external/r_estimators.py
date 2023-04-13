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
