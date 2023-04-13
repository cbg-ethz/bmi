from pathlib import Path

from bmi.interface import Pathlike, BaseModel
from bmi.estimators.external.external_estimator import ExternalEstimator


# TODO(frdrc): figure out how to ship external code in installed packages
R_CODE_DIR = Path(__file__).parent.parent.parent.parent.parent / "external"


class RKSGEstimator(ExternalEstimator):
    # TODO(frdrc): variant neighbours etc etc
    def __init__(self):
        pass

    # TODO(frdrc)
    def parameters(self) -> BaseModel:
        return None

    def _build_command(self, path: Pathlike, dim_x: int, dim_y: int):
        sample_path_abs = str(Path(path).absolute())
        estimator_r_path_abs = str((R_CODE_DIR / 'rmi.R').absolute())
        return ['Rscript', estimator_r_path_abs, sample_path_abs, str(dim_x), str(dim_y)]
