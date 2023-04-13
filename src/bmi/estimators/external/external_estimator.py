import subprocess
import tempfile

from numpy.typing import ArrayLike

import bmi.utils as utils
from bmi.interface import IMutualInformationPointEstimator, Pathlike


def _run_command_and_read_mi(args: list[str]) -> float:
    raw_output = subprocess.check_output(args)
    output: str = raw_output.decode().strip()

    try:
        mi_estimate = float(output)
    except Exception as e:
        raise ValueError(f"Failed to cast output to float, got output '{output}' and error '{e}'.")

    return mi_estimate


class ExternalEstimator(IMutualInformationPointEstimator):
    def _build_command(self, path: Pathlike, dim_x: int, dim_y: int) -> list[str]:
        raise NotImplementedError

    def estimate_from_path(self, path: Pathlike):
        dim_x, dim_y = utils.read_sample_dims(path)
        command = self._build_command(path, dim_x, dim_y)
        return _run_command_and_read_mi(command)

    def estimate(self, x: ArrayLike, y: ArrayLike) -> float:
        with tempfile.NamedTemporaryFile() as file:
            utils.save_sample(file.name, x, y)
            return self.estimate_from_path(file.name)
