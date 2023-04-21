from typing import Optional

import pydantic
import yaml

from bmi.benchmark.utils.dict_dumper import DictDumper
from bmi.benchmark.utils.timer import Timer
from bmi.estimators.external.external_estimator import ExternalEstimator
from bmi.interface import BaseModel, IMutualInformationPointEstimator, Pathlike
from bmi.utils import read_sample


class RunResult(BaseModel):
    """Class keeping the output of a single estimator run."""

    mi_estimate: float
    time_in_seconds: float
    success: bool
    estimator_id: str
    task_id: str
    n_samples: int
    seed: Optional[int]
    additional_information: dict = pydantic.Field(default_factory=dict)

    def dump(self, path):
        with open(path, "w") as outfile:
            yaml.dump(self.dict(), outfile, Dumper=DictDumper)


def run_estimator(
    estimator: IMutualInformationPointEstimator,
    estimator_id: str,
    sample_path: Pathlike,
    task_id: str,
    seed: Optional[int] = None,
    additional_information: Optional[dict] = None,
) -> RunResult:
    additional_information = additional_information or {}
    samples_x, samples_y = read_sample(sample_path)
    n_samples = len(samples_x)

    try:
        timer = Timer()

        # external estimators read sample on their own
        if isinstance(estimator, ExternalEstimator):
            estimate_result = estimator.estimate_with_info_from_path(sample_path)
        else:
            estimate_result = estimator.estimate_with_info(samples_x, samples_y)

        time_measured = timer.check()

        # prepare info for run result
        mi_estimate = estimate_result.mi_estimate
        time_reported = estimate_result.time_in_seconds
        time_in_seconds = time_reported if time_reported is not None else time_measured
        success = True
        additional_information |= estimate_result.additional_information

    # something went wrong
    except Exception as e:
        mi_estimate = float("nan")
        time_in_seconds = float("nan")
        success = False
        additional_information |= {"error": str(e)}
        # when a subprocess fails, e.output is set
        if hasattr(e, "output"):
            additional_information |= {"subprocess_output": str(e.output)}

    return RunResult(
        mi_estimate=mi_estimate,
        time_in_seconds=time_in_seconds,
        success=success,
        estimator_id=estimator_id,
        task_id=task_id,
        n_samples=n_samples,
        seed=seed,
        additional_information=additional_information,
    )
