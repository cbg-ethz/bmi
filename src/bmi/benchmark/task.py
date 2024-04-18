from typing import Optional

import numpy as np
import pydantic
import yaml

from bmi.benchmark.utils.dict_dumper import DictDumper
from bmi.interface import BaseModel, ISampler, Pathlike
from bmi.utils import save_sample


class TaskMetadata(BaseModel):
    task_id: str
    task_name: str
    dim_x: int
    dim_y: int
    mi_true: pydantic.NonNegativeFloat
    task_params: dict = pydantic.Field(default_factory=dict)


class Task:
    """Essentialy a named, preconfigured sampler."""

    def __init__(
        self,
        sampler: ISampler,
        task_id: str,
        task_name: str,
        task_params: Optional[dict] = None,
    ):
        self.sampler = sampler
        self._task_id = task_id
        self._task_name = task_name
        self._task_params = task_params

    @property
    def metadata(self) -> TaskMetadata:
        return TaskMetadata(
            task_id=self._task_id,
            task_name=self._task_name,
            dim_x=self.sampler.dim_x,
            dim_y=self.sampler.dim_y,
            mi_true=self.sampler.mutual_information(),
            task_params=self._task_params or dict(),
        )

    @property
    def id(self) -> str:
        return self.metadata.task_id

    @property
    def name(self) -> str:
        return self.metadata.task_name

    @property
    def params(self) -> dict:
        return self.metadata.task_params

    @property
    def dim_x(self) -> int:
        return self.metadata.dim_x

    @property
    def dim_y(self) -> int:
        return self.metadata.dim_y

    @property
    def mutual_information(self) -> float:
        return self.metadata.mi_true

    def save_metadata(self, path: Pathlike) -> None:
        with open(path, "w") as outfile:
            yaml.dump(self.metadata.dict(), outfile, Dumper=DictDumper)

    def sample(self, n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        return self.sampler.sample(n_points=n_samples, rng=seed)

    def save_sample(self, path: Pathlike, n_samples: int, seed: int) -> None:
        samples_x, samples_y = self.sample(n_samples, seed)
        save_sample(path, samples_x, samples_y)


# TODO(frdrc): read_sample('path/to/saved/sample.csv') -> tuple[array, array]

# TODO(frdrc):
# > dump_task('path/', task, seeds=[0, 1, 2], samples=[1000, 2000])
#
# should create:
#
# path/
#   task_id/
#     metadata.yaml
#     samples/
#       1000-0.csv
#       1000-1.csv
#       1000-2.csv
#       2000-0.csv
#       2000-1.csv
#       2000-2.csv
