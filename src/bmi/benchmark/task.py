from typing import Optional

import numpy as np
import pandas as pd
import pydantic
import yaml

from bmi.interface import BaseModel, ISampler, Pathlike
from .utils.dict_dumper import DictDumper


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
        self.metadata = TaskMetadata(
            task_id=task_id,
            task_name=task_name,
            dim_x=self.sampler.dim_x,
            dim_y=self.sampler.dim_y,
            mi_true=self.sampler.mutual_information(),
            task_params=task_params or dict(),
        )

    @property
    def id(self):
        return self.metadata.task_id

    @property
    def name(self):
        return self.metadata.task_name

    @property
    def dim_x(self):
        return self.metadata.dim_x

    @property
    def dim_y(self):
        return self.metadata.dim_y

    @property
    def mutual_information(self):
        return self.metadata.mi_true

    def save_metadata(self, path: Pathlike):
        with open(path, "w") as outfile:
            yaml.dump(self.metadata.dict(), outfile, Dumper=DictDumper)

    def sample(self, n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        return self.sampler.sample(n_points=n_samples, rng=seed)

    def save_sample(self, path: Pathlike, n_samples: int, seed: int):
        samples_x, samples_y = self.sample(n_samples, seed)

        data = pd.DataFrame(
            np.hstack([samples_x, samples_y]),
            columns=[f'X{i}' for i in range(self.dim_x)] + [f'Y{i}' for i in range(self.dim_y)]
        )
        data.to_csv(path, index=False)


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
