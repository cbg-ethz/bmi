from typing import Generator, Iterable, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import pydantic

import bmi.benchmark.filesys.api as fs
from bmi.interface import BaseModel, ISampler, Pathlike, Seed


class TaskMetadata(BaseModel):
    task_id: str
    dim_x: int
    dim_y: int
    n_samples: int
    mi_true: pydantic.NonNegativeFloat
    task_params: dict = pydantic.Field(default_factory=dict)


class Task:
    """Class representing a given mutual information estimation task.

    We imagine that most of the tasks will be created either via `Task.load()`
    functionality or the factory method `generate_task()`.
    """

    def __init__(
        self, metadata: TaskMetadata, samples: Union[fs.SamplesDict, pd.DataFrame]
    ) -> None:
        self.metadata = metadata
        # TODO(Pawel): Add dimension validation if dictionary is passed, rather than a data frame
        self._samples: fs.SamplesDict = (
            fs.dataframe_to_dictionary(samples, dim_x=metadata.dim_x, dim_y=metadata.dim_y)
            if isinstance(samples, pd.DataFrame)
            else samples
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(metadata={repr(self.metadata)}, samples={repr(self._samples)})"
        )

    def __str__(self) -> str:
        seeds_str = " ".join(map(str, self._samples.keys()))

        return f"{type(self).__name__}({self.metadata} seeds={seeds_str}"

    def keys(self) -> Set[Seed]:
        return set(self._samples.keys())

    def __getitem__(self, item: Seed) -> fs.SamplesXY:
        return self._samples[item]

    def __iter__(self) -> Generator[Tuple[Seed, fs.SamplesXY], None, None]:
        for seed, vals in self._samples.items():
            yield seed, vals

    def __eq__(self, other) -> bool:
        # Make sure the type is right
        if not isinstance(other, type(self)):
            return False
        # Check if the metadata is the same
        if other.metadata != self.metadata:
            return False

        # Check if the samples are the same.
        # This will need to be split into several checks

        # First, check if we have the same random seeds
        if self.keys() != other.keys():
            return False

        # Then, for every seed check whether the
        # samples are the same
        for seed in self.keys():
            self_x, self_y = self[seed]
            other_x, other_y = other[seed]

            if not np.allclose(self_x, other_x):
                return False
            if not np.allclose(self_y, other_y):
                return False
        return True

    @property
    def task_id(self) -> str:
        return self.metadata.task_id

    @property
    def dim_x(self) -> int:
        return self.metadata.dim_x

    @property
    def dim_y(self) -> int:
        return self.metadata.dim_y

    @property
    def mi_true(self) -> float:
        return self.metadata.mi_true

    @property
    def n_samples(self) -> int:
        return self.metadata.n_samples

    @property
    def task_params(self) -> dict:
        return self.metadata.task_params

    def save(self, path: Pathlike, exist_ok: bool = False) -> None:
        """Saves the task to the disk.

        Args:
            path: path to the directory (without any extension) where
              the task should be saved to
            exist_ok: if True, we can overwrite existing files.
              Otherwise an exception is raised.
        """
        task_directory = fs.TaskDirectory(path)
        df = fs.dictionary_to_dataframe(self._samples)
        task_directory.save(metadata=self.metadata, samples=df, exist_ok=exist_ok)

    @classmethod
    def load(cls, path: Pathlike) -> "Task":
        """Loads the task from the disk."""
        task_directory = fs.TaskDirectory(path)

        metadata = TaskMetadata(**task_directory.load_metadata())
        samples_df: pd.DataFrame = task_directory.load_samples()

        return cls(
            metadata=metadata,
            samples=samples_df,
        )


def generate_task(
    sampler: ISampler,
    n_samples: int,
    seeds: Iterable[int],
    task_id: str,
    task_params: Optional[dict] = None,
) -> Task:
    """A factory method generating a task from a given sampler.

    Args:
        sampler: sampler used to generate the samples and the ground-truth mutual information
        n_samples: number of samples to be generated from the sampler
        seeds: list of seeds for which we should generate the samples
        task_id: a unique task id used to identify the task
        task_params: an optional dictionary with parameters of the task
    """
    task_params = {} if task_params is None else task_params

    metadata = TaskMetadata(
        task_id=task_id,
        dim_x=sampler.dim_x,
        dim_y=sampler.dim_y,
        n_samples=n_samples,
        mi_true=sampler.mutual_information(),
        task_params=task_params,
    )

    samples = {seed: sampler.sample(n_points=n_samples, rng=seed) for seed in seeds}

    return Task(
        metadata=metadata,
        samples=samples,
    )
