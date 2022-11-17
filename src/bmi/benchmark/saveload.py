import dataclasses
import pathlib
from typing import Union

import numpy as np
import pydantic
from numpy.typing import ArrayLike

from bmi.interface import ISampler


class TaskMetadata(pydantic.BaseModel):
    task_id: str
    dim_x: int
    dim_y: int
    n_samples: int
    random_seed: int
    mi_true: pydantic.NonNegativeFloat


@dataclasses.dataclass
class Task:
    metadata: TaskMetadata
    x_samples: np.ndarray
    y_samples: np.ndarray


def generate_task(sampler: ISampler, n_samples: int, seed: int, task_id: str) -> Task:
    metadata = TaskMetadata(
        task_id=task_id,
        dim_x=sampler.dim_x,
        dim_y=sampler.dim_y,
        n_samples=n_samples,
        random_seed=seed,
        mi_true=sampler.mutual_information(),
    )

    x, y = sampler.sample(n_points=n_samples, rng=seed)

    return Task(
        metadata=metadata,
        x_samples=x,
        y_samples=y,
    )


class RunResult(pydantic.BaseModel):
    """Class keeping the output of a single model run."""

    task_id: str
    mi_estimate: float
    time_in_seconds: float


class TaskDirectory(pathlib.Path):
    def __init__(self, path: Union[str, pathlib.Path]) -> None:
        super().__init__(path)

        self.task_metadata = self / "task_metadata.json"
        self.points_x = self / "x_samples.csv"
        self.points_y = self / "y_samples.csv"

    def can_load(self) -> bool:
        return (
            self.is_dir()
            and self.task_metadata.exists()
            and self.points_x.exists()
            and self.points_y.exists()
        )


_DELIMITER: str = ","


def _save_samples(x: ArrayLike, y: ArrayLike, directory: TaskDirectory) -> None:
    np.savetxt(directory.points_x, x, delimiter=_DELIMITER)
    np.savetxt(directory.points_y, y, delimiter=_DELIMITER)


def _load_samples(directory: TaskDirectory) -> tuple[np.ndarray, np.ndarray]:
    x = np.loadtxt(directory.points_x, delimiter=_DELIMITER)
    y = np.loadtxt(directory.points_y, delimiter=_DELIMITER)
    assert len(x) == len(y)

    return x, y


def _save_metadata(metadata: TaskMetadata) -> None:
    raise NotImplementedError


def _load_metadata(directory: TaskDirectory) -> TaskMetadata:
    raise NotImplementedError


def save_task(task: Task, exist_ok: bool = True) -> None:
    pass


def load_task(directory: Union[pathlib.Path, str]) -> Task:
    directory = TaskDirectory(directory)
    assert directory.can_load()

    x, y = _load_samples(directory)
    metadata = _load_metadata(directory)

    assert x.shape == (metadata.n_samples, metadata.dim_x)
    assert y.shape == (metadata.n_samples, metadata.dim_y)

    return Task(
        x_samples=x,
        y_samples=y,
        metadata=metadata,
    )
