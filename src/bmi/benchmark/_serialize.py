import pathlib
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pydantic
import yaml

_PREFIX_X: str = "X"
_PREFIX_Y: str = "Y"
_SEED_COLUMN: str = "seed"

Pathlike = Union[str, pathlib.Path]
Seed = int
SamplesXY = Tuple[np.ndarray, np.ndarray]
SamplesDict = Dict[Seed, SamplesXY]
ColumnName = str


def _column_names(prefix: str, dim: int) -> List[ColumnName]:
    return [f"{prefix}{i}" for i in range(1, dim + 1)]


def column_names(dim_x: int, dim_y: int, prefix_x: str, prefix_y: str) -> List[ColumnName]:
    return _column_names(prefix_x, dim_x) + _column_names(prefix_y, dim_y)


def samples_to_dataframe(
    seed: int,
    samples_x: np.ndarray,
    samples_y: np.ndarray,
    prefix_x: str = _PREFIX_X,
    prefix_y: str = _PREFIX_Y,
    seed_column: ColumnName = _SEED_COLUMN,
) -> pd.DataFrame:
    n_samples, dim_x = samples_x.shape
    n_samples_, dim_y = samples_y.shape

    if n_samples != n_samples_:
        raise ValueError(f"Number of samples needs to be equal, but {n_samples} != {n_samples_}.")

    paired = np.hstack([samples_x, samples_y])
    columns = column_names(dim_x=dim_x, dim_y=dim_y, prefix_x=prefix_x, prefix_y=prefix_y)

    df = pd.DataFrame(paired)
    df.columns = columns

    # The first column will represent seed
    df.insert(loc=0, column=seed_column, value=seed)

    return df


def dict_to_dataframe(
    dct: SamplesDict,
    prefix_x: str = _PREFIX_X,
    prefix_y: str = _PREFIX_Y,
    seed_column: ColumnName = _SEED_COLUMN,
) -> pd.DataFrame:
    dfs = [
        samples_to_dataframe(
            seed=seed,
            samples_x=samples_x,
            samples_y=samples_y,
            prefix_x=prefix_x,
            prefix_y=prefix_y,
            seed_column=seed_column,
        )
        for seed, (samples_x, samples_y) in dct.items()
    ]
    return pd.concat(dfs)


def _sortingkey(prefix: str) -> Callable[[str], int]:
    n = len(prefix)

    def f(col: str) -> int:
        return int(col[n:])

    return f


def _extract_prefix_columns(
    dataframe: pd.DataFrame, prefix: str, dim: Optional[int]
) -> np.ndarray:
    columns = [col for col in dataframe.columns if col.startswith(prefix)]

    if dim is not None:
        if len(columns) != dim:
            raise ValueError(
                f"We expected {dim} columns but obtained {columns} matching prefix {prefix}."
            )

    # Now sort the columns by the index
    columns = sorted(columns, key=_sortingkey(prefix))

    return dataframe[columns].values


def dataframe_to_dict(
    df: pd.DataFrame,
    dim_x: Optional[int] = None,
    dim_y: Optional[int] = None,
    prefix_x: str = _PREFIX_X,
    prefix_y: str = _PREFIX_Y,
    seed_column: ColumnName = _SEED_COLUMN,
) -> SamplesDict:
    return {
        seed: (
            _extract_prefix_columns(minidf, prefix=prefix_x, dim=dim_x),
            _extract_prefix_columns(minidf, prefix=prefix_y, dim=dim_y),
        )
        for seed, minidf in df.groupby(seed_column)
    }


class TaskDirectory:
    def __init__(self, path: Pathlike) -> None:
        self.path = pathlib.Path(path)

        self.task_metadata = self.path / "metadata.json"
        self.samples = self.path / "samples.csv"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(path={self.path})"

    def can_load(self) -> bool:
        return self.path.is_dir() and self.task_metadata.exists() and self.samples.exists()

    def save(
        self, metadata: pydantic.BaseModel, samples: pd.DataFrame, exist_ok: bool = False
    ) -> None:
        self.path.mkdir(parents=True, exist_ok=exist_ok)

        # Save metadata
        with open(self.task_metadata, "w") as outfile:
            yaml.dump(metadata.dict(), outfile)

        # Save samples
        samples.to_csv(self.samples, index=False)

    def load_metadata(self) -> dict:
        with open(self.task_metadata) as f:
            return yaml.load(f, Loader=yaml.SafeLoader)

    def load_samples(self) -> pd.DataFrame:
        return pd.read_csv(self.samples)
