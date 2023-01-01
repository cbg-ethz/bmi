import pathlib
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from bmi.interface import BaseModel, Pathlike, Seed

_PREFIX_X: str = "X"
_PREFIX_Y: str = "Y"
_SEED_COLUMN: str = "seed"

SamplesXY = Tuple[np.ndarray, np.ndarray]
SamplesDict = Dict[Seed, SamplesXY]
ColumnName = str


def _column_names(prefix: str, dim: int) -> List[ColumnName]:
    """Generates column names by merging prefix with 1-indexed
    number.

    Example:
        prefix="X" and dim=3 should generate
        ["X1", "X2", "X3"]
    """
    return [f"{prefix}{i}" for i in range(1, dim + 1)]


def column_names(dim_x: int, dim_y: int, prefix_x: str, prefix_y: str) -> List[ColumnName]:
    return _column_names(prefix_x, dim_x) + _column_names(prefix_y, dim_y)


def samples_to_dataframe(
    seed: int,
    samples_x: np.ndarray,
    samples_y: np.ndarray,
    _prefix_x: str = _PREFIX_X,
    _prefix_y: str = _PREFIX_Y,
    _seed_column: ColumnName = _SEED_COLUMN,
) -> pd.DataFrame:
    """Wraps samples into a dataframe of in our custom
    format [seed, X dimensions, Y dimensions].

    Args:
        seed: seed used to generate these samples,
          the whole column will have the same value
        samples_x: shape (n_samples, dim_x)
        samples_y: shape (n_samples, dim_y)
        _prefix_x: prefix used to index the X dimensions
        _prefix_y: prefix used to index the Y dimensions
        _seed_column: the name of the column storing the seed

    Returns:
        data frame with columns
          ["seed", "X1", "X2", ..., "Y1", "Y2", ...]
        (unless the column names are non-default)

    Note:
        We strongly advise *against* changing the column names
          from the default values

    See Also:
        dictionary_to_dataframe, which is a thin wrapper around
          this function
    """
    n_samples, dim_x = samples_x.shape
    n_samples_, dim_y = samples_y.shape

    if n_samples != n_samples_:
        raise ValueError(f"Number of samples needs to be equal, but {n_samples} != {n_samples_}.")

    paired = np.hstack([samples_x, samples_y])
    columns = column_names(dim_x=dim_x, dim_y=dim_y, prefix_x=_prefix_x, prefix_y=_prefix_y)

    df = pd.DataFrame(paired)
    df.columns = columns

    # The first column will represent seed
    df.insert(loc=0, column=_seed_column, value=seed)

    return df


def dictionary_to_dataframe(
    dct: SamplesDict,
    _prefix_x: str = _PREFIX_X,
    _prefix_y: str = _PREFIX_Y,
    _seed_column: ColumnName = _SEED_COLUMN,
) -> pd.DataFrame:
    """Wraps samples into a dataframe of in our custom
    format [seed, X dimensions, Y dimensions].

    Args:
        dct: dictionary mapping random seeds to samples
            Each sample should be a tuple (x_samples, y_samples)
        _prefix_x: prefix used to index the X dimensions
        _prefix_y: prefix used to index the Y dimensions
        _seed_column: the name of the column storing the seed

    Returns:
        data frame with columns
          ["seed", "X1", "X2", ..., "Y1", "Y2", ...]
        (unless the column names are non-default)

    Note:
        We strongly advise *against* changing the column names
          from the default values

    See Also:
        samples_to_dataframe, which does all the "real" work
        dataframe_to_dictionary, the inverse function
    """
    dfs = [
        samples_to_dataframe(
            seed=seed,
            samples_x=samples_x,
            samples_y=samples_y,
            _prefix_x=_prefix_x,
            _prefix_y=_prefix_y,
            _seed_column=_seed_column,
        )
        for seed, (samples_x, samples_y) in dct.items()
    ]
    return pd.concat(dfs)


def _sortingkey(prefix: str) -> Callable[[str], int]:
    """We use this to generate a sorting key in cases
    where we want to sort "X1", "X10", "X3", "X2"
    by the integer-valued suffix, i.e., we
    want to obtain "X1", "X2", "X3", "X10"
    """
    n = len(prefix)

    def f(col: str) -> int:
        return int(col[n:])

    return f


def _find_matching_column_names(
    columns: List[ColumnName],
    prefix: str,
    dim: Optional[int],
) -> List[ColumnName]:
    """Returns column names matching specified prefix,
    sorted by the index-valued suffix.

    Args:
        columns: list of names to select the right ones from
        prefix: a column will be selected if it starts with `prefix`
        dim: how many output columns are expected.
            If not None and we find a different number of columns,
            we will raise a ValueError

    Returns:
        list of column names starting with `prefix`

    Raises:
        ValueError, if `dim` is not None, but it doesn't
          equal the number of the columns with matching prefix

    Example:
        if column names are ["seed", "X2", "X1", "Y8", "X10"]
        and the prefix is "X", we will return ["X1", "X2", "X10"]
        in this specific order as 1 < 2 < 10
    """
    columns = [col for col in columns if col.startswith(prefix)]

    if dim is not None:
        if len(columns) != dim:
            raise ValueError(
                f"We expected {dim} columns but obtained {columns} matching prefix {prefix}."
            )

    # Now sort the columns by the index
    return sorted(columns, key=_sortingkey(prefix))


def _extract_prefix_columns(
    dataframe: pd.DataFrame, prefix: str, dim: Optional[int]
) -> np.ndarray:
    """Extracts a slice of the data frame with the columns matching
    the specified prefix.

    Returns:
        data frame slice, corresponding to the columns matching
          the prefix, sorted by the increasing index

    Raises:
        ValueError, if `dim` is specified but the number of the columns
          matching the `prefix` is different from `dim`

    Note:
        1. It's a thin wrapper around `_find_matching_column_names`.
        2. The columns are sorted by the corresponding index,
          see the example below.

    Example:
        if column names are ["seed", "X2", "X1", "Y8", "X10"]
        and the prefix is "X", we will slice on ["X1", "X2", "X10"]
        in this specific order
    """
    columns = _find_matching_column_names(
        columns=dataframe.columns,
        prefix=prefix,
        dim=dim,
    )
    return dataframe[columns].values


def dataframe_to_dictionary(
    df: pd.DataFrame,
    dim_x: Optional[int] = None,
    dim_y: Optional[int] = None,
    _prefix_x: str = _PREFIX_X,
    _prefix_y: str = _PREFIX_Y,
    _seed_column: ColumnName = _SEED_COLUMN,
) -> SamplesDict:
    """Parses a data frame in the specified format
    to a dictionary mapping random seeds to samples.

    Args:
        df: data frame with columns `_seed_column`
          `prefix_x`1, `prefix_x`2, ..., `prefix_y`1, `prefix_y`2, ...
           storing samples for different seeds
        dim_x: how many dimensions the X variable should have
        dim_y: how many dimensions the Y variables should have
        _prefix_x: naming convention, we want X1, X2, ...
        _prefix_y: naming convention, we want Y1, Y2, ....
        _seed_column: naming convention for the column

    See Also:
        dictionary_to_dataframe, the inverse to this function
    """
    return {
        seed: (
            _extract_prefix_columns(minidf, prefix=_prefix_x, dim=dim_x),
            _extract_prefix_columns(minidf, prefix=_prefix_y, dim=dim_y),
        )
        for seed, minidf in df.groupby(_seed_column)
    }


class OurCustomDumper(yaml.SafeDumper):
    """The default dumper in PyYAML has problems with the following objects:
      - Paths
      - NumPy arrays and NumPy floats

    Hence, we need to convert them manually to other formats.

    Note:
        This dumper should be extended in case you saw unexpected entry in the YAML file, as
        "&id" or "!!python".
    """

    def represent_data(self, data):
        if isinstance(data, pathlib.Path):  # Convert Paths to strings.
            return super().represent_data(str(data))
        elif isinstance(data, np.generic):  # Convert NumPy floats to floats
            return super().represent_data(data.item())
        elif isinstance(data, np.ndarray):
            return super().represent_data(data.tolist())

        return super().represent_data(data)


class TaskDirectory:
    """Auxiliary class helping to save and load metadata and dataframes."""

    METADATA: str = "metadata.yaml"
    SAMPLES: str = "samples.csv"

    def __init__(self, path: Pathlike) -> None:
        """
        Args:
            path: path to the directory
        """
        self.path = pathlib.Path(path)

        self.task_metadata = self.path / self.METADATA
        self.samples = self.path / self.SAMPLES

    def __repr__(self) -> str:
        return f"{type(self).__name__}(path={self.path})"

    def can_load(self) -> bool:
        return self.path.is_dir() and self.task_metadata.exists() and self.samples.exists()

    def save(self, metadata: BaseModel, samples: pd.DataFrame, exist_ok: bool = False) -> None:
        """Saves metadata and samples to the disk.

        Args:
            metadata: BaseModel to be serialized
            samples: pandas data frame to be saved
            exist_ok: if True and the directory already exists, it will
              overwrite its contents. If False, it will just raise an exception.
        """
        self.path.mkdir(parents=True, exist_ok=exist_ok)

        # Save metadata
        with open(self.task_metadata, "w") as outfile:
            yaml.dump(metadata.dict(), outfile, Dumper=OurCustomDumper)

        # Save samples
        samples.to_csv(self.samples, index=False)

    def load_metadata(self) -> dict:
        """Loads a YAML file into a dictionary.

        Note:
            No validation is performed.
        """
        with open(self.task_metadata) as f:
            return yaml.load(f, Loader=yaml.SafeLoader)

    def load_samples(self) -> pd.DataFrame:
        """Loads samples into a pandas DataFrame.

        Note:
            No validation is performed.
        """
        return pd.read_csv(self.samples)
