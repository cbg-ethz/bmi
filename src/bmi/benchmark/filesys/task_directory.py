import pathlib

import pandas as pd
import yaml

from bmi.benchmark.filesys.serialize_dataframe import _SEED_COLUMN, ColumnName
from bmi.benchmark.filesys.serialize_object import OurCustomDumper
from bmi.interface import BaseModel, Pathlike


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

    def seeds(self, _seed_column: ColumnName = _SEED_COLUMN) -> list[int]:
        df = pd.read_csv(self.samples, usecols=[_seed_column])
        return sorted(df[_seed_column].unique())
