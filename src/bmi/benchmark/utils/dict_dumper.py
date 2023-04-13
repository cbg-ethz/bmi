import pathlib
from enum import Enum

import numpy as np
import yaml


class DictDumper(yaml.SafeDumper):
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
        elif isinstance(data, Enum):
            return super().represent_data(data.value)

        return super().represent_data(data)
