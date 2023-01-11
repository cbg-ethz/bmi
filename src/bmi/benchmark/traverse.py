from pathlib import Path
from typing import Iterable

import yaml

from bmi.benchmark.core import TaskMetadata
from bmi.benchmark.filesys.api import OurCustomDumper, TaskDirectory
from bmi.interface import Pathlike, RunResult

_MetadataDict = dict[str, TaskMetadata]
_ResultList = list[RunResult]


class LoadTaskMetadata:
    @staticmethod
    def from_path(path: Pathlike) -> TaskMetadata:
        return TaskMetadata(**TaskDirectory(path).load_metadata())

    @classmethod
    def from_paths(cls, paths: Iterable[Pathlike]) -> _MetadataDict:
        ret = {}
        for path in paths:
            task_metadata = cls.from_path(path)
            task_id = task_metadata.task_id
            if task_id in ret:
                raise ValueError(
                    f"Task {task_id} loaded from path {path} "
                    f"is already present in the dictionary: {ret[task_id]}."
                )
            else:
                ret[task_id] = task_metadata
        return ret

    @classmethod
    def from_directory(cls, path: Pathlike) -> _MetadataDict:
        return cls.from_paths(Path(path).iterdir())


class SaveLoadRunResults:
    @staticmethod
    def from_path(path: Pathlike) -> RunResult:
        with open(path) as f:
            return RunResult(**yaml.load(f, Loader=yaml.SafeLoader))

    @classmethod
    def from_paths(cls, paths: Iterable[Pathlike]) -> _ResultList:
        return [cls.from_path(path) for path in paths]

    @classmethod
    def from_directory(cls, path: Pathlike) -> _ResultList:
        return cls.from_paths(Path(path).iterdir())

    @staticmethod
    def dump(result: RunResult, *, path: Pathlike) -> None:
        with open(path, "w") as outfile:
            yaml.dump(result.dict(), outfile, Dumper=OurCustomDumper)
