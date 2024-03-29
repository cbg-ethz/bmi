"""The tests for public API."""

import pytest


def test_api_imports() -> None:
    import bmi  # noqa: F401 import not at the top of the file


SUBMODULES = [
    "benchmark",
    "estimators",
    "samplers",
    "transforms",
]


@pytest.mark.parametrize("submodule", SUBMODULES)
def test_api_exports_submodules(submodule: str) -> None:
    import bmi as bmi  # noqa: F401 import not at the top of the file

    assert hasattr(bmi, submodule)
