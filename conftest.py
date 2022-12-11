import pytest


def pytest_addoption(parser):
    parser.addoption("--run-r", action="store_true", default=False, help="Run tests requiring R.")


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_r: mark test as requiring R dependencies to run.")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-r"):
        # --run-r given in cli: do not skip tests requiring R
        return
    skip_r = pytest.mark.skip(reason="Need --run-r option to run")
    for item in items:
        if "requires_r" in item.keywords:
            item.add_marker(skip_r)
