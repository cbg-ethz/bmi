import dataclasses

import pytest


@dataclasses.dataclass
class TurnOnTestSuiteArgument:
    """CLI argument together with a keyword used to mark tests turned off by default.

    Attrs:
        cli_flag: CLI flag to be used to turn on an optional test suite, e.g., --run-r
        help_message: CLI help message, describing the flag
        mark_keyword: keyword used to mark the unit tests to be skipped
            if the flag is not specified.
            Note that it should be a valid Python name, e.g., requires_r
        inivalue_line_description: a short description of the marker `mark_keyword`.
            See also: `inivalue_line`
    """

    cli_flag: str
    help_message: str
    mark_keyword: str
    inivalue_line_description: str

    def reason(self) -> str:
        """Reason for skipping, generated when the tests are run."""
        return f"Need {self.cli_flag} option to run."

    def inivalue_line(self) -> str:
        return f"{self.mark_keyword}: {self.inivalue_line_description}"


TURN_ON_ARGUMENTS = [
    TurnOnTestSuiteArgument(
        cli_flag="--run-r",
        help_message="Run tests requiring R dependencies.",
        mark_keyword="requires_r",
        inivalue_line_description="mark test as requiring R dependencies to run.",
    ),
    TurnOnTestSuiteArgument(
        cli_flag="--run-julia",
        help_message="Run tests requiring Julia dependencies.",
        mark_keyword="requires_julia",
        inivalue_line_description="mark test as requiring Julia estimators to run.",
    ),
]


def pytest_addoption(parser):
    for argument in TURN_ON_ARGUMENTS:
        parser.addoption(
            argument.cli_flag,
            action="store_true",
            default=False,
            help=argument.help_message,
        )


def pytest_configure(config):
    for argument in TURN_ON_ARGUMENTS:
        config.addinivalue_line("markers", argument.inivalue_line())


def add_skipping_markers(
    argument: TurnOnTestSuiteArgument,
    config,
    items,
) -> None:
    if not config.getoption(argument.cli_flag):
        skip = pytest.mark.skip(reason=argument.reason())
        for item in items:
            if argument.mark_keyword in item.keywords:
                item.add_marker(skip)


def pytest_collection_modifyitems(config, items):
    for argument in TURN_ON_ARGUMENTS:
        add_skipping_markers(argument=argument, config=config, items=items)
