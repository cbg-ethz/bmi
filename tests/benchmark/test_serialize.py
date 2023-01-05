from typing import Tuple

import pytest

import bmi.benchmark.filesys.serialize_dataframe as se


def test_column_names() -> None:
    columns = se.column_names(dim_x=3, dim_y=2, prefix_x="X", prefix_y="Y")
    assert columns == ["X1", "X2", "X3", "Y1", "Y2"]


@pytest.mark.parametrize(
    "params",
    [
        ("X", ["X1", "T2", "Y3", "X4"], ["X1", "X4"]),
        ("X", ["X1", "Y1", "X10", "X2"], ["X1", "X2", "X10"]),
        ("ab", ["ab2", "ab10", "ab1", "ab100"], ["ab1", "ab2", "ab10", "ab100"]),
        ("X-", ["X-1", "X-10", "X-2", "Y-5"], ["X-1", "X-2", "X-10"]),
    ],
)
@pytest.mark.parametrize("use_dim", [True, False])
def test_find_matching_column_names(
    params: Tuple[str, list, list],
    use_dim: bool,
) -> None:
    """Tests for find_matching_column_names checking if
    it works as expected on several simple cases."""
    prefix, all_columns, expected_output = params

    # Whether we specify the (correct) dimension
    #   or leave it as an optional argument
    dim = len(expected_output) if use_dim else None
    output = se._find_matching_column_names(columns=all_columns, prefix=prefix, dim=dim)

    assert output == expected_output


@pytest.mark.parametrize("params", [("X", ["X1", "X2", "Y2"], 2), ("Y", ["Y2", "Y4", "Y9"], 3)])
@pytest.mark.parametrize("diff", [-2, 1, 2, 3])
def test_find_matching_column_names_raises(
    params: Tuple[str, list, int],
    diff: int,
) -> None:
    """Tests if `find_matching_column_names` raises a ValueError
    if the number of expected columns is different from the dimension provided.

    Args:
        params: tuple (prefix, input list, length of the output)
        diff: non-zero number added to the length of the output
          to try to get the ValueError
    """
    prefix, columns, correct_dim = params
    with pytest.raises(ValueError):
        se._find_matching_column_names(
            columns=columns,
            prefix=prefix,
            dim=correct_dim + diff,
        )


def test_find_matching_column_names_wrong_suffix() -> None:
    with pytest.raises(ValueError):
        se._find_matching_column_names(
            columns=["seed", "X1", "X2", "Xenon"],
            prefix="X",
            dim=None,
        )
