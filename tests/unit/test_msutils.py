from logging import exception
import pytest

from npanalyst import msutils


ERROR_COLS = ["PrecMz", "RetTime"]


def test_make_error_col_names():
    expected = ["PrecMz_low", "RetTime_low", "PrecMz_high", "RetTime_high"]
    assert msutils.make_error_col_names(ERROR_COLS) == expected
