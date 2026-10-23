"""Tests for the I/O utilities."""

import re

import pytest

from pyoncvpsp.io._utils import fort_float
from pyoncvpsp.io.output._data import RE_FLOAT as RE_FLOAT_DATA
from pyoncvpsp.io.output._text import RE_FLOAT


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("1.5", 1.5),
        ("-2.0E-03", -2.0e-3),
        ("3.25D+02", 325.0),
        ("4.0d-1", 0.4),
        # an E or D edit descriptor with a 2-digit exponent field at |exponent| > 99 (e24.16)
        ("2.1869480064996404-119", 2.1869480064996404e-119),
        ("-1.0000000000000000+105", -1.0e105),
        ("7.-101", 7.0e-101),
    ],
)
def test_fort_float(text, value):
    assert fort_float(text) == value


@pytest.mark.parametrize("pattern", [RE_FLOAT, RE_FLOAT_DATA])
def test_re_float_bare_exponent(pattern):
    """RE_FLOAT takes a 3-digit exponent without a letter whole, and leaves run-together fields apart."""
    assert re.fullmatch(pattern, "2.1869480064996404-119")
    assert re.fullmatch(pattern, "-1.0000000000000000+105")
    # f-format fields that run together: the sign starts the next field
    match = re.match(pattern, "0.50000-123.45678")
    assert match is not None and match.group() == "0.50000"
    assert [
        fort_float(x) for x in re.findall(pattern, "1.0E-02  2.1869480064996404-119")
    ] == [
        1.0e-2,
        2.1869480064996404e-119,
    ]
