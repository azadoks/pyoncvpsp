"""Tests for the substring pre-filter of the line searches."""

import re

import pytest

from pyoncvpsp.io.output._lines import LineIndex, required_literal
from pyoncvpsp.io.output._sections import _RE, _RE_ANY

LINES = [
    "",
    "Begin PSPCODE8",
    "  Wellstate for l =  1  n =  3",
    "wellstate: Negative-energy state for l= 2 found for maximum barrier",
    "!p   0  0.1  -1.0",
    "",
    " !L  0.2  -3.0",
    "WARNING: negative energy of -0.5",
    "end",
]


@pytest.mark.parametrize(
    "pattern, literal",
    [
        (r"^Begin PSPCODE8", "Begin PSPCODE8"),
        (r"\s+Wellstate for l =\s+(?P<ell>\d)", "Wellstate for l ="),
        # Groups of the top-level sequence are matched exactly once
        (r"ab(cd)ef\s+x", "abcdef"),
        # Alternatives, optional groups and repeats are not required
        (r"foo|barbaz", ""),
        (r"x(?:longer)?y", "x"),
        (r"(?:abc)+", ""),
        # Case-insensitive matching has no literal
        (r"(?i)Wellstate", ""),
        (r"Well(?i:STATE)", "Well"),
        (r"", ""),
    ],
)
def test_required_literal(pattern, literal):
    assert required_literal(re.compile(pattern)) == literal


@pytest.mark.parametrize(
    "pattern",
    [
        r"^Begin PSPCODE8",
        r"ellstate",
        r"[Ww]ellstate",
        r"(?i)WELLSTATE",
        r"!\s+(?P<marker>[\s-])?(?P<l>\d)",
        r" !L\s+",
        r"^$",
        r"negative energy of\s*-",
        r"e",
        r"",
    ],
)
@pytest.mark.parametrize("start, stop", [(0, None), (2, 7), (3, 4), (5, 5), (8, 20)])
def test_finditer_is_search_on_every_line(pattern, start, stop):
    regex = re.compile(pattern)
    end = len(LINES) if stop is None else min(stop, len(LINES))
    expected = [
        (i, m.span()) for i in range(start, end) if (m := regex.search(LINES[i]))
    ]
    found = [(i, m.span()) for i, m in LineIndex(LINES).finditer(regex, start, stop)]
    assert found == expected


def test_containing_finds_each_line_once():
    assert list(LineIndex(LINES).containing("e")) == [1, 2, 3, 7, 8]
    assert list(LineIndex(LINES).containing("\n")) == []
    assert list(LineIndex([]).containing("e")) == []


@pytest.mark.parametrize("name", _RE)
def test_any_marker_matches_where_a_marker_does(name):
    # A line must pass the combined pre-filter wherever one of the markers matches it
    lines = [
        "Begin loop to  construct optimized pseudo wave functions",
        "   Wellstate for l =  1  n =  3",
        "  cnorm=   1.0  ll=  1  n=  2 kap= -2",
        "Calculating optimized projector #   1",
        " for l=   0",
        "Calculating first optimized projector for l=   1",
        "B matrix Hermiticity error, ll=  1  kap=  1",
        "Diagnostic tests using semi-local pseudopotentials",
        "    1    2.0000   1.00000  -0.12345",
        "Test configuration 3",
        "scattering, iprj=  1,  l=  2",
        "Begin PSPCODE8",
    ]
    for line in lines:
        if _RE[name].match(line):
            assert _RE_ANY.match(line)
