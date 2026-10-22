"""Tests for the substring pre-filter of the line searches."""

import re

import pytest

from pyoncvpsp.io import OncvpspTextParser
from pyoncvpsp.io.output._lines import LineIndex, required_literals
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
    "pattern, literals",
    [
        (r"^Begin PSPCODE8", {"Begin PSPCODE8"}),
        (r"\s+Wellstate for l =\s+(?P<ell>\d)", {"Wellstate for l ="}),
        # Groups of the top-level sequence are matched exactly once
        (r"ab(cd)ef\s+x", {"abcdef"}),
        # One of the alternatives is matched; the set with the longest shortest literal wins,
        # then the smallest
        (r"foo|barbaz", {"foo", "barbaz"}),
        (
            r"^(?:Fortran runtime error:|forrtl: severe)",
            {"Fortran runtime error:", "forrtl: severe"},
        ),
        (r"(?P<rel>sr|rel)atom: WARNING", {"atom: WARNING"}),
        (r"x(?:ab|cd)y", {"ab", "cd"}),
        (r"(?:ab|c\d)y", {"y"}),
        # Optional groups and repeats are not required
        (r"x(?:longer)?y", {"x"}),
        (r"(?:abc)+", set()),
        # Case-insensitive matching has no literal
        (r"(?i)Wellstate", set()),
        (r"Well(?i:STATE)", {"Well"}),
        (r"", set()),
    ],
)
def test_required_literals(pattern, literals):
    assert required_literals(re.compile(pattern)) == literals


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
        r"Begin|end",
        r"^(?:wellstate|WARNING):",
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


@pytest.mark.parametrize(
    "needles",
    [
        {"e"},
        {"Wellstate", "end"},
        {"!", "Begin"},
        {"", "x"},
        {"\n", "end"},
        {"zzz"},
        set(),
    ],
)
@pytest.mark.parametrize("start, stop", [(0, None), (2, 7), (8, 20)])
def test_containing_any_is_any_containing(needles, start, stop):
    end = len(LINES) if stop is None else min(stop, len(LINES))
    expected = [
        i
        for i in range(start, end)
        if any(n in LINES[i] for n in needles if "\n" not in n)
    ]
    assert list(LineIndex(LINES).containing_any(needles, start, stop)) == expected


PSEUDO_FILE_OUTPUT = """\
oncvpsp: ERROR debl = 0.0
Begin PSPCODE8
oncvpsp: ERROR debl = 1.0
END_PSP
Begin PSP_UPF
<UPF version="2.0.1">
upfout: ERROR iexc = 99 is presently unsupported for UPF output
"""


def test_messages_are_not_searched_for_in_complete_pseudo_files():
    parser = OncvpspTextParser(text=PSEUDO_FILE_OUTPUT, check=False)
    assert parser._pseudo_file_ranges == [(2, 3)]
    # The UPF file was cut short by its error, and is searched
    assert [e["line_numbers"][0] for e in parser.errors] == [1, 7]


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
