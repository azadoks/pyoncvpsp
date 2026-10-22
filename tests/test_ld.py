"""Tests of the parser of the output of `oncvpsp_ld.x` (azadoks/oncvpsp)."""

import gzip
from pathlib import Path

import numpy as np
import pytest

from pyoncvpsp.io import (
    OncvpspLdError,
    OncvpspTextParser,
    ld_command,
    ld_table,
    parse_oncvpsp_ld,
)
from pyoncvpsp.io.ld import LD_COLUMNS

LD_DIR = Path(__file__).parent / "data" / "ld"


def read(name: str) -> str:
    with gzip.open(LD_DIR / f"{name}.out.gz", "rt", encoding="latin-1") as f:
        return f.read()


def test_default_is_the_diagnostics_block():
    # Without -r, -l or -e, oncvpsp_ld.x prints the log derivatives of the run's DIAGNOSTICS block
    table = parse_oncvpsp_ld(read("oncvpsp_ld-sr_14_Si"), returncode=0)
    parser = OncvpspTextParser(text=read("oncvpsp-4.0.1-sr_14_Si_ps"))
    diagnostics = parser.diagnostics_ld_table
    assert tuple(table) == tuple(diagnostics) == LD_COLUMNS
    assert len(table["l"]) == 800
    for column in LD_COLUMNS:
        np.testing.assert_array_equal(table[column], diagnostics[column])
    assert sorted(set(table["l"])) == [0, 1, 2, 3] and set(table["ikap"]) == {1}


def test_radii_and_channels():
    # Fully relativistic, -r 2.5,3 -l 0,1 -e -1,0.5,0.1: a block per l, j and radius
    table = parse_oncvpsp_ld(read("oncvpsp_ld-fr_14_Si_r2.5,3_l0,1"))
    blocks = sorted(set(zip(table["l"], table["ikap"], table["r"])))
    assert blocks == [
        (0, 1, 2.5),
        (0, 1, 3.0),
        (1, 1, 2.5),
        (1, 1, 3.0),
        (1, 2, 2.5),
        (1, 2, 3.0),
    ]
    assert len(table["l"]) == 6 * 15
    assert table["e_ha"].max() == pytest.approx(0.5) and table["e_ha"].min() > -1.0
    assert table["l"].dtype.kind == "i" and table["r"].dtype.kind == "f"


@pytest.mark.parametrize(
    ("text", "returncode", "match"),
    [
        (
            "oncvpsp_ld: ERROR --channels must be values of l from 0 to 3 separated by commas\n",
            1,
            "--channels",
        ),
        ("read_ae_file: ERROR cannot open /nonexistent\n", None, "cannot open"),
        ("# ld_r l ikap r(bohr)\nld_r   0   1  2.5E+00\n", 2, "status 2"),
        ("build: oncvpsp_ld.x (scalar-relativistic)\n", 0, "no log derivatives"),
    ],
)
def test_errors(text, returncode, match):
    with pytest.raises(OncvpspLdError, match=match):
        parse_oncvpsp_ld(text, returncode=returncode)


def test_ld_line_outside_its_block():
    with pytest.raises(ValueError):
        ld_table(["ld_r 0 1 2.5", "ld 1 1 0.1 0.2 0.3"])


def test_empty_table():
    table = ld_table([])
    assert tuple(table) == LD_COLUMNS and all(len(v) == 0 for v in table.values())
    assert (
        OncvpspTextParser(
            text="ONCVPSP  (Optimized Norm-Conservinng Vanderbilt PSeudopotential)\n",
            check=False,
        )
        .diagnostics_ld_table["l"]
        .size
        == 0
    )


def test_ld_command():
    assert ld_command("Si.ae", "Si.ps") == [
        "oncvpsp_ld.x",
        "-a",
        "Si.ae",
        "-p",
        "Si.ps",
    ]
    assert ld_command(
        "Si.ae",
        Path("Si.ps"),
        radii=[2.5, 3],
        channels=[0, 1],
        energies=(-1, 0.5, 0.1),
        executable="/opt/bin/oncvpsp_ld.x",
    ) == [
        "/opt/bin/oncvpsp_ld.x",
        "-a",
        "Si.ae",
        "-p",
        "Si.ps",
        "-r",
        "2.5,3.0",
        "-l",
        "0,1",
        "-e",
        "-1.0,0.5,0.1",
    ]
