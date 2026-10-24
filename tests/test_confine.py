"""Tests of the parser of the output of `oncvpsp_confine.x` (azadoks/oncvpsp)."""

import gzip
from pathlib import Path

import numpy as np
import pytest

from pyoncvpsp.io import (
    OncvpspConfineError,
    confine_command,
    confine_tables,
    parse_oncvpsp_confine,
)
from pyoncvpsp.io.confine import CONFINE_COLUMNS

CONFINE_DIR = Path(__file__).parent / "data" / "confine"


def read(name: str) -> str:
    with gzip.open(CONFINE_DIR / f"{name}.out.gz", "rt", encoding="latin-1") as f:
        return f.read()


def test_defaults():
    # Scalar-relativistic Si, the defaults: rs at 1.25, 1.5 and 2 times rc = 2.7, all four l,
    # the walls 0.01, 0.03 and 0.1 after the free atom
    tables = parse_oncvpsp_confine(read("oncvpsp_confine-sr_14_Si"), returncode=0)
    for name, columns in CONFINE_COLUMNS.items():
        assert tuple(tables[name]) == columns
    edges = tables["edges"]
    assert len(edges["l"]) == 81 and not np.isnan(edges["e_ps"]).any()
    blocks = sorted(set(zip(edges["l"], edges["ikap"], edges["rs"])))
    assert len(blocks) == 12
    assert sorted({b[2] for b in blocks}) == pytest.approx([3.375, 4.05, 5.4])
    assert set(edges["edge"]) == {1, 2, 3} and edges["n"].dtype.kind == "i"
    assert edges["e_ae"][0] == pytest.approx(-0.538074269593285)
    assert edges["e_ps"][0] == pytest.approx(-0.5380771098484938)
    assert len(tables["ecut"]["l"]) == 405 and len(tables["floor"]["l"]) == 81
    assert sorted(set(tables["ecut"]["tol"]))[::-1] == pytest.approx(
        [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    )
    assert len(tables["conv"]["l"]) == 0
    assert tables["wall_r0"] == pytest.approx(2.7)
    wall = tables["wall"]
    assert list(wall["k"]) == pytest.approx([0.0, 0.01, 0.03, 0.1])
    assert wall["de_ae"][0] == 0.0 and wall["de_ps"][-1] == pytest.approx(
        0.05524784067914146
    )
    eig = tables["wall_eig"]
    assert (
        len(eig["k"]) == 8
        and list(eig["n"][:2]) == [3, 3]
        and list(eig["l"][:2]) == [0, 1]
    )
    assert len(tables["wall_failed"]["k"]) == 0


def test_fully_relativistic_table():
    # Fully relativistic, -l 1 -r 3.5 -t -w 0.05: both j of p, the ws_conv lines, one wall
    tables = parse_oncvpsp_confine(read("oncvpsp_confine-fr_14_Si_l1_r3.5_t_w0.05"))
    edges = tables["edges"]
    assert set(zip(edges["l"], edges["ikap"], edges["rs"])) == {
        (1, 1, 3.5),
        (1, 2, 3.5),
    }
    conv = tables["conv"]
    assert len(conv["l"]) == 254 and set(conv["ikap"]) == {1, 2}
    assert list(tables["wall"]["k"]) == pytest.approx([0.0, 0.05])
    assert set(tables["wall_eig"]["ikap"]) == {1, 2}


def test_missing_and_failed():
    tables = confine_tables(
        [
            "ws_r   2   1  4.0000000000000000E+00",
            "ws   2   1   1   3 -1.0000000000000000E-01 -1.0000000000000000E-01",
            "ws_floor   2   1   1   3  2.0000000000000000E+02  1.0000000000000000E-09",
            "ws_missing   2   1   2   4  5.0000000000000000E-01",
            "wall_r0  2.7000000000000002E+00",
            "wall  0.0000000000000000E+00 -2.8E+02 -3.7E+00  0.0E+00  0.0E+00",
            "wall_failed  1.0000000000000001E-01 ps",
        ]
    )
    edges = tables["edges"]
    assert list(edges["edge"]) == [1, 2] and list(edges["n"]) == [3, 4]
    assert edges["e_ae"][1] == 0.5 and np.isnan(edges["e_ps"][1])
    assert list(tables["wall_failed"]["k"]) == [0.1]
    assert list(tables["wall_failed"]["atom"]) == ["ps"]


def test_empty():
    # -s -w none: only the build line
    tables = parse_oncvpsp_confine(
        "build: oncvpsp_confine.x (scalar-relativistic)\n", 0
    )
    assert tables["wall_r0"] is None
    assert all(
        len(column) == 0 for name in CONFINE_COLUMNS for column in tables[name].values()
    )


@pytest.mark.parametrize(
    ("text", "returncode", "match"),
    [
        (
            "oncvpsp_confine: ERROR the radius   1.00000 is not beyond the largest core radius"
            "   2.70000 bohr\n",
            1,
            "not beyond",
        ),
        ("# ws_r l ikap rs(bohr)\nws_r   0   1  3.375E+00\n", 2, "status 2"),
    ],
)
def test_errors(text, returncode, match):
    with pytest.raises(OncvpspConfineError, match=match):
        parse_oncvpsp_confine(text, returncode=returncode)


def test_sphere_line_outside_its_block():
    with pytest.raises(ValueError):
        confine_tables(["ws_r 0 1 3.0", "ws 1 1 1 2 -0.1 -0.1"])


def test_confine_command():
    assert confine_command("Si.ae", "Si.ps") == [
        "oncvpsp_confine.x",
        "-a",
        "Si.ae",
        "-p",
        "Si.ps",
    ]
    assert confine_command(
        "Si.ae",
        Path("Si.ps"),
        radii=[3.5],
        channels=[1],
        energies=(-1, 0.5, 0.1),
        ecut_max=300,
        table=True,
        sphere=False,
        walls=[0.05],
        wall_radius=2.5,
        executable="/opt/bin/oncvpsp_confine.x",
    ) == [
        "/opt/bin/oncvpsp_confine.x",
        "-a",
        "Si.ae",
        "-p",
        "Si.ps",
        "-r",
        "3.5",
        "-l",
        "1",
        "-e",
        "-1.0,0.5,0.1",
        "-c",
        "300.0",
        "-t",
        "-s",
        "-w",
        "0.05",
        "-R",
        "2.5",
    ]
    assert confine_command("Si.ae", "Si.ps", walls=[])[-2:] == ["-w", "none"]
