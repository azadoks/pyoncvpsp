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
    assert list(edges["partner"]) == ["ps", "missing"]
    assert list(tables["wall_failed"]["k"]) == [0.1]
    assert list(tables["wall_failed"]["atom"]) == ["ps"]


def test_core_band():
    # Nd with an f ghost, the defaults: the 4f core band has no pseudopotential partner, in
    # each of the three spheres, and every other edge pairs (azadoks/oncvpsp 5731bf5)
    tables = parse_oncvpsp_confine(read("oncvpsp_confine-sr_60_Nd_GHOST"), returncode=0)
    edges = tables["edges"]
    core = edges["partner"] == "core"
    assert (
        core.sum() == 9
        and set(edges["l"][core]) == {3}
        and set(edges["n"][core]) == {4}
    )
    assert (
        np.isnan(edges["e_ps"][core]).all() and not np.isnan(edges["e_ps"][~core]).any()
    )
    assert edges["e_ae"][core][0] == pytest.approx(-0.2698804675547143)
    assert len(tables["extra"]["l"]) == 0 and len(tables["paired_off"]["l"]) == 0
    summary = tables["summary"]
    assert summary["n_missing"] == summary["n_extra"] == summary["n_paired_off"] == 0
    assert summary["n_wall_failed"] == 0
    assert summary["max_de"] == pytest.approx(1.001499741151296e-04)
    assert summary["max_dde"] == pytest.approx(1.143558909461717e-03)


def test_ghost():
    # The same files, -l 3 -e -55,-45,0.1 -w none: the bound f ghost at -50.06 Ha has no
    # all-electron partner, its centre and top edges in each sphere
    tables = parse_oncvpsp_confine(
        read("oncvpsp_confine-sr_60_Nd_GHOST_l3_e-55,-45_wnone"), returncode=0
    )
    assert len(tables["edges"]["l"]) == 0
    extra = tables["extra"]
    assert set(extra["l"]) == {3} and set(extra["ikap"]) == {1}
    assert sorted(zip(extra["rs"], extra["edge"])) == [
        (3.25, 2),
        (3.25, 3),
        (pytest.approx(3.9), 2),
        (pytest.approx(3.9), 3),
        (5.2, 2),
        (5.2, 3),
    ]
    assert extra["e_ps"][0] == pytest.approx(-50.05600858013628)
    summary = tables["summary"]
    assert (
        summary["n_extra"] == 6 and summary["n_missing"] == summary["n_paired_off"] == 0
    )
    assert summary["n_wall_failed"] == -1 and np.isnan(summary["max_dde"])
    assert len(tables["wall"]["k"]) == 0 and tables["wall_r0"] is None


def test_paired_off():
    tables = confine_tables(
        [
            "ws_r   1   1  3.5000000000000000E+00",
            "ws   1   1   2   2 -1.0000000000000000E-01 -9.0000000000000000E-02",
            "ws_paired_off   1   1   2   2 -1.0100000000000000E-01",
        ]
    )
    assert tables["paired_off"] == {
        "l": [1],
        "ikap": [1],
        "rs": [3.5],
        "edge": [2],
        "n": [2],
        "e_ps": [pytest.approx(-0.101)],
    }


def test_summary():
    tables = parse_oncvpsp_confine(read("oncvpsp_confine-sr_14_Si"))
    assert tables["summary"] == {
        "max_de": pytest.approx(6.617900241756569e-06),
        "ecut4": pytest.approx(7.851065777484514),
        "ecut5": pytest.approx(9.515190644784115),
        "n_missing": 0,
        "n_extra": 0,
        "n_paired_off": 0,
        "n_wall_failed": 0,
        "max_dde": pytest.approx(3.2503680529316625e-05),
        "ecut4_top": pytest.approx(7.851065777485118),
        "ecut5_top": pytest.approx(9.515190644741338),
    }
    # -w none -c 5: no cutoff reached, the walls left out
    summary = confine_tables(
        [
            "confine_summary  1.8065124526289722E-06                Infinity                Infinity"
            "     0     0     0    -1                     NaN                Infinity"
            "                Infinity"
        ]
    )["summary"]
    assert summary["ecut4"] == summary["ecut5"] == float("inf")
    assert summary["ecut4_top"] == summary["ecut5_top"] == float("inf")
    assert summary["n_wall_failed"] == -1 and np.isnan(summary["max_dde"])
    assert isinstance(summary["n_missing"], int)
    # Before b1c6ce3 the line ends at max_dde
    summary = confine_tables(
        [
            "confine_summary  1.8065124526289722E-06  7.85E+00  9.52E+00     0     0     0     0"
            "  3.25E-05"
        ]
    )["summary"]
    assert summary["max_dde"] == 3.25e-05
    assert summary["ecut4_top"] is None and summary["ecut5_top"] is None


def test_empty():
    # -s -w none: only the build line and a summary of left-out parts
    tables = parse_oncvpsp_confine(
        "build: oncvpsp_confine.x (scalar-relativistic)\n"
        "confine_summary                     NaN                     NaN                     NaN"
        "    -1    -1    -1    -1                     NaN\n",
        0,
    )
    assert tables["wall_r0"] is None
    assert all(
        len(column) == 0 for name in CONFINE_COLUMNS for column in tables[name].values()
    )
    assert np.isnan(tables["summary"]["max_de"]) and tables["summary"]["n_extra"] == -1
    assert confine_tables([])["summary"] is None


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
