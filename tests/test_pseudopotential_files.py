"""Tests of the psp8 and PSML parsers and of the quantities of the output that the comparison of
two outputs needs (the optimization diagnostics, the bound ghost tests and the pseudo-atom energy)."""

import numpy as np
import pytest

from pyoncvpsp.io import parse_psml, parse_psp8
from pyoncvpsp.io.output import OncvpspTextParser


def test_psp8_scalar_relativistic(parse_output):
    psp8 = parse_psp8(parse_output("oncvpsp-4.0.1-sr_14_Si").psp8_string)
    assert psp8["title"].startswith("Si    ONCVPSP-4.0.1")
    assert (psp8["zatom"], psp8["zion"], psp8["pspd"]) == (14.0, 4.0, "260111")
    assert (
        psp8["pspcod"],
        psp8["pspxc"],
        psp8["lmax"],
        psp8["lloc"],
        psp8["mmax"],
    ) == (
        8,
        2,
        2,
        4,
        500,
    )
    assert (psp8["rchrg"], psp8["fchrg"], psp8["qchrg"]) == (4.99, 0.25, 0.0)
    assert psp8["nproj"] == [2, 2, 2]
    assert psp8["extension_switch"] == [1, 1]
    assert psp8["nprojso"] is None
    assert psp8["r"].shape == (500,)
    assert psp8["r"][:3] == pytest.approx([0.0, 0.01, 0.02])
    assert psp8["local_potential"].shape == (500,)
    assert [(p["l"], p["kind"], p["beta"].shape) for p in psp8["projectors"]] == [
        (0, "scalar", (2, 500)),
        (1, "scalar", (2, 500)),
        (2, "scalar", (2, 500)),
    ]
    assert psp8["projectors"][0]["ekb"] == pytest.approx(
        [2.8244596943684, -5.5131063148786e-02]
    )
    assert psp8["model_core"].shape == (5, 500)
    assert psp8["densities"].shape == (3, 500)
    assert psp8["input"].startswith("#")
    assert "# ATOM AND REFERENCE CONFIGURATION" in psp8["input"]


def test_psp8_spin_orbit(parse_output):
    psp8 = parse_psp8(parse_output("oncvpsp-4.0.1-fr_14_Si_icmod7").psp8_string)
    assert psp8["extension_switch"][0] == 3
    assert psp8["nprojso"] == [4, 4]
    assert [(p["l"], p["kind"], len(p["ekb"])) for p in psp8["projectors"]] == [
        (0, "scalar", 2),
        (1, "scalar", 2),
        (2, "scalar", 2),
        (1, "spin_orbit", 4),
        (2, "spin_orbit", 4),
    ]
    # fchrg = 1 flags the model core of icmod 7
    assert psp8["fchrg"] == 1.0
    assert psp8["model_core"].shape == (5, psp8["mmax"])


def test_psp8_rejects_other_formats():
    with pytest.raises(ValueError, match="pspcod"):
        parse_psp8(
            "title\n 14.0 4.0 0 zatom\n 6 2 2 4 500 0 pspcod\n"
            " 0.0 0.0 0.0 rchrg\n 2 2 2 nproj\n 1 1 ext\n 0\n"
        )


def test_psml(parse_output):
    psml = parse_psml(parse_output("oncvpsp-4.0.1-sr_08_O_ionic_tail").psml_string)
    npts = len(psml["grid"])
    assert npts == 2292
    assert psml["header"]["atomic-label"] == "O"
    assert "# ATOM AND REFERENCE CONFIGURATION" in psml["input"]
    assert [(s["n"], s["l"], s["j"]) for s in psml["semilocal_potentials"]] == [
        (2, 0, None),
        (2, 1, None),
        (3, 2, None),
    ]
    assert all(len(s["values"]) == npts for s in psml["semilocal_potentials"])
    assert [(p["set"], p["l"], p["seq"]) for p in psml["projectors"]][:3] == [
        ("scalar_relativistic", 0, 1),
        ("scalar_relativistic", 0, 2),
        ("scalar_relativistic", 1, 1),
    ]
    assert psml["projectors"][0]["ekb"] == pytest.approx(5.93747314193)
    assert all(p["r"] is None for p in psml["projectors"])
    assert len(psml["local_potential"]["values"]) == npts
    assert psml["valence_charge"]["total-charge"] == 6
    assert psml["core_charge"] is not None
    assert len(psml["core_charge"]["values"]) == npts


def test_psml_rejects_other_documents():
    with pytest.raises(ValueError, match="not a PSML document"):
        parse_psml("<upf></upf>")


def test_optimization_diagnostics(parse_output):
    diagnostics = parse_output("oncvpsp-4.0.1-sr_14_Si").optimization_diagnostics
    assert [(d["l"], d["iproj"]) for d in diagnostics] == [
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    ]
    first = diagnostics[0]
    assert first["constraints"]["pswf"] == pytest.approx(
        [0.20169637, -0.18168790, 0.13441969, -0.02256484, -0.24449972]
    )
    assert np.abs(first["constraints"]["diff"]).max() < 1e-12
    assert first["norm_fraction"] == 0.770856
    assert (first["qcut"], first["ecut"]) == (5.01, 12.53)
    assert (first["q_infinity_ke"], first["e_inf"]) == (20.0, 200.0)
    for diagnostic in diagnostics:
        assert all(value is not None for key, value in diagnostic.items() if key != "j")
        assert diagnostic["ke_ratio"] == pytest.approx(1.0, abs=1e-6)


def test_bound_ghost_tests(parse_output):
    rows = parse_output("oncvpsp-4.0.1-sr_14_Si").bound_ghost_tests
    assert [(r["set"], r["n"], r["l"]) for r in rows] == [(None, 1, 0), (None, 2, 1)]
    assert rows[0]["eig"] == pytest.approx(rows[0]["eig_basis"], abs=1e-4)
    fully_relativistic = parse_output("oncvpsp-4.0.1-fr_14_Si_icmod7").bound_ghost_tests
    assert {r["set"] for r in fully_relativistic} == {"+", "-"}


def test_pseudoatom_total_energy(parse_output):
    assert parse_output("oncvpsp-4.0.1-sr_14_Si").pseudoatom_total_energy == -3.792023
    # a large energy runs into the label
    parser = OncvpspTextParser(
        lines=["Pseudoatom total energy-1269.498648"], check=False
    )
    assert parser.pseudoatom_total_energy == -1269.498648
