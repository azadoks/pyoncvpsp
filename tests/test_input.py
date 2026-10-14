"""Tests for the ONCVPSP input model."""

import io

import pytest
from pydantic import ValidationError

from pyoncvpsp.io.input import OncvpspInput

# ONCVPSP's own nitrogen example (tests/data/07_N.dat)
N_DAT = """\
# ATOM AND REFERENCE CONFIGURATION
# atsym, z, nc, nv, iexc   psfile
    N    7     1   2   3   psp8
#
# n, l, f  (nc+nv lines)
    1    0    2.0
    2    0    2.0
    2    1    3.0
#
# PSEUDOPOTENTIAL AND OPTIMIZATION
# lmax
    1
#
# l, rc, ep, ncon, nbas, qcut  (lmax+1 lines, l's must be in order)
    0    1.70    0.00    4    8    8.00
    1    1.70    0.00    4    8    8.00
#
# LOCAL POTENTIAL
# lloc, lpopt, rc(5), dvloc0
    4    5    1.5    0.0
#
# VANDERBILT-KLEINMAN-BYLANDER PROJECTORs
# l, nproj, debl  (lmax+1 lines, l's in order)
    0    2    1.50
    1    2    1.50
#
# MODEL CORE CHARGE
# icmod, fcfact
    0    0.1
#
# LOG DERIVATIVE ANALYSIS
# epsh1, epsh2, depsh
   -2.0  3.0  0.02
#
# OUTPUT GRID
# rlmax, drl
    4.0  0.01
#
# TEST CONFIGURATIONS
# ncnf
    3
#
#   nvcnf (repeated ncnf times)
#   n, l, f  (nvcnf lines, repeated follwing nvcnf's ncnf times)
    2
    2    0    2.0
    2    1    2.0
#
    2
    2    0    1.0
    2    1    3.0
#
    2
    2    0    1.0
    2    1    2.0
"""


@pytest.fixture
def nitrogen() -> OncvpspInput:
    return OncvpspInput.from_dat_string(N_DAT)


def test_from_dat_string(nitrogen):
    assert nitrogen.oncvpsp.atsym == "N"
    assert nitrogen.oncvpsp.z == 7.0
    assert nitrogen.oncvpsp.iexc == 3
    assert [state.f for state in nitrogen.reference_configuration] == [2.0, 2.0, 3.0]
    assert nitrogen.vkb_projectors.nproj == [2, 2]
    assert len(nitrogen.test_configurations) == 3
    assert nitrogen.log_derivative_analysis.rxpsh is None


@pytest.mark.parametrize("with_comment", [True, False])
def test_dat_round_trip(nitrogen, with_comment):
    assert (
        OncvpspInput.from_dat_string(nitrogen.as_dat_string(with_comment=with_comment))
        == nitrogen
    )


def test_toml_and_dict_round_trip(nitrogen):
    assert OncvpspInput.from_toml_string(nitrogen.as_toml_string()) == nitrogen
    assert OncvpspInput.from_dict(nitrogen.as_dict()) == nitrogen
    buffer = io.StringIO()
    nitrogen.to_toml_file(buffer)
    buffer.seek(0)
    assert OncvpspInput.from_toml_file(buffer) == nitrogen


@pytest.mark.parametrize(
    ("atom", "atsym", "z"),
    [
        ("    N    7", "N", 7.0),
        # ONCVPSP echoes the atom line with format (a2,f6.2), which fuses the fields for z >= 100
        ("  Og118.00", "Og", 118.0),
        ("  Og 118.00", "Og", 118.0),
    ],
)
def test_atom_line(atom, atsym, z):
    lines = list[str](N_DAT.splitlines())
    lines[2] = f"{atom}     1   2   3   psp8"
    parsed = OncvpspInput.from_dat_string("\n".join(lines))
    assert parsed.oncvpsp.atsym == atsym
    assert parsed.oncvpsp.z == z


def test_metapsp_r2scan_and_icmod5_round_trip(nitrogen):
    metapsp = nitrogen.merge(
        oncvpsp={"iexc": 5},
        model_core_charge={"icmod": 5, "fcfact": 0.5, "rcfact": 0.25},
    )
    dat = metapsp.as_dat_string()
    assert OncvpspInput.from_dat_string(dat) == metapsp


@pytest.mark.parametrize(
    "update",
    [
        {"oncvpsp": {"iexc": 0}},
        {"oncvpsp": {"iexc": 6}},
        {"oncvpsp": {"psfile": "xml"}},
        {"oncvpsp": {"atsym": "Abc"}},
        {"model_core_charge": {"icmod": 6}},
        {"model_core_charge": {"icmod": 1, "fcfact": 0.0}},
        {"output_grid": {"rlmax": 1.0}},
    ],
)
def test_invalid_input(nitrogen, update):
    with pytest.raises(ValidationError):
        nitrogen.merge(update)


def test_libxc_functional(nitrogen):
    assert nitrogen.merge(oncvpsp={"iexc": -101130}).oncvpsp.iexc == -101130
