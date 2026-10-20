"""Tests for the ONCVPSP input model."""

import io

import pytest
import toml
from pydantic import ValidationError

import numpy as np

from pyoncvpsp.io.input import OncvpspInput
from pyoncvpsp.utils import build_mesh

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


# ONCVPSP's tests/data/toml/07_N.toml, the TOML form of 07_N.dat made by its
# scripts/input_text_to_toml.py, which oncvpsp.x reads with -t
N_TOML = """\
[[test_configurations]]
n = [ 2, 2,]
l = [ 0, 1,]
f = [ 2.0, 2.0,]

[[test_configurations]]
n = [ 2, 2,]
l = [ 0, 1,]
f = [ 1.0, 3.0,]

[[test_configurations]]
n = [ 2, 2,]
l = [ 0, 1,]
f = [ 1.0, 2.0,]

[oncvpsp]
atsym = "N"
z = 7.0
nc = 1
nv = 2
iexc = 3

[linear_mesh]
a = 0.01
rmax = 4.0

[reference_configuration]
n = [ 1, 2, 2,]
l = [ 0, 0, 1,]
f = [ 2.0, 2.0, 3.0,]

[pseudopotentials]
lmax = 1
l = [ 0, 1,]
rc = [ 1.7, 1.7,]
ep = [ 0.0, 0.0,]
ncon = [ 4, 4,]
nbas = [ 8, 8,]
qcut = [ 8.0, 8.0,]

[local_potential]
lloc = 4
lpopt = 5
rcloc = 1.5
dvloc0 = 0.0

[vkb_projectors]
l = [ 0, 1,]
nproj = [ 2, 2,]
debl = [ 1.5, 1.5,]

[model_core_charge]
icmod = 0

[log_derivative_analysis]
epsh1 = -2.0
epsh2 = 3.0
depsh = 0.02

[pp_output]
psfile = "psp8"
"""


def test_toml_layout_matches_oncvpsp(nitrogen):
    data = toml.loads(nitrogen.as_toml_string())
    assert data["linear_mesh"] == {"a": 0.01, "rmax": 4.0}
    assert data["pp_output"] == {"psfile": "psp8"}
    assert "output_grid" not in data
    assert "psfile" not in data["oncvpsp"]


def test_reads_oncvpsp_toml(nitrogen):
    # 07_N.dat gives fcfact 0.1 with icmod 0, which the TOML form leaves out
    assert OncvpspInput.from_toml_string(N_TOML) == nitrogen.merge(
        model_core_charge={"fcfact": 0.0}
    )


def test_reads_previous_toml_layout(nitrogen):
    # pyoncvpsp wrote the model's fields as they are before it wrote ONCVPSP's layout
    assert OncvpspInput.from_toml_string(toml.dumps(nitrogen.model_dump())) == nitrogen


def test_toml_teter_grid_keys(nitrogen):
    grid = {
        "teter_amp_min": 1.0,
        "teter_amp_max": 4.0,
        "teter_amp_step": 0.5,
        "teter_scale_min": 1.0,
        "teter_scale_max": 1.5,
        "teter_scale_step": 0.1,
    }
    teter = nitrogen.merge(model_core_charge={"icmod": 4, **grid})
    core = toml.loads(teter.as_toml_string())["model_core_charge"]
    assert {
        key: core[key] for key in core if key.endswith(("_min", "_max", "_step"))
    } == {
        "fcfact_min": 1.0,
        "fcfact_max": 4.0,
        "fcfact_step": 0.5,
        "rcfact_min": 1.0,
        "rcfact_max": 1.5,
        "rcfact_step": 0.1,
    }
    assert OncvpspInput.from_toml_string(teter.as_toml_string()) == teter


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


@pytest.mark.parametrize("fcfact", [25.0, 0.0, -0.8])
@pytest.mark.parametrize("rcfact", [0.0, 1.0e-4, 3.0e-7])
def test_icmod6_round_trip(nitrogen, fcfact, rcfact):
    budgeted = nitrogen.merge(
        model_core_charge={"icmod": 6, "fcfact": fcfact, "rcfact": rcfact}
    )
    dat = budgeted.as_dat_string()
    assert OncvpspInput.from_dat_string(dat) == budgeted


@pytest.mark.parametrize("icmod", [6, 7])
@pytest.mark.parametrize(
    ("rcfact", "slack"), [(0.0, 1.0), (1.0e-5, 1.0), (1.0e-5, 0.7), (2.5e-6, 0.85)]
)
def test_budgeted_core_round_trip(nitrogen, icmod, rcfact, slack):
    budgeted = nitrogen.merge(
        model_core_charge={
            "icmod": icmod,
            "fcfact": 0.0,
            "rcfact": rcfact,
            "slack": slack,
        }
    )
    dat = budgeted.as_dat_string()
    assert OncvpspInput.from_dat_string(dat) == budgeted


def test_slack_writes_default_error(nitrogen):
    # slack follows rcfact on the line, and ONCVPSP rejects rcfact 0 there
    slack = nitrogen.merge(model_core_charge={"icmod": 7, "fcfact": 0.0, "slack": 0.7})
    words = slack.model_core_charge.as_dat_string().split()
    assert words == ["7", "0.00000", "1.00000e-05", "0.70000"]
    assert (
        OncvpspInput.from_dat_string(slack.as_dat_string()).model_core_charge.rcfact
        == 1.0e-5
    )


@pytest.mark.parametrize(
    "update",
    [
        {"oncvpsp": {"iexc": 0}},
        {"oncvpsp": {"iexc": 6}},
        {"oncvpsp": {"psfile": "xml"}},
        {"oncvpsp": {"atsym": "Abc"}},
        {"model_core_charge": {"icmod": 2, "fcfact": -0.5}},
        {"model_core_charge": {"icmod": 8, "fcfact": 25.0}},
        {"model_core_charge": {"icmod": 4, "slack": 0.7}},
        {"model_core_charge": {"icmod": 6, "slack": 0.0}},
        {"model_core_charge": {"icmod": 7, "slack": 1.5}},
        {"model_core_charge": {"icmod": 1, "fcfact": 0.0}},
        {"output_grid": {"rlmax": 1.0}},
        {"log_mesh": {"amesh": 0.999}},
        {"log_mesh": {"rr1": 1.7}},
        {"log_mesh": {"rmax": 4.0}},
        {"oncvpsp": {"c_light": 0.5}},
        {"output_grid": {"upf_mesh": -1}},
    ],
)
def test_invalid_input(nitrogen, update):
    with pytest.raises(ValidationError):
        nitrogen.merge(update)


def test_libxc_functional(nitrogen):
    assert nitrogen.merge(oncvpsp={"iexc": -101130}).oncvpsp.iexc == -101130


def test_log_mesh_default(nitrogen):
    assert not nitrogen.log_mesh.is_set
    assert "LOG MESH" not in nitrogen.as_dat_string()
    assert "log_mesh" not in toml.loads(nitrogen.as_toml_string())
    np.testing.assert_array_equal(nitrogen.radial_mesh, build_mesh(7.0))


# the optional last line of the text input of azadoks/oncvpsp: amesh, rr1, rmax, with -1 for the
# built-in rr1
N_MESH_DAT = (
    N_DAT
    + """#
# amesh, rr1, rmax
  1.004  -1  50.0
"""
)


@pytest.fixture
def nitrogen_mesh() -> OncvpspInput:
    return OncvpspInput.from_dat_string(N_MESH_DAT)


def test_log_mesh_from_dat_string(nitrogen_mesh, nitrogen):
    assert nitrogen_mesh.log_mesh.model_dump() == {
        "amesh": 1.004,
        "rr1": None,
        "rmax": 50.0,
    }
    assert nitrogen_mesh.merge(log_mesh={"amesh": None, "rmax": None}) == nitrogen
    mesh = nitrogen_mesh.radial_mesh
    np.testing.assert_allclose(mesh[1:] / mesh[:-1], 1.004)
    # as in ONCVPSP, mmax = int(log(rmax / rr1) / log(amesh)) points from rr1, so up to two steps short
    assert mesh[0] == 0.0005 / 10 and mesh[-1] < 50.0 < mesh[-1] * 1.004**2


@pytest.mark.parametrize("with_comment", [True, False])
def test_log_mesh_dat_round_trip(nitrogen_mesh, with_comment):
    dat = nitrogen_mesh.as_dat_string(with_comment=with_comment)
    assert dat.splitlines()[-1].split() == ["1.004", "-1.0", "50.0"]
    assert OncvpspInput.from_dat_string(dat) == nitrogen_mesh


def test_log_mesh_toml(nitrogen_mesh):
    # ONCVPSP's [log_mesh], without the keys that keep the built-in values
    assert toml.loads(nitrogen_mesh.as_toml_string())["log_mesh"] == {
        "amesh": 1.004,
        "rmax": 50.0,
    }
    assert (
        OncvpspInput.from_toml_string(nitrogen_mesh.as_toml_string()) == nitrogen_mesh
    )
    assert OncvpspInput.from_dict(nitrogen_mesh.as_dict()) == nitrogen_mesh


def test_log_mesh_nonpositive_is_default(nitrogen):
    # ONCVPSP takes a value of 0 or less for the built-in one
    assert not nitrogen.merge(
        log_mesh={"amesh": 0, "rr1": -1.0, "rmax": -1}
    ).log_mesh.is_set


def test_build_mesh_relativistic():
    # the fully-relativistic code does not cap rr1
    assert build_mesh(7.0)[0] == 0.0005 / 10
    assert build_mesh(7.0, relativistic=True)[0] == 0.0005 / 7.0


def test_c_light_and_upf_mesh_default(nitrogen):
    assert nitrogen.oncvpsp.c_light is None
    assert nitrogen.output_grid.upf_mesh == 0
    assert len(nitrogen.oncvpsp.as_dat_string().split()) == 6
    assert nitrogen.output_grid.as_dat_string().split() == ["4.00000", "0.01000"]
    data = toml.loads(nitrogen.as_toml_string())
    assert "c_light" not in data["oncvpsp"]
    assert data["pp_output"] == {"psfile": "psp8"}


def test_c_light_nonpositive_is_default(nitrogen):
    # ONCVPSP takes a value of 0 or less for the built-in 137.036
    assert nitrogen.merge(oncvpsp={"c_light": 0}).oncvpsp.c_light is None
    assert nitrogen.merge(oncvpsp={"c_light": -1.0}).oncvpsp.c_light is None


# the optional seventh value of the atom line (c_light) and third value of the output grid line
# (upf_mesh) of azadoks/oncvpsp
N_FLAGS_DAT = N_DAT.replace(
    "    N    7     1   2   3   psp8", "    N    7     1   2   3   upf  137.0359895"
).replace("    4.0  0.01", "    4.0  0.01  1500")


@pytest.fixture
def nitrogen_flags() -> OncvpspInput:
    return OncvpspInput.from_dat_string(N_FLAGS_DAT)


def test_c_light_and_upf_mesh_from_dat_string(nitrogen_flags, nitrogen):
    assert nitrogen_flags.oncvpsp.c_light == 137.0359895
    assert nitrogen_flags.output_grid.upf_mesh == 1500
    assert (
        nitrogen_flags.merge(
            oncvpsp={"c_light": None, "psfile": "psp8"}, output_grid={"upf_mesh": 0}
        )
        == nitrogen
    )


@pytest.mark.parametrize("with_comment", [True, False])
def test_c_light_and_upf_mesh_dat_round_trip(nitrogen_flags, with_comment):
    dat = nitrogen_flags.as_dat_string(with_comment=with_comment)
    assert OncvpspInput.from_dat_string(dat) == nitrogen_flags


def test_c_light_and_upf_mesh_toml(nitrogen_flags):
    # c_light in [oncvpsp] and upf_mesh in [pp_output], as ONCVPSP reads them
    data = toml.loads(nitrogen_flags.as_toml_string())
    assert data["oncvpsp"]["c_light"] == 137.0359895
    assert data["pp_output"] == {"psfile": "upf", "upf_mesh": 1500}
    assert "upf_mesh" not in data["linear_mesh"]
    assert (
        OncvpspInput.from_toml_string(nitrogen_flags.as_toml_string()) == nitrogen_flags
    )
    assert OncvpspInput.from_dict(nitrogen_flags.as_dict()) == nitrogen_flags
