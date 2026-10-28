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


def test_rxpsh_no_warning(nitrogen, recwarn):
    """azadoks/oncvpsp reads rxpsh (and checks it against the core radii): writing it warns of nothing."""
    nitrogen.log_derivative_analysis.rxpsh = 2.5
    text = nitrogen.as_dat_string()
    assert not recwarn.list
    assert OncvpspInput.from_dat_string(text).log_derivative_analysis.rxpsh == 2.5


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
        {"oncvpsp": {"iexc": 8}},
        {"oncvpsp": {"psfile": "xml"}},
        {"oncvpsp": {"atsym": "Abc"}},
        {"model_core_charge": {"icmod": 2, "fcfact": -0.5}},
        {"model_core_charge": {"icmod": 8, "fcfact": 25.0}},
        {"model_core_charge": {"icmod": 4, "slack": 0.7}},
        {"model_core_charge": {"icmod": 6, "slack": 0.0}},
        {"model_core_charge": {"icmod": 7, "slack": 1.5}},
        {
            "model_core_charge": {
                "icmod": 3,
                "fcfact": 1.0,
                "rcfact": 1.0,
                "budget_cutoff": "weighted",
            }
        },
        {"model_core_charge": {"icmod": 6, "budget_cutoff": "mean"}},
        {"model_core_charge": {"icmod": 1, "fcfact": 0.0}},
        {"output_grid": {"rlmax": 1.0}},
        {"log_mesh": {"amesh": 0.999}},
        {"log_mesh": {"rr1": 1.7}},
        {"log_mesh": {"rmax": 4.0}},
        {"log_mesh": {"adams_order": 3}},
        {"log_mesh": {"adams_order": 9}},
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
        "adams_order": None,
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


@pytest.mark.parametrize("iexc", [6, 7])
def test_vwn_functionals(nitrogen_flags, iexc):
    # azadoks/oncvpsp 522b58a: 6 Slater + VWN5, 7 with the relativistic exchange at c_light; 5 is METAPSP's
    vwn = nitrogen_flags.merge(oncvpsp={"iexc": iexc})
    assert OncvpspInput.from_dat_string(vwn.as_dat_string()) == vwn
    assert OncvpspInput.from_toml_string(vwn.as_toml_string()) == vwn


# ep_as_given, well_onset and well_tail: the optional second to fourth values of the lmax line
# (azadoks/oncvpsp, phase 8 of doc/smoothness_plan.md), and keys of [pseudopotentials] in TOML
N_WELL_DAT = N_DAT.replace("# lmax\n    1\n", "# lmax\n    1    1    1.2    1\n")


@pytest.fixture
def nitrogen_well() -> OncvpspInput:
    return OncvpspInput.from_dat_string(N_WELL_DAT)


def test_well_options_default(nitrogen):
    psps = nitrogen.pseudopotentials
    assert not psps.ep_as_given and psps.well_onset is None and psps.well_tail == "well"
    assert not psps.has_well_options
    assert psps.as_dat_string().split("\n")[0].split() == ["1"]
    data = toml.loads(nitrogen.as_toml_string())
    assert not {"ep_as_given", "well_onset", "well_tail"} & set(
        data["pseudopotentials"]
    )


def test_well_options_from_dat_string(nitrogen_well, nitrogen):
    psps = nitrogen_well.pseudopotentials
    assert psps.ep_as_given and psps.well_onset == 1.2 and psps.well_tail == "free"
    assert (
        nitrogen_well.merge(
            pseudopotentials={
                "ep_as_given": False,
                "well_onset": None,
                "well_tail": "well",
            }
        )
        == nitrogen
    )


@pytest.mark.parametrize("with_comment", [True, False])
def test_well_options_dat_round_trip(nitrogen_well, with_comment):
    dat = nitrogen_well.as_dat_string(with_comment=with_comment)
    assert OncvpspInput.from_dat_string(dat) == nitrogen_well


def test_well_options_partial_line(nitrogen):
    # a line that ends early leaves the rest at their defaults; 0 is the default onset
    dat = N_DAT.replace("# lmax\n    1\n", "# lmax\n    1    0    0.0\n")
    assert OncvpspInput.from_dat_string(dat) == nitrogen
    dat = N_DAT.replace("# lmax\n    1\n", "# lmax\n    1    1\n")
    assert OncvpspInput.from_dat_string(dat).pseudopotentials.ep_as_given


def test_well_options_toml(nitrogen_well):
    data = toml.loads(nitrogen_well.as_toml_string())
    psps = data["pseudopotentials"]
    assert psps["ep_as_given"] is True
    assert psps["well_onset"] == 1.2
    assert psps["well_tail"] == "free"
    assert (
        OncvpspInput.from_toml_string(nitrogen_well.as_toml_string()) == nitrogen_well
    )


@pytest.mark.parametrize(
    "update",
    [
        {"pseudopotentials": {"well_onset": 0.99}},
        {"pseudopotentials": {"well_tail": "open"}},
    ],
)
def test_well_options_invalid(nitrogen, update):
    with pytest.raises(ValueError):
        nitrogen.merge(update)


# well_tail = "confined" (2 on the lmax line) and well_n, the optional fourth column of the
# projector lines (azadoks/oncvpsp 4286074)
N_CONFINED_DAT = N_DAT.replace(
    "# lmax\n    1\n", "# lmax\n    1    0    0.0    2\n"
).replace("    1    2    1.50\n", "    1    2    1.50    4\n")


def test_confined_well_n(nitrogen):
    confined = OncvpspInput.from_dat_string(N_CONFINED_DAT)
    assert confined.pseudopotentials.well_tail == "confined"
    assert confined.vkb_projectors.well_n == [0, 4]
    assert [
        line.split() for line in confined.vkb_projectors.as_dat_string().split("\n")
    ] == [["0", "2", "1.50000", "0"], ["1", "2", "1.50000", "4"]]
    assert OncvpspInput.from_dat_string(confined.as_dat_string()) == confined
    assert OncvpspInput.from_toml_string(confined.as_toml_string()) == confined
    assert toml.loads(confined.as_toml_string())["vkb_projectors"]["well_n"] == [0, 4]
    # without well_n the projector lines keep three columns, and TOML leaves it out
    plain = confined.merge(vkb_projectors={"well_n": None})
    assert all(
        len(line.split()) == 3
        for line in plain.vkb_projectors.as_dat_string().split("\n")
    )
    assert "well_n" not in toml.loads(plain.as_toml_string())["vkb_projectors"]
    assert "well_n" not in toml.loads(nitrogen.as_toml_string())["vkb_projectors"]


@pytest.mark.parametrize(
    "update",
    [
        # oncvpsp: ERROR well_n is for well_tail = "confined" only
        {"pseudopotentials": {"well_tail": "well"}},
        {"vkb_projectors": {"well_n": [0, -1]}},
        {"vkb_projectors": {"well_n": [4]}},
    ],
)
def test_confined_well_n_invalid(update):
    with pytest.raises(ValueError):
        OncvpspInput.from_dat_string(N_CONFINED_DAT).merge(update)


# well_norm: the optional sixth value of the lmax line, after wave_regularization, which is then
# written too (azadoks/oncvpsp a575a64)
@pytest.mark.parametrize(
    ("value", "well_norm", "well_tail"),
    [
        (0, None, "well"),
        (1, "space", "well"),
        (2, "rc", "free"),
        (1, "space", "confined"),
    ],
)
def test_well_norm(nitrogen, value, well_norm, well_tail):
    tail = ("well", "free", "confined").index(well_tail)
    dat = N_DAT.replace(
        "# lmax\n    1\n", f"# lmax\n    1    0    0.0    {tail}    0.0    {value}\n"
    )
    read = OncvpspInput.from_dat_string(dat)
    assert read.pseudopotentials.well_norm == well_norm
    assert read == nitrogen.merge(
        pseudopotentials={"well_tail": well_tail, "well_norm": well_norm}
    )
    lmax_line = read.pseudopotentials.as_dat_string().split("\n")[0].split()
    assert len(lmax_line) == (6 if well_norm else 4 if well_tail != "well" else 1)
    assert OncvpspInput.from_dat_string(read.as_dat_string()) == read
    assert OncvpspInput.from_toml_string(read.as_toml_string()) == read
    psps = toml.loads(read.as_toml_string())["pseudopotentials"]
    assert psps.get("well_norm") == well_norm


@pytest.mark.parametrize(
    "update",
    [
        # oncvpsp: ERROR well_norm = "space" (1) is not defined for well_tail = "free"
        {"pseudopotentials": {"well_tail": "free", "well_norm": "space"}},
        {"pseudopotentials": {"well_norm": "matched"}},
    ],
)
def test_well_norm_invalid(nitrogen, update):
    with pytest.raises(ValueError):
        nitrogen.merge(update)


def _with_d_well_state(nitrogen: OncvpspInput, d_ep: float, **pseudopotentials) -> dict:
    # N with a d channel, which has no valence state: its first projector state is a well state
    # at d_ep
    return {
        "pseudopotentials": {
            "lmax": 2,
            "l": [0, 1, 2],
            "rc": [1.7] * 3,
            "ep": [0.0, 0.0, d_ep],
            "ncon": [4] * 3,
            "nbas": [8] * 3,
            "qcut": [8.0] * 3,
            **pseudopotentials,
        },
        "vkb_projectors": {"l": [0, 1, 2], "nproj": [2, 2, 1], "debl": [1.5, 1.5, 0.0]},
    }


# oncvpsp: ERROR well_tail = "free" needs ep >= 0 (azadoks/oncvpsp fd5aaf0)
def test_free_tail_negative_ep_invalid(nitrogen):
    with pytest.raises(ValueError, match="needs ep >= 0"):
        nitrogen.merge(
            _with_d_well_state(nitrogen, -0.1, well_tail="free", ep_as_given=True)
        )


@pytest.mark.parametrize(
    "pseudopotentials",
    [
        # ep >= 0
        {"well_tail": "free", "ep_as_given": True, "ep": [0.0, 0.0, 0.0]},
        # without ep_as_given the program replaces a negative ep
        {"well_tail": "free"},
        # only the free tail needs it
        {"well_tail": "confined", "ep_as_given": True},
        # an ep of a channel with a valence state is not a well state's
        {"well_tail": "free", "ep_as_given": True, "ep": [-0.1, -0.1, 0.0]},
    ],
)
def test_free_tail_negative_ep_valid(nitrogen, pseudopotentials):
    nitrogen.merge(_with_d_well_state(nitrogen, -0.1, **pseudopotentials))


# wave_regularization: the optional fifth value of the lmax line (azadoks/oncvpsp), after the well
# options, and a key of [pseudopotentials] in TOML
def test_wave_regularization_default(nitrogen):
    assert nitrogen.pseudopotentials.wave_regularization == 0.0
    assert nitrogen.pseudopotentials.as_dat_string().split("\n")[0].split() == ["1"]
    data = toml.loads(nitrogen.as_toml_string())
    assert "wave_regularization" not in data["pseudopotentials"]


def test_wave_regularization_from_dat_string(nitrogen):
    dat = N_DAT.replace("# lmax\n    1\n", "# lmax\n    1    0    0.0    0    0.25\n")
    psps = OncvpspInput.from_dat_string(dat).pseudopotentials
    assert psps.wave_regularization == 0.25
    assert not psps.has_well_options
    assert (
        OncvpspInput.from_dat_string(dat).merge(
            pseudopotentials={"wave_regularization": 0.0}
        )
        == nitrogen
    )


@pytest.mark.parametrize("with_comment", [True, False])
@pytest.mark.parametrize("well", [False, True])
def test_wave_regularization_dat_round_trip(
    nitrogen, nitrogen_well, with_comment, well
):
    regularized = (nitrogen_well if well else nitrogen).merge(
        pseudopotentials={"wave_regularization": 0.25}
    )
    dat = regularized.as_dat_string(with_comment=with_comment)
    # the well options are written at their defaults before it
    words = regularized.pseudopotentials.as_dat_string().split("\n")[0].split()
    assert len(words) == 5 and float(words[4]) == 0.25
    assert OncvpspInput.from_dat_string(dat) == regularized


def test_wave_regularization_toml(nitrogen):
    regularized = nitrogen.merge(pseudopotentials={"wave_regularization": 0.25})
    data = toml.loads(regularized.as_toml_string())
    assert data["pseudopotentials"]["wave_regularization"] == 0.25
    assert OncvpspInput.from_toml_string(regularized.as_toml_string()) == regularized


def test_wave_regularization_invalid(nitrogen):
    with pytest.raises(ValueError):
        nitrogen.merge(pseudopotentials={"wave_regularization": -0.1})


# budget_cutoff: the optional fifth value of the icmod 6, 7 line (0 "max", 1 "weighted"), after
# slack, and a key of [model_core_charge] in TOML (azadoks/oncvpsp)
@pytest.mark.parametrize("icmod", [6, 7])
@pytest.mark.parametrize("slack", [1.0, 0.7])
@pytest.mark.parametrize("with_comment", [True, False])
def test_budget_cutoff_dat_round_trip(nitrogen, icmod, slack, with_comment):
    weighted = nitrogen.merge(
        model_core_charge={
            "icmod": icmod,
            "fcfact": 0.0,
            "rcfact": 1.0e-5,
            "slack": slack,
            "budget_cutoff": "weighted",
        }
    )
    words = weighted.model_core_charge.as_dat_string().split()
    assert words == [str(icmod), "0.00000", "1.00000e-05", f"{slack:.5f}", "1"]
    dat = weighted.as_dat_string(with_comment=with_comment)
    assert OncvpspInput.from_dat_string(dat) == weighted


def test_budget_cutoff_default(nitrogen):
    budgeted = nitrogen.merge(
        model_core_charge={"icmod": 6, "fcfact": 0.0, "rcfact": 1.0e-5}
    )
    assert budgeted.model_core_charge.budget_cutoff == "max"
    assert len(budgeted.model_core_charge.as_dat_string().split()) == 3
    assert (
        "budget_cutoff"
        not in toml.loads(budgeted.as_toml_string())["model_core_charge"]
    )
    dat = budgeted.as_dat_string().replace(" 1.00000e-05", " 1.00000e-05    1.0    0")
    assert OncvpspInput.from_dat_string(dat) == budgeted


def test_budget_cutoff_toml(nitrogen):
    weighted = nitrogen.merge(
        model_core_charge={"icmod": 6, "fcfact": 0.0, "budget_cutoff": "weighted"}
    )
    data = toml.loads(weighted.as_toml_string())
    assert data["model_core_charge"]["budget_cutoff"] == "weighted"
    assert OncvpspInput.from_toml_string(weighted.as_toml_string()) == weighted


def test_wave_regularization_echo(nitrogen):
    # the lmax line as azadoks/oncvpsp echoes it in the output, '(i5,i5,es24.16,i5,es24.16)'
    echo = "    1    0  0.0000000000000000E+00    0  2.5000000000000000E-01"
    dat = N_DAT.replace(
        "# lmax\n    1\n",
        f"# lmax, ep_as_given, well_onset, well_tail, wave_regularization\n{echo}\n",
    )
    assert OncvpspInput.from_dat_string(dat) == nitrogen.merge(
        pseudopotentials={"wave_regularization": 0.25}
    )


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        # icmod 6, fcfact 1079: i5 and f10.5 run together in the echo of the output
        (
            "    61079.00000  1.0000E-05   0.70000",
            {"icmod": 6, "fcfact": 1079.0, "rcfact": 1.0e-5, "slack": 0.7},
        ),
        ("    3 1000.00000   1.50000", {"icmod": 3, "fcfact": 1000.0, "rcfact": 1.5}),
    ],
)
def test_model_core_echo_with_wide_fcfact(line, expected):
    dat = N_DAT.replace(
        "# icmod, fcfact\n    0    0.1\n", f"# icmod, fcfact, rcfact\n{line}\n"
    )
    core = OncvpspInput.from_dat_string(dat).model_core_charge
    assert {key: getattr(core, key) for key in expected} == pytest.approx(expected)


# ecut_weight: the optional fifth column of the projector lines, after well_n (azadoks/oncvpsp
# 270c1e6); budget_cutoff "max_all", 2 on the icmod line (9ed4066)
def test_ecut_weight(nitrogen):
    dat = N_DAT.replace("    1    2    1.50\n", "    1    2    1.50    0    0.3\n")
    read = OncvpspInput.from_dat_string(dat)
    assert read.vkb_projectors.ecut_weight == [-1.0, 0.3]
    assert read.vkb_projectors.well_n is None
    assert [
        line.split() for line in read.vkb_projectors.as_dat_string().split("\n")
    ] == [
        ["0", "2", "1.50000", "0", "-1.00000"],
        ["1", "2", "1.50000", "0", "0.30000"],
    ]
    assert OncvpspInput.from_dat_string(read.as_dat_string()) == read
    assert OncvpspInput.from_toml_string(read.as_toml_string()) == read
    assert toml.loads(read.as_toml_string())["vkb_projectors"]["ecut_weight"] == [
        -1.0,
        0.3,
    ]
    # all negative is the default
    assert nitrogen.merge(vkb_projectors={"ecut_weight": [-1.0, -2.0]}) == nitrogen
    assert "ecut_weight" not in toml.loads(nitrogen.as_toml_string())["vkb_projectors"]


def test_budget_cutoff_max_all(nitrogen):
    core = {"icmod": 6, "fcfact": 0.0, "rcfact": 1e-5, "budget_cutoff": "max_all"}
    max_all = nitrogen.merge(model_core_charge=core)
    line = max_all.model_core_charge.as_dat_string().split("\n")[-1].split()
    assert line[-1] == "2"
    assert OncvpspInput.from_dat_string(max_all.as_dat_string()) == max_all
    assert OncvpspInput.from_toml_string(max_all.as_toml_string()) == max_all


def test_adams_order(nitrogen_mesh, nitrogen):
    # the optional fourth value of the log mesh line: written only when given, and in [log_mesh]
    order = nitrogen_mesh.merge(log_mesh={"adams_order": 6})
    for with_comment in (True, False):
        dat = order.as_dat_string(with_comment=with_comment)
        assert dat.splitlines()[-1].split() == ["1.004", "-1.0", "50.0", "6"]
        assert OncvpspInput.from_dat_string(dat) == order
    assert toml.loads(order.as_toml_string())["log_mesh"] == {
        "amesh": 1.004,
        "rmax": 50.0,
        "adams_order": 6,
    }
    # the order alone gives the line too, with the built-in mesh
    alone = nitrogen.merge(log_mesh={"adams_order": 8})
    assert alone.log_mesh.is_set
    assert alone.as_dat_string().splitlines()[-1].split() == [
        "-1.0",
        "-1.0",
        "-1.0",
        "8",
    ]
    assert OncvpspInput.from_dat_string(alone.as_dat_string()) == alone
