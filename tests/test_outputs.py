"""Regression and consistency tests over a minimal set of real ONCVPSP/METAPSP outputs.

The outputs in `data/outputs` were selected to cover every program version, relativistic treatment,
error, warning, and output feature seen in a large collection of runs (see `data/outputs/README.md`).
Regenerate the snapshots with `pytest --update-snapshots` after an intended change to the parser.
"""

import json
from pathlib import Path
import re

import pytest

from pyoncvpsp.io.input import OncvpspInput
from pyoncvpsp.io.output import OncvpspTextParser

DATA_DIR = Path(__file__).parent / "data"


def test_snapshot(output_name, parse_output, snapshot):
    snapshot(output_name, parse_output(output_name).output)


def test_to_dict_is_json_serializable(output_name, parse_output):
    json.dumps(parse_output(output_name).to_dict(), allow_nan=True)


def test_check(output_name, parse_output, read_output):
    parser = parse_output(output_name)
    exceptions = parser.check(doraise=False)
    names = [exception.name for exception in exceptions]
    # Errors of a failed test configuration in a completed run are treated as warnings
    test_config_errors = [
        error for error in parser.errors if parser._is_test_configuration_error(error)
    ]
    fatal_errors = [error for error in parser.errors if error not in test_config_errors]
    if parser.errors:
        assert names == [error["name"] for error in fatal_errors]
    elif not parser.completed:
        assert names == ["incomplete_output"]
    else:
        assert names == []
    if exceptions:
        with pytest.raises(ExceptionGroup):
            OncvpspTextParser(text=read_output(output_name))
    else:
        OncvpspTextParser(text=read_output(output_name))
    with_warnings = parser.check(doraise=False, werror=True)
    assert len(with_warnings) == len(exceptions) + len(test_config_errors) + len(
        parser.warnings
    )


def test_completed_matches_missing_plot_data(output_name, parse_output):
    parser = parse_output(output_name)
    assert parser.completed == (parser.missing_plot_data == [])


def test_message_lines_removed_from_clean_lines(output_name, parse_output):
    parser = parse_output(output_name)
    message_lines = {
        n
        for message in parser.errors + parser.warnings
        for n in message["line_numbers"]
    }
    assert len(parser.clean_lines) == len(parser.raw_lines) - len(message_lines)


def test_completed_output_consistency(completed_output_name, parse_output):
    parser = parse_output(completed_output_name)
    inp = parser.input
    relativistic = parser.is_relativistic
    # Reference configuration: one entry per state, or per (state, j) when fully relativistic
    n_states = sum(
        1 if (not relativistic or ell == 0) else 2
        for ell in inp["reference_configuration"]["l"]
    )
    assert len(parser.reference_configuration) == n_states
    # Every plotted wavefunction is associated with a state, whose eigenvalue is known unless
    # ONCVPSP printed only its target energy (fully-relativistic v4, j = l - 1/2 wellstates)
    for wavefunction in parser.wavefunctions:
        assert (wavefunction["eig_ae"] is not None) != (
            wavefunction.get("energy_guess") is not None
        ), wavefunction["l"]
        assert (
            wavefunction["r"].shape
            == wavefunction["wfn_ae"].shape
            == wavefunction["wfn_ps"].shape
        )
        ell = wavefunction["l"]
        if not relativistic:
            assert wavefunction["j"] is None
        else:
            # j = l + 1/2 for l = 0
            assert wavefunction["j"] in ((ell + 0.5, ell - 0.5) if ell else (0.5,))
    # ONCVPSP v2/v3 plot both optimized pseudo-wavefunctions of every non-local channel with two
    # projectors
    version = parser.program_information["version"]
    two_projector_ls = [
        ell
        for ell, n in zip(inp["vkb_projectors"]["l"], inp["vkb_projectors"]["nproj"])
        if n == 2 and ell != inp["local_potential"]["lloc"]
    ]
    expected = (
        []
        if parser.program_information["program"] == "METAPSP" or version.startswith("4")
        else [
            (ell, j, iproj)
            for ell in two_projector_ls
            for j in (
                [None]
                if not relativistic
                else ([0.5] if ell == 0 else [ell + 0.5, ell - 0.5])
            )
            for iproj in (1, 2)
        ]
    )
    optimized = parser.optimized_pseudo_wavefunctions
    assert [(w["l"], w["j"], w["iproj"]) for w in optimized] == expected
    for wavefunction in optimized:
        assert wavefunction["r"].shape == wavefunction["wfn_ps"].shape
    # All log-derivative blocks share one energy grid
    log_derivatives = parser.log_derivatives
    assert len({len(block["e"]) for block in log_derivatives}) == 1
    for block in log_derivatives:
        assert (
            block["e"].shape
            == block["log_deriv_ae"].shape
            == block["log_deriv_ps"].shape
        )
    # Wellstates and localized bound states are never both reported for the same projector
    nproj = inp["vkb_projectors"]["nproj"]
    for state in parser.wellstate_metadata + parser.localized_bound_states:
        assert isinstance(state["n"], int)
        # The local channel (nproj = 0) also has an all-electron state, e.g. for plotting
        assert 1 <= state["iproj"] <= max(nproj[state["l"]], 1)
    keys = [
        (state["l"], state["j"], state["iproj"])
        for state in parser.wellstate_metadata + parser.localized_bound_states
    ]
    assert len(keys) == len(set(keys))


@pytest.mark.filterwarnings("ignore:`rxpsh` is not supported")
def test_input_round_trip(output_name, parse_output):
    parser = parse_output(output_name)
    if not parser.input:
        pytest.skip("no input echoed in output")
    inp = OncvpspInput.from_dict(parser.input)
    for with_comment in (True, False):
        assert (
            OncvpspInput.from_dat_string(inp.as_dat_string(with_comment=with_comment))
            == inp
        )
    assert OncvpspInput.from_toml_string(inp.as_toml_string()) == inp


def _truncations(lines: list[str]) -> list[tuple[str, int, list[str]]]:
    """Points at which to truncate a completed output, with the sections then expected missing."""

    def first(pattern: str) -> int:
        return next(i for i, line in enumerate(lines) if re.match(pattern, line))

    start = first(r"DATA FOR PLOTTING")
    gnuscript = first(r"^GNUSCRIPT")
    after = ["log_derivatives", "gnuplot_script"]
    return [
        ("before_plot_data", start, ["DATA FOR PLOTTING"]),
        (
            "potentials",
            first(r"^!p\s"),
            [
                "unscreened_semilocal_potentials",
                "charge_densities",
                "wavefunctions",
                "convergence_profiles",
                *after,
            ],
        ),
        (
            "charge_densities",
            first(r"^!r\s"),
            ["charge_densities", "wavefunctions", "convergence_profiles", *after],
        ),
        (
            "wavefunctions",
            first(r"^&\s"),
            ["wavefunctions", "convergence_profiles", *after],
        ),
        ("convergence_profiles", first(r"^!C\s"), ["convergence_profiles", *after]),
        # Partway through the last log-derivative block
        ("log_derivatives", gnuscript - 5, after),
        ("gnuplot_script", gnuscript, ["gnuplot_script"]),
    ]


def test_truncated_output_is_incomplete(completed_output_name, read_output):
    lines = read_output(completed_output_name).splitlines()
    for label, stop, expected in _truncations(lines):
        parser = OncvpspTextParser(lines=lines[:stop], check=False)
        assert not parser.completed, label
        assert parser.missing_plot_data == expected, label
        if not parser.errors:
            (exception,) = parser.check(doraise=False)
            assert exception.name == "incomplete_output"
            assert all(section in exception.description for section in expected)


def test_budgeted_core(parse_output):
    budgeted = parse_output("oncvpsp-4.0.1-sr_14_Si_bc").budgeted_core
    assert budgeted == {
        "icmod": 6,
        "ecut": 25.0,
        "error": 1.0e-4,
        "ecut_pseudo_waves": None,
        "factor": None,
        "budget_source": None,
        "slack": 1.0,
        "r_s": 0.695197,
        "rc_min": 2.703023,
        "r_core_max": None,
        "fill": "Martins",
        "a": 5.36621515,
        "b": -6.23065052,
        "c": 1.87082158,
        "coefficients": None,
        "ecut_needed": 24.8513,
        "r_s_smallest": None,
        "hardness_mismatch": None,
        "candidates": None,
    }
    capped = parse_output("oncvpsp-4.0.1-sr_N_budget_cap").budgeted_core
    assert capped is not None
    assert capped["r_core_max"] == 2.873894
    assert capped["r_s"] < capped["r_core_max"] < capped["rc_min"]
    stopped = parse_output("oncvpsp-4.0.1-sr_N_budget_too_small")
    assert stopped.budgeted_core is None
    assert [error["name"] for error in stopped.errors] == ["modcore_budget_no_model"]
    assert parse_output("oncvpsp-4.0.1-sr_14_Si").budgeted_core is None
    auto = parse_output("oncvpsp-4.0.1-sr_14_Si_bca").budgeted_core
    assert auto is not None
    assert auto["ecut_pseudo_waves"] == 33.5528
    assert auto["factor"] == 1.0
    assert auto["ecut"] == auto["ecut_pseudo_waves"]
    assert auto["ecut_needed"] <= auto["ecut"]


def test_log_mesh(parse_output):
    # the mesh set by the input, echoed after the test configurations with full precision
    parser = parse_output("oncvpsp-4.0.1-sr_14_Si_log_mesh")
    assert parser.input["log_mesh"] == {
        "amesh": 1.004,
        "rr1": 0.0005 / 14,
        "rmax": 50.0,
    }
    for name in ("oncvpsp-4.0.1-sr_14_Si", "oncvpsp-4.0.1-sr_14_Si_bc"):
        assert parse_output(name).input["log_mesh"] == {
            "amesh": None,
            "rr1": None,
            "rmax": None,
        }


def test_split_stages(parse_output, read_output):
    # oncvpsp_ae.x writes the all-electron atom and test configurations to a file, which
    # oncvpsp_ps.x reads and then prints what oncvpsp.x prints
    ae = parse_output("oncvpsp-4.0.1-sr_14_Si_ae")
    assert ae.stage == "ae"
    assert ae.completed
    assert ae.ae_data_file == "si.ae"
    assert ae.missing_plot_data == []
    assert ae.check(doraise=False) == []
    ps = parse_output("oncvpsp-4.0.1-sr_14_Si_ps")
    assert ps.stage == "ps"
    assert ps.completed
    assert ps.ae_data_file is None
    assert ps.reference_configuration == ae.reference_configuration
    assert parse_output("oncvpsp-4.0.1-sr_14_Si").stage is None
    lines = read_output("oncvpsp-4.0.1-sr_14_Si_ae").splitlines()[:-1]
    truncated = OncvpspTextParser(lines=lines, check=False)
    assert not truncated.completed
    (exception,) = truncated.check(doraise=False)
    assert exception.name == "incomplete_output"


def test_budgeted_core_v2(parse_output):
    # azadoks/oncvpsp 9b90162: the minimum-tail fill, slack and the hardness-matched core (icmod 7)
    auto = parse_output("oncvpsp-4.0.1-sr_14_Si_icmod6_auto").budgeted_core
    assert auto["icmod"] == 6 and auto["fill"] == "minimum tail"
    assert auto["a"] is None and len(auto["coefficients"]) == 9
    assert auto["ecut"] == auto["ecut_pseudo_waves"] == 33.5528
    assert auto["slack"] == 1.0 and auto["candidates"] is None
    parser = parse_output("oncvpsp-4.0.1-sr_14_Si_icmod7_slack0.7")
    assert parser.input["model_core_charge"]["slack"] == 0.7
    hardness = parser.budgeted_core
    assert hardness["icmod"] == 7 and hardness["slack"] == 0.7
    assert hardness["ecut_needed"] <= hardness["slack"] * hardness["ecut"]
    assert hardness["hardness_mismatch"] == 3.09e-8
    chosen = [c for c in hardness["candidates"] if c["fits"]]
    assert [c["r_s"] for c in chosen] == [hardness["r_s"]]
    radii = [c["r_s"] for c in hardness["candidates"]]
    assert radii == sorted(radii, reverse=True) and radii[0] == hardness["rc_min"]
    relativistic = parse_output("oncvpsp-4.0.1-fr_14_Si_icmod7").budgeted_core
    assert (
        relativistic["icmod"] == 7
        and relativistic["r_s"] > relativistic["r_s_smallest"]
    )


@pytest.mark.parametrize(
    ("name", "fill", "icmod"),
    [
        ("I_icmod6_martins_fill_fragment", "Martins", 6),
        ("I_icmod7_fragment", "minimum tail", 7),
    ],
)
def test_budgeted_core_fragments(name, fill, icmod):
    # model-core blocks of the routine run on PPG's I densities: a Martins fill, and mixed candidates
    with open(DATA_DIR / "fragments" / f"{name}.txt", encoding="utf-8") as fp:
        budgeted = OncvpspTextParser(
            lines=fp.read().splitlines(), check=False
        ).budgeted_core
    assert budgeted is not None
    assert budgeted["icmod"] == icmod and budgeted["fill"] == fill
    if icmod == 7:
        fills = {c["fill"] for c in budgeted["candidates"]}
        assert fills == {"minimum tail", "Martins"}
        assert budgeted["r_s_smallest"] == 0.974534
    else:
        assert budgeted["coefficients"] is None and budgeted["c"] == 0.187504012


def test_ionic_tail(parse_output):
    # azadoks/oncvpsp 5e706ad: V_loc switched to the frozen-core ionic tail beyond rc, and its diagnostics
    parser = parse_output("oncvpsp-4.0.1-sr_08_O_ionic_tail")
    assert parser.ionic_tail == {"r_start": 1.4531, "r_end": 2.9062, "steps": 5}
    assert parser.coulomb_tail == {
        "alpha_z": 6.2386769285,
        "truncation": [
            {"r": 6.0, "change": 3.348e-08},
            {"r": 10.0, "change": 4.9176e-09},
        ],
        "r_coulomb": 2.8059,
        "max_nonionic": 3.8286e-05,
    }
    relativistic = parse_output("oncvpsp-4.0.1-fr_74_W_ionic_tail")
    assert relativistic.ionic_tail is not None and relativistic.coulomb_tail is not None
    assert parse_output("oncvpsp-4.0.1-sr_14_Si").ionic_tail is None
    assert parse_output("oncvpsp-4.0.1-sr_14_Si").coulomb_tail is None


def test_budget_from_tables(parse_output):
    # azadoks/oncvpsp 59d0ba8: fcfact 0 is the largest Ecut of the convergence tables at the error per electron,
    # ONCVPSP's recommended cutoff
    parser = parse_output("oncvpsp-4.0.1-sr_14_Si_icmod7_tables")
    budgeted = parser.budgeted_core
    assert budgeted is not None
    assert budgeted["budget_source"] == "tables"
    assert budgeted["ecut_pseudo_waves"] == budgeted["ecut"] == 33.5991
    assert budgeted["ecut"] == pytest.approx(parser.ecut_recommended, abs=0.01)
    weighted = parse_output("oncvpsp-4.0.1-sr_14_Si_icmod6_auto").budgeted_core
    assert weighted is not None and weighted["budget_source"] == "weighted"
