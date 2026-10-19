"""Regression and consistency tests over a minimal set of real ONCVPSP/METAPSP outputs.

The outputs in `data/outputs` were selected to cover every program version, relativistic treatment,
error, warning, and output feature seen in a large collection of runs (see `data/outputs/README.md`).
Regenerate the snapshots with `pytest --update-snapshots` after an intended change to the parser.
"""

import json
import re

import pytest

from pyoncvpsp.io.input import OncvpspInput
from pyoncvpsp.io.output import OncvpspTextParser


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
        "ecut": 25.0,
        "error": 1.0e-4,
        "ecut_pseudo_waves": None,
        "factor": None,
        "r_s": 0.695197,
        "rc_min": 2.703023,
        "r_core_max": None,
        "a": 5.36621515,
        "b": -6.23065052,
        "c": 1.87082158,
        "ecut_needed": 24.8513,
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
