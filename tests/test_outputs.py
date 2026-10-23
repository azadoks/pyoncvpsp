"""Regression and consistency tests over a minimal set of real ONCVPSP/METAPSP outputs.

The outputs in `data/outputs` were selected to cover every program version, relativistic treatment,
error, warning, and output feature seen in a large collection of runs (see `data/outputs/README.md`).
Regenerate the snapshots with `pytest --update-snapshots` after an intended change to the parser.
"""

import json
from pathlib import Path
import re

import numpy as np
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
    weighted = parse_output("oncvpsp-4.0.1-sr_14_Si_budget_weighted")
    core = weighted.budgeted_core
    assert core["budget_source"] == "weighted_mean"
    assert core["ecut_pseudo_waves"] == 9.3091
    assert weighted.input["model_core_charge"]["budget_cutoff"] == "weighted"
    diag = weighted.diagnostics["ecut_weighted"]
    assert diag["error"] == 1.0e-4
    assert diag["ecut"] == pytest.approx(9.3090796161463860)
    assert [(w["l"], w["iproj"], w["weight"]) for w in diag["weights"]] == [
        (0, 1, 2.0),
        (1, 1, 2.0),
        (2, 1, 10.0),
    ]
    assert (
        parse_output("oncvpsp-4.0.1-sr_14_Si_bca").diagnostics.get("ecut_weighted")
        is None
    )


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


def test_c_light_and_upf_mesh(parse_output):
    # c_light echoed at full precision at the end of the atom line, upf_mesh after rlmax, drl
    parser = parse_output("oncvpsp-4.0.1-sr_14_Si_new_options")
    assert parser.input["oncvpsp"]["iexc"] == 7
    assert parser.input["oncvpsp"]["c_light"] == 137.0359895
    assert parser.input["output_grid"]["upf_mesh"] == 3500
    assert parser.completed
    assert 'functional="RXC  VWN  NOGX NOGC"' in parser.upf_string
    assert 'mesh_size="  2166"' in parser.upf_string
    for name in ("oncvpsp-4.0.1-sr_14_Si", "oncvpsp-4.0.1-sr_14_Si_log_mesh"):
        parsed = parse_output(name).input
        assert parsed["oncvpsp"]["c_light"] is None
        assert parsed["output_grid"]["upf_mesh"] == 0


def test_ae_reference_energies(parse_output):
    # azadoks/oncvpsp 35f1150 on: the residual of the screening potential and the energy terms
    energies = parse_output("oncvpsp-4.0.1-sr_14_Si_new_options").ae_reference_energies
    assert energies == {
        "iterations": 25,
        "total_energy": -288.696531,
        "screening_residual": 4.96e-11,
        "terms": {
            "total": -288.6965311033,
            "kinetic": 288.8633056401,
            "electron_nucleus": -690.0657357738,
            "hartree": 131.9675528627,
            "exchange_correlation": -19.4616538323,
        },
    }
    terms = energies["terms"]
    assert terms["total"] == pytest.approx(
        terms["kinetic"]
        + terms["electron_nucleus"]
        + terms["hartree"]
        + terms["exchange_correlation"],
        abs=1e-9,
    )
    # earlier builds print the iterations and the total energy only
    older = parse_output("oncvpsp-4.0.1-sr_14_Si").ae_reference_energies
    assert older is not None
    assert older["screening_residual"] is None and older["terms"] is None


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


def test_diagnostics(parse_output):
    # azadoks/oncvpsp O-6: the DIAGNOSTICS block, the Ecut crossings and log derivatives to 17 digits,
    # the printed tables rounded from them
    parser = parse_output("oncvpsp-4.0.1-sr_07_N_opt_diagnostics")
    diagnostics = parser.diagnostics
    crossings = diagnostics["ecut_crossings"]
    assert [(c["l"], c["j"], c["iproj"]) for c in crossings] == [
        (0, None, 1),
        (0, None, 2),
        (1, None, 1),
        (1, None, 2),
    ]
    for crossing in crossings:
        assert crossing["eresid"] == pytest.approx(1e-2 * 10.0 ** (-0.5 * np.arange(7)))
    for profile in parser._optimize_convergence_profiles:
        crossing = next(
            c
            for c in crossings
            if (c["l"], c["iproj"]) == (profile["l"], profile["iproj"])
        )
        assert profile["ecut"] == pytest.approx(crossing["ecut"][::2], abs=0.005)
    for printed, full in zip(
        parser.log_derivatives, diagnostics["log_derivatives"], strict=True
    ):
        assert (printed["l"], printed["j"]) == (full["l"], full["j"])
        assert full["rxpsh"] == pytest.approx(printed["rxpsh"], abs=0.005)
        for key in ("e", "log_deriv_ae", "log_deriv_ps"):
            assert full[key] == pytest.approx(printed[key], abs=5e-7)
    # X-3: the ghost tests' margins, the printed bound-state rows rounded from them
    margins = diagnostics["ghost_margins"]
    assert [(m["l"], m["j"], m["n"]) for m in margins] == [(0, None, 1), (1, None, 2)]
    for printed, full in zip(parser.bound_ghost_tests, margins, strict=True):
        assert (printed["n"], printed["l"]) == (full["n"], full["l"])
        for key in ("eig", "eig_basis"):
            assert full[key] == pytest.approx(printed[key], abs=5e-7)
    localization = diagnostics["ghost_localization"]
    assert [(g["l"], g["j"]) for g in localization] == [(0, None), (1, None)]
    assert all(g["r_over_rc"] > 1.0 for g in localization)
    relativistic = parse_output("oncvpsp-4.0.1-fr_14_Si_diagnostics").diagnostics
    assert [(m["l"], m["j"], m["n"]) for m in relativistic["ghost_margins"]] == [
        (0, 0.5, 1),
        (1, 1.5, 2),
        (1, 0.5, 2),
        (2, 2.5, 3),
        (2, 1.5, 3),
    ]
    # l = 2 has no bound state: compared with the vacuum level
    assert relativistic["ghost_margins"][3]["eig"] == 0.0
    assert [(c["l"], c["j"]) for c in relativistic["log_derivatives"]] == [
        (0, 0.5),
        (1, 1.5),
        (1, 0.5),
        (2, 2.5),
        (2, 1.5),
        (3, 3.5),
        (3, 2.5),
    ]
    assert {(c["l"], c["j"], c["iproj"]) for c in relativistic["ecut_crossings"]} == {
        (ell, j, iproj)
        for ell, j in [(0, 0.5), (1, 1.5), (1, 0.5), (2, 2.5), (2, 1.5)]
        for iproj in (1, 2)
    }
    assert parse_output("oncvpsp-4.0.1-sr_14_Si").diagnostics == {}


def test_diagnostics_full(parse_output):
    # azadoks/oncvpsp 9889e33: Ecut crossings to 1e-7 with ekin_check, the projectors'
    # conditioning, the test configurations and the model core's extent, to 17 digits
    parser = parse_output("oncvpsp-4.0.1-sr_14_Si_diagnostics_full")
    diagnostics = parser.diagnostics
    for crossing in diagnostics["ecut_crossings"]:
        assert crossing["eresid"] == pytest.approx(
            1e-2 * 10.0 ** (-0.5 * np.arange(11))
        )
        assert abs(crossing["ekin_check"]) < 1e-8
    conditioning = diagnostics["projector_conditioning"]
    assert [(c["l"], c["j"]) for c in conditioning] == [(0, None), (1, None), (2, None)]
    for channel in conditioning:
        b = channel["b_eigenvalues"]
        assert list(b) == sorted(b)
        assert channel["condition"] == pytest.approx(abs(b).max() / abs(b).min())
        coefficients = channel["coefficients"]
        assert list(abs(coefficients)) == sorted(abs(coefficients), reverse=True)
    configurations = diagnostics["test_configurations"]
    assert [c["index"] for c in configurations] == [0, 1, 2, 3]
    assert not any(c["failed"] for c in configurations)
    assert configurations[0]["de_ae"] == 0.0
    for full in configurations:
        assert full["error"] == pytest.approx(full["de_ae"] - full["de_ps"], abs=1e-15)
        assert [(s["n"], s["l"], s["j"]) for s in full["states"]] == [
            (3, 0, None),
            (3, 1, None),
        ]
    assert configurations[1]["states"][0]["occupation"] == 1.0
    core = diagnostics["model_core"]
    assert core["rc"] == pytest.approx(2.7)
    assert [e["fraction"] for e in core["extent"]] == [1e-2, 1e-4]
    for extent in core["extent"]:
        assert core["rc"] < extent["r_ae"] == pytest.approx(extent["r_model"])
    assert core["max"]["rho"] > core["min"]["rho"] == pytest.approx(core["rho_model"])
    relativistic = parse_output("oncvpsp-4.0.1-fr_14_Si_diagnostics_full").diagnostics
    assert [(c["l"], c["j"]) for c in relativistic["projector_conditioning"]] == [
        (0, 0.5),
        (1, 1.5),
        (1, 0.5),
        (2, 2.5),
        (2, 1.5),
    ]
    assert [
        (s["l"], s["j"]) for s in relativistic["test_configurations"][0]["states"]
    ] == [
        (0, 0.5),
        (1, 1.5),
        (1, 0.5),
    ]
    # an output from before these lines has none of them
    old = parse_output("oncvpsp-4.0.1-sr_07_N_opt_diagnostics").diagnostics
    assert old["projector_conditioning"] == old["test_configurations"] == []
    assert old["model_core"] is None


def test_diagnostics_vloc_cutoff(parse_output):
    # azadoks/oncvpsp e7655cd: the cutoffs of the local potential's Fourier convergence at
    # the table's levels, 1e-2 down by half decades to 1e-5, and the transforms' check
    vloc = parse_output("oncvpsp-4.0.1-sr_14_Si_vloc_cutoff").diagnostics["vloc_cutoff"]
    assert vloc["eresid"] == pytest.approx(1e-2 * 10.0 ** (-0.5 * np.arange(7)))
    assert list(vloc["ecut"]) == sorted(vloc["ecut"])
    assert vloc["ecut"][0] > 0.0
    assert abs(vloc["check"]) < 1e-9
    old = parse_output("oncvpsp-4.0.1-sr_14_Si_diagnostics_full").diagnostics
    assert old["vloc_cutoff"] is None


def test_diagnostics_optimization(parse_output):
    # azadoks/oncvpsp 9bb82b7: per optimized wave, E_resid at qcut, the regularization's term
    # and e1 and |f1| of E_resid(qcut) alone; at qcut 22 the l = 0 problem is flat to round-off
    crossings = parse_output("oncvpsp-4.0.1-sr_07_N_flat").diagnostics["ecut_crossings"]
    for crossing in crossings:
        assert crossing["regularization"] > 0.0
        assert crossing["eresid_qcut"] > 0.0 and crossing["f1"] >= 0.0
    flat = [c for c in crossings if c["l"] == 0 and c["iproj"] == 1][0]
    assert flat["e1"] < 1e-12 and flat["f1"] < 1e-12
    old = parse_output("oncvpsp-4.0.1-sr_14_Si_vloc_cutoff").diagnostics[
        "ecut_crossings"
    ]
    assert all("e1" not in c for c in old)


def test_diagnostics_well_share(parse_output):
    # azadoks/oncvpsp 74b1c54: E_resid at qcut of each confined well state with the well's tail
    # and the free tail; the second s state, at debl 13.5, is above qcut's wave vector
    parser = parse_output("oncvpsp-4.0.1-sr_14_Si_well_share")
    crossings = {(c["l"], c["iproj"]): c for c in parser.diagnostics["ecut_crossings"]}
    assert {k for k, c in crossings.items() if "eresid_qcut_well" in c} == {
        (0, 2),
        (1, 2),
        (2, 1),
        (2, 2),
    }
    assert crossings[0, 2]["eresid_qcut_well"] is None
    assert crossings[0, 2]["eresid_qcut_free"] is None
    for key in [(1, 2), (2, 1), (2, 2)]:
        # the well's value is the minimization's own E_resid at qcut
        assert crossings[key]["eresid_qcut_well"] == pytest.approx(
            crossings[key]["eresid_qcut"], rel=1e-10
        )
        assert crossings[key]["eresid_qcut_free"] > 0.0
    assert [(m.name, m.l, m.iproj) for m in parser.messages] == [
        ("well_state_qcut_below_k", 0, 2),
        ("well_tail_share", 1, 2),
    ]
    old = parse_output("oncvpsp-4.0.1-sr_07_N_flat").diagnostics["ecut_crossings"]
    assert all("eresid_qcut_well" not in c for c in old)


def test_confined_well(parse_output):
    # azadoks/oncvpsp 4286074, well_tail = "confined": the wall's radius in place of the
    # asymptotic potential and half-point radius, and well_n in the input echo
    parser = parse_output("oncvpsp-4.0.1-sr_14_Si_confined")
    assert parser.input["pseudopotentials"]["well_tail"] == "confined"
    assert parser.input["vkb_projectors"]["well_n"] == [0, 5, 0]
    states = {(s["l"], s["iproj"]): s for s in parser.wellstate_metadata}
    assert {k: s["n"] for k, s in states.items()} == {
        (0, 2): 4,
        (1, 2): 5,
        (2, 1): 3,
        (2, 2): 4,
    }
    for state in states.values():
        assert state["wall_radius"] > 2.7
        assert state["asymptotic_potential"] is None
        assert state["half_point_radius"] is None
    # the confined states have the well's share too
    crossings = parser.diagnostics["ecut_crossings"]
    assert {(c["l"], c["iproj"]) for c in crossings if "eresid_qcut_well" in c} == set(
        states
    )
    assert parser.messages == []
    hamann = parse_output("oncvpsp-4.0.1-sr_14_Si_well_share")
    assert all(s["wall_radius"] is None for s in hamann.wellstate_metadata)


def test_ecut_weight_band_cutoffs(parse_output):
    # azadoks/oncvpsp 270c1e6 and 3eff290: ecut_weight in the input echo, and the cutoffs at the
    # valence energies of the d channel, which has no occupied state
    parser = parse_output("oncvpsp-4.0.1-sr_14_Si_ecut_weight")
    assert parser.input["vkb_projectors"]["ecut_weight"] == [-1.0, -1.0, 0.3]
    assert parser.input["vkb_projectors"]["well_n"] is None
    bands = parser.diagnostics["band_cutoffs"]
    assert [(b["l"], b["j"]) for b in bands] == [(2, None)] * 3
    assert [b["e"] for b in bands] == sorted(b["e"] for b in bands)
    for band in bands:
        assert band["eresid"] == pytest.approx([1e-2, 1e-3, 1e-4, 1e-5])
        assert list(band["ecut"]) == sorted(band["ecut"])
        assert 0.0 < band["eresid_qcut"] < 1e-4
    # the band rows, which follow the VKB diagnostic tests, are not tests
    assert {t["l"] for t in parser.diagnostic_tests_vkb} == {0, 1, 2}
    assert len(parser.diagnostic_tests_vkb) == 6
    assert (
        parse_output("oncvpsp-4.0.1-sr_14_Si_confined").diagnostics["band_cutoffs"]
        == []
    )


def test_vkb_bound_ghost(parse_output):
    # azadoks/oncvpsp 61e586a: lschvkbb's ierr 200 is a bound ghost of the projectors, and its
    # message is three lines
    parser = parse_output("oncvpsp-4.0.1-sr_3_Li_vkb_bound_ghost")
    assert parser.failure is not None
    assert parser.failure.name == "vkb_bound_ghost"
    assert parser.failure.l == 0
    assert len(parser.failure.primary.line_numbers) == 3
    assert not any(
        "bound ghost of the projectors" in line for line in parser.clean_lines
    )


def test_free_tail_too_fast(parse_output):
    # azadoks/oncvpsp e781800: with the free tail, a well state at a negative energy whose tail
    # grows too fast for its window at qcut (kappa L > 12) stops the run
    failure = parse_output("oncvpsp-4.0.1-sr_53_I_free_tail_too_fast").failure
    assert failure is not None
    assert failure.name == "free_tail_too_fast"
    assert failure.l == 2
    assert len(failure.primary.line_numbers) == 2
    assert float(failure.primary.groups["qcut"]) == pytest.approx(1.43262)
