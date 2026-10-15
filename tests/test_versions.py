"""Targeted tests of version-specific output handling, checked against the ONCVPSP/METAPSP sources."""

import math

import numpy as np
import pytest

from pyoncvpsp.channels import ChannelKey, ProjectorKey
from pyoncvpsp.io.output import OncvpspOutputError, OncvpspTextParser


def _states(states: list[dict]) -> list[tuple]:
    return [(s["n"], s["l"], s["j"], s["iproj"]) for s in states]


def test_oncvpsp_v2_fully_relativistic_reference_configuration(parse_output):
    # ONCVPSP 2.1.1 prints a second header line (l+1/2, l-1/2) for the reference configuration
    parser = parse_output("oncvpsp-2.1.1-fr_08_O_r")
    assert parser.program_information["version"] == "2.1.1"
    assert parser.is_relativistic
    assert [(c["n"], c["l"], c["j"]) for c in parser.reference_configuration] == [
        (1, 0, 0.5),
        (2, 0, 0.5),
        (2, 1, 1.5),
        (2, 1, 0.5),
    ]
    assert len(parser.log_derivatives) == 3


def test_oncvpsp_v2_test_configuration_failure(parse_output):
    # The pseudo-atom fails to converge in a test configuration, but the run completes
    parser = parse_output("oncvpsp-2.1.1-sr_Al-sp")
    assert parser.completed
    assert [e["name"] for e in parser.errors] == ["lschvkbb_not_converged"]
    assert [w["name"] for w in parser.warnings] == [
        "fully_non_local_ps_atom_no_solution"
    ]
    # The pseudopotential is not affected: the error is fatal only with `werror`
    assert parser.check() == []
    assert parser.failure is None
    assert parser.test_configuration_status == [
        {
            "config_index": 0,
            "completed": False,
            "messages": [
                {
                    "name": "lschvkbb_not_converged",
                    "level": "error",
                    "line_numbers": [297],
                    "lines": [
                        "psatom: lschvkbb convergence error n,l,okb,iter=   1   0   1   2"
                    ],
                },
                {
                    "name": "fully_non_local_ps_atom_no_solution",
                    "level": "warning",
                    "line_numbers": [298, 299],
                    "lines": [
                        "run_config: ERROR for fully non-local  PS atom,",
                        " no output for this configuration",
                    ],
                },
            ],
        }
    ]
    assert parser.test_configuration_energies == [None]
    with pytest.raises(ExceptionGroup):
        parser.check(werror=True)


def test_oncvpsp_v3_plots_no_projectors(parse_output):
    # ONCVPSP v3 '@' blocks are the pseudo wavefunctions of the projector states, not projectors
    parser = parse_output("oncvpsp-3.3.0-fr_La-sp_r")
    assert any(line.startswith("@") for line in parser.raw_lines)
    assert parser.vkb_projectors == []


def test_oncvpsp_v3_fully_relativistic_negative_energy_states(parse_output):
    # wellstate_r finds a bound 4f state for both j = l +/- 1/2 at the maximum barrier and uses it
    # in place of a wellstate; n and kappa are reported only in the following renormalization line
    parser = parse_output("oncvpsp-3.3.0-fr_La-sp_r")
    assert [w["name"] for w in parser.warnings] == [
        "wellstate_negative_energy_state"
    ] * 2
    localized = parser.localized_bound_states
    assert _states(localized) == [(4, 3, 3.5, 1), (4, 3, 2.5, 1)]
    assert [ChannelKey.from_record(s).kappa for s in localized] == [-4, 3]
    assert [s["eig_ae"] for s in localized] == [-0.155292, -0.165419]
    # The second projectors are wellstates
    assert (5, 3, 3.5, 2) in _states(parser.wellstate_metadata)
    assert (5, 3, 2.5, 2) in _states(parser.wellstate_metadata)
    # One wavefunction per (l, j), at the energy of the first projector state
    f_states = [w for w in parser.wavefunctions if w["l"] == 3]
    assert [(w["n"], w["j"], w["iproj"], w["is_bound"]) for w in f_states] == [
        (4, 3.5, 1, True),
        (4, 2.5, 1, True),
    ]
    assert [w["eig_ae"] for w in f_states] == [-0.155292, -0.165419]


def test_oncvpsp_v3_wellstate_for_local_channel(parse_output):
    # ONCVPSP v3 also computes an all-electron state for the local channel (nproj = 0)
    parser = parse_output("oncvpsp-3.2.2-sr_H.singleProjector")
    assert parser.input["vkb_projectors"]["nproj"] == [1, 0]
    assert _states(parser.wellstate_metadata) == [(2, 1, None, 1)]
    assert [
        (w["n"], w["l"], w["iproj"], w["is_bound"]) for w in parser.wavefunctions
    ] == [
        (1, 0, 1, True),
        (2, 1, 1, False),
    ]


def test_oncvpsp_v3_ghost_without_sign(parse_output):
    # ONCVPSP v3 only tests for bound ghosts, and does not print a sign
    parser = parse_output("oncvpsp-3.3.0-sr_Ba-sp-high")
    assert [(g["l"], g["sign"], g["j"]) for g in parser.ghosts] == [(0, "-", None)]


def test_oncvpsp_v3_fused_atom_line(parse_output):
    parser = parse_output("oncvpsp-3.2.3-fr_Og-d_r")
    assert parser.input["oncvpsp"]["atsym"] == "Og"
    assert parser.input["oncvpsp"]["z"] == 118.0


def test_oncvpsp_v4_fully_relativistic(parse_output):
    parser = parse_output("oncvpsp-4.0.1-fr_74_W_r")
    assert parser.program_information["relativistic_treatment"] == "relativistic"
    assert parser.is_relativistic
    # Log derivatives for l = 0..3, with both j for l > 0
    assert [(ld["l"], ld["j"]) for ld in parser.log_derivatives] == [
        (0, 0.5),
        (1, 1.5),
        (1, 0.5),
        (2, 2.5),
        (2, 1.5),
        (3, 3.5),
        (3, 2.5),
    ]
    assert parser.vkb_spin_orbit_projector_coefficients
    # One plotted projector per wavefunction
    assert len(parser.vkb_projectors) == len(parser.wavefunctions)


def test_metapsp(parse_output):
    parser = parse_output("metapsp-1.0.1-nr_20_Ca2_m")
    assert parser.program_information["program"] == "METAPSP"
    assert not parser.is_relativistic
    inp = parser.input
    assert inp["oncvpsp"]["iexc"] == 5
    assert inp["model_core_charge"]["icmod"] == 5
    # METAPSP plots kinetic energy densities
    densities = parser.kinetic_energy_densities
    assert (
        densities["r"].shape == densities["tau_ps"].shape == densities["tau_mod"].shape
    )
    assert densities["r"].size > 0
    # METAPSP runs no test configurations, so there are no pseudo-atom eigenvalues
    assert parser.test_configurations == []
    assert all(math.isnan(w["eig_ps"]) for w in parser.wavefunctions)
    # A localized bound state is used in place of a wellstate for the 3d projector
    assert [w["name"] for w in parser.warnings] == [
        "localized_bound_state_for_projector"
    ]
    assert _states(parser.localized_bound_states) == [(3, 2, None, 1)]
    (state,) = [w for w in parser.wavefunctions if (w["l"], w["iproj"]) == (2, 1)]
    assert state["is_bound"] and state["n"] == 3 and state["f"] == 0.0
    assert np.isclose(state["eig_ae"], -0.06614626)


def test_error_before_plot_data(parse_output):
    parser = parse_output("oncvpsp-4.0.1-sr_pspot_has_node")
    assert not parser.completed
    assert parser.missing_plot_data == ["DATA FOR PLOTTING"]
    assert [e["name"] for e in parser.errors] == ["first_pswf_has_node"]
    # The ONCVPSP error is reported instead of the incomplete output
    with pytest.raises(ExceptionGroup) as excinfo:
        parser.check()
    assert [
        e.name for e in excinfo.value.exceptions if isinstance(e, OncvpspOutputError)
    ] == ["first_pswf_has_node"]
    assert parser.wavefunctions == []
    assert parser.log_derivatives == []


def test_oncvpsp_v3_negative_energy_second_projector(parse_output):
    # ep + debl < 0 for the second s projector: solved as a bound state (n = np + 1), reported
    # only in a warning and the following renormalization line (oncvpsp_r.f90:451-464)
    parser = parse_output("oncvpsp-3.2.3-fr_Og-d_r")
    assert [
        (s["n"], s["l"], s["j"], s["iproj"], s["eig_ae"])
        for s in parser.localized_bound_states
    ] == [(8, 0, 0.5, 2, -0.052635)]


def test_oncvpsp_v4_fully_relativistic_partner_wellstates(parse_output):
    # j = l - 1/2 wellstates are solved in the well of their j = l + 1/2 partner, and ONCVPSP
    # prints only their target energy (oncvpsp_r.F90:486-503)
    parser = parse_output("oncvpsp-4.0.1-fr_74_W_r")
    partners = [s for s in parser.wellstate_metadata if s["j"] == s["l"] - 0.5]
    assert [(s["l"], s["iproj"]) for s in partners] == [
        (1, 2),
        (1, 3),
        (2, 2),
        (2, 3),
        (3, 1),
        (3, 2),
    ]
    assert all(s["eig_ae"] is None for s in partners)
    assert [s["energy_guess"] for s in partners] == [
        3.1429,
        5.0095,
        4.8113,
        9.8232,
        0.1,
        3.1,
    ]
    others = [s for s in parser.wellstate_metadata if s not in partners]
    assert all(s["eig_ae"] is not None and s["energy_guess"] is None for s in others)
    # The layout predicts every reported wellstate
    layout = parser.projector_layout
    assert layout is not None
    for state in parser.wellstate_metadata:
        assert layout.get(ProjectorKey.from_record(state)) is not None


# The outputs below were generated from clean upstream builds: ONCVPSP 4.0.1 at oncvpsp/oncvpsp
# dc150ab, the ONCVPSP 3.3.1 release tarball, and METAPSP at azadoks/metapsp bde6d65.


@pytest.mark.parametrize(
    "name",
    [
        "oncvpsp-4.0.1-sr_12_Mg_valence_gt_nproj",
        "metapsp-1.0.2-nr_20_Ca_valence_gt_nproj",
    ],
)
def test_v4_valence_states_beyond_nproj_are_not_projector_states(parse_output, name):
    # Two s valence states with nproj = 1: only the first is used (oncvpsp.F90:422)
    parser = parse_output(name)
    assert parser.completed
    s_valence = [c for c in parser.reference_configuration if c["l"] == 0][-2:]
    assert len(s_valence) == 2
    assert parser.input["vkb_projectors"]["nproj"][0] == 1
    layout = parser.projector_layout
    assert layout is not None
    (slot,) = layout.channel(ChannelKey(0))
    assert slot.is_valence and slot.n == s_valence[0]["n"]
    assert list(parser._valence_projector_states) == [
        s.projector for s in layout if s.is_valence
    ]
    assert [(w["n"], w["iproj"]) for w in parser.wavefunctions if w["l"] == 0] == [
        (s_valence[0]["n"], 1)
    ]


def test_v3_fully_relativistic_negative_energy_second_projector_l1(parse_output):
    # ep + debl < 0 for the second p projector of both j: each is solved as a bound 4p state and
    # reported only in a warning (oncvpsp_r.f90:451-464); the s channel has a second valence state
    parser = parse_output("oncvpsp-3.3.1-fr_20_Ca_negative_second")
    assert parser.completed
    assert [w["name"] for w in parser.warnings] == [
        "negative_energy_second_projector"
    ] * 2
    # Both are printed before the header of their j-channel, the second after the l = 1 headers of
    # the first j-channel; the `cnorm` lines before them tell which l they are about
    assert [(m.l, m.iproj) for m in parser.messages] == [(1, 2), (1, 2)]
    assert [
        (s["n"], s["l"], s["j"], s["iproj"], s["eig_ae"])
        for s in parser.localized_bound_states
    ] == [(4, 1, 1.5, 2, -0.052734), (4, 1, 0.5, 2, -0.053432)]
    assert parser.wellstate_metadata == []


def test_v4_fully_relativistic_localized_bound_state(parse_output):
    # wellstate_r returns a localized bound 3d state for j = 5/2 (reported only in a warning), and
    # j = 3/2 is solved in the same (unmodified) potential, so it is a localized bound state too;
    # ONCVPSP prints only its target energy (oncvpsp_r.F90:483-503, wellstate_r.f90:83-91)
    parser = parse_output("oncvpsp-4.0.1-fr_22_Ti_localized")
    assert parser.completed
    assert [
        (s["n"], s["l"], s["j"], s["iproj"], s["eig_ae"], s["energy_guess"])
        for s in parser.localized_bound_states
    ] == [(3, 2, 2.5, 1, -1.13874299, None), (3, 2, 1.5, 1, None, 0.0)]
    # The moderately localized (p) and negative-ep (d) fallbacks give wellstates at 0.05 and 0.25
    assert [
        (s["l"], s["j"], s["iproj"], s["eig_ae"], s["energy_guess"])
        for s in parser.wellstate_metadata
    ] == [
        (1, 1.5, 2, 0.05, None),
        (1, 0.5, 2, None, -0.9933),
        (2, 2.5, 2, 0.25, None),
        (2, 1.5, 2, None, -0.1413),
    ]
    d_first = [w for w in parser.wavefunctions if (w["l"], w["iproj"]) == (2, 1)]
    assert [(w["j"], w["is_bound"], w["f"]) for w in d_first] == [
        (2.5, True, 0.0),
        (1.5, True, 0.0),
    ]


@pytest.mark.parametrize(
    "name",
    ["oncvpsp-4.0.1-sr_N_test_config_failed", "oncvpsp-4.0.1-fr_N_test_config_failed"],
)
def test_v4_failed_test_configuration(parse_output, read_output, name):
    # The pseudo-atom of test configuration 3 does not converge; run_config prints a warning and
    # moves on to configuration 4 (run_config.f90:239-246, run_config_r.f90:257-266)
    parser = parse_output(name)
    assert parser.completed and parser.failure is None
    assert [len(states) for states in parser.test_configurations] == (
        [4, 4, 4, 0, 4] if parser.is_relativistic else [3, 3, 3, 0, 3]
    )
    status = parser.test_configuration_status
    assert [s["config_index"] for s in status] == [0, 1, 2, 3, 4]
    assert [s["completed"] for s in status] == [True, True, True, False, True]
    assert [s["messages"] for s in status if s["config_index"] != 3] == [[]] * 4
    failed = status[3]["messages"]
    assert [(m["name"], m["level"]) for m in failed] == [
        ("lschvkbb_not_converged", "error"),
        ("fully_non_local_ps_atom_no_solution", "warning"),
    ]
    # The main program numbers the header from 0 = reference (jj - 1), but run_config prints jj
    assert failed[1]["lines"][-1] == " WARNING no output for configuration 4"
    for message in parser.messages:
        assert message.config_index == 3
        assert message.implicated_parameters == ["test_configurations__2"]
    energies = parser.test_configuration_energies
    assert energies[3] is None
    assert all(e is not None for i, e in enumerate(energies) if i != 3)
    # The errors of a failed test configuration are not fatal
    assert parser.check() == []
    OncvpspTextParser(text=read_output(name))
    assert [e.name for e in parser.check(doraise=False, werror=True)] == [
        "lschvkbb_not_converged",
        "fully_non_local_ps_atom_no_solution",
    ]


def test_test_configuration_energies(parse_output):
    # run_config prints D exponents (run_config.f90:282-288)
    parser = parse_output("oncvpsp-4.0.1-sr_N_test_config_failed")
    energies = parser.test_configuration_energies
    assert energies[1] == {
        "etot_ae_ref": -5.44530351e01,
        "etot_ae_test": -5.40301448e01,
        "etot_ae_diff": 4.23e-01,
        "etot_ps_ref": -9.66950850e00,
        "etot_ps_test": -9.24688666e00,
        "etot_ps_diff": 4.23e-01,
        "excitation_error": 2.68e-04,
    }
    assert energies[0] is not None and energies[0]["etot_ae_diff"] == 0.0
    assert [e["excitation_error"] for e in energies if e is not None] == [
        1.42e-14,
        2.68e-04,
        3.69e-04,
        6.49e-05,
    ]
    # The reference energies are the same for all configurations
    assert len({(e["etot_ae_ref"], e["etot_ps_ref"]) for e in energies if e}) == 1


def test_test_configuration_energies_fully_relativistic(parse_output):
    # run_config_r of ONCVPSP v4 prints E exponents (run_config_r.f90:330-336)
    parser = parse_output("oncvpsp-4.0.1-fr_74_W_r")
    energies = parser.test_configuration_energies
    assert len(energies) == len(parser.test_configurations) == 4
    assert energies[1] is not None
    assert energies[1]["etot_ae_ref"] == -1.61564270e04
    assert energies[1]["etot_ae_test"] == -1.61560913e04
    assert energies[1]["excitation_error"] == -1.32e-04


@pytest.mark.parametrize(
    "name", ["oncvpsp-2.1.1-fr_08_O_r", "oncvpsp-3.3.1-sr_F", "oncvpsp-4.0.1-sr_14_Si"]
)
def test_test_configuration_energies_are_consistent(parse_output, name):
    parser = parse_output(name)
    energies = parser.test_configuration_energies
    assert len(energies) == len(parser.test_configurations)
    for e in energies:
        assert e is not None
        # Differences and the excitation error are printed with 3 significant digits
        assert math.isclose(
            e["etot_ae_diff"],
            e["etot_ae_test"] - e["etot_ae_ref"],
            rel_tol=1e-2,
            abs_tol=1e-9,
        )
        assert math.isclose(
            e["etot_ps_diff"],
            e["etot_ps_test"] - e["etot_ps_ref"],
            rel_tol=1e-2,
            abs_tol=1e-9,
        )
        assert math.isclose(
            e["excitation_error"],
            e["etot_ae_diff"] - e["etot_ps_diff"],
            rel_tol=0.0,
            abs_tol=2e-2 * max(abs(e["etot_ae_diff"]), 1e-9),
        )
    assert all(s["completed"] for s in parser.test_configuration_status)


def test_build_line_relativistic_treatment_is_trusted(parse_output):
    # The non-relativistic ONCVPSP 4.0.1 executable says scalar-relativistic in its header; the
    # build line printed by azadoks/oncvpsp builds names the executable and its real treatment
    parser = parse_output("oncvpsp-4.0.1-nr_08_O_build_info")
    assert parser.raw_lines[1].startswith("scalar-relativistic version 4.0.1")
    assert parser.program_information["relativistic_treatment"] == "non-relativistic"
    assert not parser.is_relativistic
    assert parser.build_information == {
        "executable": "oncvpspnr.x",
        "relativistic_treatment": "non-relativistic",
        "commit": "1f1125869b08df3cea3de95cc7c0ce84d1983f99",
        "source": "clean",
        "source_diff": None,
        "compiled": "2026-10-15T10:11:10Z",
    }
    assert parser.program_information["build"] == parser.build_information


def test_build_line_modified_source(read_output):
    lines = read_output("oncvpsp-4.0.1-sr_14_Si").splitlines()
    build = (
        "build: oncvpspr_ps.x (fully-relativistic), commit 1f1125869b08df3cea3de95cc7c0ce84d1983f99,"
        " source modified (diff b7a4a15da1ec), compiled 2026-10-15T10:06:21Z"
    )
    parser = OncvpspTextParser(
        lines=lines[:9] + [build, "", ""] + lines[9:], check=False
    )
    build = parser.build_information
    assert build is not None
    assert build["source"] == "modified"
    assert build["source_diff"] == "b7a4a15da1ec"
    assert parser.program_information["relativistic_treatment"] == "fully-relativistic"
    assert parser.is_relativistic


def test_no_build_line(parse_output):
    parser = parse_output("oncvpsp-4.0.1-sr_14_Si")
    assert parser.build_information is None
    assert parser.program_information["build"] is None
    assert parser.program_information["relativistic_treatment"] == "scalar-relativistic"
