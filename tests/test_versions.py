"""Targeted tests of version-specific output handling, checked against the ONCVPSP/METAPSP sources."""

import math

import numpy as np
import pytest

from pyoncvpsp.channels import ChannelKey, ProjectorKey
from pyoncvpsp.io.output import OncvpspOutputError


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
    with pytest.raises(ExceptionGroup):
        parser.check()


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


def test_v3_fully_relativistic_negative_energy_second_projector_l1(parse_output):
    # ep + debl < 0 for the second p projector of both j: each is solved as a bound 4p state and
    # reported only in a warning (oncvpsp_r.f90:451-464); the s channel has a second valence state
    parser = parse_output("oncvpsp-3.3.1-fr_20_Ca_negative_second")
    assert parser.completed
    assert [w["name"] for w in parser.warnings] == [
        "negative_energy_second_projector"
    ] * 2
    assert [
        (s["n"], s["l"], s["j"], s["iproj"], s["eig_ae"])
        for s in parser.localized_bound_states
    ] == [(4, 1, 1.5, 2, -0.052734), (4, 1, 0.5, 2, -0.053432)]
    assert parser.wellstate_metadata == []
