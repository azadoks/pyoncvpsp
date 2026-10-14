"""Tests for the ONCVPSP/METAPSP error and warning definitions."""

import re

import pytest

from pyoncvpsp.io.output import ERRORS, WARNINGS
from pyoncvpsp.io.output._text import OncvpspTextParser

DEFINITIONS = [*WARNINGS, *ERRORS]

# Messages as written by the ONCVPSP 4.0.1, 3.3.1 and METAPSP sources, and the definition each
# should match. Similar messages from different subroutines must be told apart.
MESSAGES = [
    # ONCVPSP 4
    (
        "oncvpsp: ERROR all-electron reference atom not converged",
        "ae_reference_not_converged",
    ),
    ("oncvpsp: ERROR reading input file", "input_file_read_failed"),
    (
        "oncvpsp-387: lschfb convergence ERROR n,l,iter=   3   1  12",
        "lschfb_not_converged",
    ),
    (
        "sratom123: lschfb convergence ERROR n,l,iter=   3   1  12",
        "lschfb_not_converged",
    ),
    (
        "run_plot: lschfb convergence ERROR n,l,ierr=   3   1   1",
        "lschfb_not_converged",
    ),
    (
        "runconfig: WARNING lschfb convergence ERROR n,l,iter=   3   1  12",
        "runconfig_lschfb_not_converged",
    ),
    (
        "runconfig: WARNING ldiracfb convergence ERROR n,l,kap,iter=   2   1  -2   7",
        "ldiracfb_not_converged",
    ),
    (
        "relatom: ldiracfb convergence ERROR n,l,kap,iter,ierr=   3   1  -2   5   1",
        "ldiracfb_not_converged",
    ),
    (
        "oncvpsp_r 409: ldiracfb convergence ERROR, n= 3  l= 1kap--2",
        "ldiracfb_not_converged",
    ),
    (
        "run_diag_r: ldiracfb convergence ERROR          n,l,,kap,ierr=   3   1  -2   1",
        "ldiracfb_not_converged",
    ),
    (
        "oncvpsp_r: lschvkbb convergence ERROR, n= 2  l= 1",
        "vkb_bound_state_not_converged_r",
    ),
    ("oncvpsp: lschvkbb ERROR   6   1   3", "vkb_bound_state_not_converged"),
    (
        "psatom: WARNING lschvkbb convergence error n,l,iter=   2   1   7",
        "lschvkbb_not_converged",
    ),
    (
        "psatom_r: WARNING lschvkbb convergence error n,l,kap,iter=   2   1  -2   7",
        "lschvkbb_not_converged",
    ),
    (
        "run_diag: lschvkbb ERROR   1   1   2  -1.00000000E-01 -2.00000000E-01",
        "diag_lschvkbb_not_converged",
    ),
    (
        "run_ghosts: lschvkbb ERROR   2   1   1   -0.100000   -0.200000",
        "diag_lschvkbb_not_converged",
    ),
    (
        "run_diag: lschvkbbe ERROR   2   6   1  60  9.32315070E+00  1.03054658E+01",
        "lschvkbbe_not_converged",
    ),
    ("oncvpsp: ERROR debl =   0.000for l=", "debl_not_allowed"),
    (
        "optimize: residual-energy eigenvalue ERROR, info=   1",
        "optimize_eigenvalue_error",
    ),
    ("optimize: ERROR interval-halving search not converged", "optimize_not_converged"),
    ("run_vkb: B matrix eigenvalue ERROR, info=   1", "matrix_eigenvalue_error"),
    ("run_vkb: B^(-1)* matrix eigenvalue ERROR, info=   1", "matrix_eigenvalue_error"),
    ("sr_so_r: A* matrix eigenvalue ERROR, info=   1", "matrix_eigenvalue_error"),
    (
        "run_ghosts: hamiltonian matrix eigenvalue ERROR,           info=   1",
        "matrix_eigenvalue_error",
    ),
    ("run_vkb: sovl matrix ERROR error, info=   1", "sovl_matrix_error"),
    (
        "sbf_basis: ERROR overlap matrix eigenvalue error, info=   1",
        "sbf_overlap_eigenvalue_error",
    ),
    (
        "sbf_basis: overlap matrix eigenvalue error, info=   1",
        "sbf_overlap_eigenvalue_error",
    ),
    (
        "sbfbasis: ERROR negative eigenvalue of overlap matrix",
        "sbf_negative_overlap_eigenvalue",
    ),
    ("sbf_rc_der: argument ERROR, llin =    6 >5", "sbf_rc_der_bad_argument"),
    ("modcore:ERROR stop - singular polym matrix   1", "modcore_dgesv_failed"),
    ("qroots: ERROR - failed to find root", "qroots_no_root"),
    ("qroots: ERROR - failed to find nroot roots", "too_few_qroots"),
    ("dpnint: interpolation ERROR - out of range", "dpnint_interpolation_error"),
    (
        "run_ghosts 143: lschpsbar ERROR ierr,nn,ll,ee,emin,emax   1   2   1    0.1    0.2    0.3",
        "ghost_lschpsbar_error",
    ),
    ("       upfout: ERROR iexc =", "upf_unsupported_functional"),
    ("exc_libxc_stub: ERROR iexc =  -001012 requires libxc", "libxc_not_available"),
    ("vout: ERROR bad input mode =   3", "vout_bad_input"),
    ("psatom_r: WARNING potential failed to converge", "psatom_r_not_converged"),
    ("vkbout: dgesv ERROR, stopping info =   2", "vkbout_dgesv_failed"),
    ("WARNING - modcore not conveged", "icmod1_not_converged"),
    ("run_config: WARNING  for AE atom,", "ae_atom_no_solution"),
    (
        "run_config: WARNING for fully non-local PS atom, stg. 2",
        "fully_non_local_ps_atom_no_solution",
    ),
    (
        "ERROR wellstate: well potential iteration failed to converge, n=  3 l=  1 ep=  0.2500",
        "wellstate_not_converged",
    ),
    (
        "ERROR wellstate_r: well potential iteration failed to converge, n=  3 l=  1 kap= -2 ep=  0.2500",
        "wellstate_not_converged",
    ),
    (
        "wellstate: ERROR lschfb convergence error, n=   5 l=   2",
        "wellstate_no_bound_state",
    ),
    (
        " ERROR pspot:  first pseudo wave function has node,          program will stop",
        "first_pswf_has_node",
    ),
    (" WARNING pspot:  first pseudo wave function has node", "first_pswf_has_node"),
    ("ERROR: test_data found   2 ERROR; stopping", "bad_input"),
    ("Read ERROR, input data file line  12", "input_data_read_error"),
    ("lschpb: ERROR no classical turning point", "no_classical_turning_point"),
    (
        "modcore3: ERROR ircc (core-valence charge crossover) not found",
        "no_core_valence_crossing",
    ),
    ("ERROR: run_phsft: rxpsh=     1.23456 < rc(1)=     2.00000", "rxpsh_too_small"),
    ("sratom: WARNING failed to converge", "atom_not_converged"),
    (
        "WARNING wellstate: localized bound state found for n= 3 l= 2",
        "localized_bound_state_for_projector",
    ),
    # ONCVPSP 3
    ("ERROR: test_data found   2 errors; stopping", "bad_input"),
    (
        "modcore:: ERROR ircc (core-valence charge crossover) not found",
        "no_core_valence_crossing",
    ),
    ("modcore: ERROR stop - dgesv input error  -1", "modcore_dgesv_failed"),
    ("run_config: WARNING  for AE atom,", "ae_atom_no_solution"),
    (
        "run_config: ERROR for fully non-local PS atom,",
        "fully_non_local_ps_atom_no_solution",
    ),
    ("psatom: argument ERROR - okb=   3", "psatom_bad_argument"),
    (
        "psatom: WARNING lschvkbb convergence error n,l,okb,iter=   1   0   1   2",
        "lschvkbb_not_converged",
    ),
    (
        "psatom: lschvkbb convergence error n,l,okb,iter=   1   0   1   2",
        "lschvkbb_not_converged",
    ),
    ("oncvpsp: lschfb convergence ERROR, n= 3  l= 1", "lschfb_not_converged"),
    ("oncvpsp 409: lschfb convergence ERROR, n= 3  l= 1", "lschfb_not_converged"),
    (
        "WARNING: negative energy of  -0.461797 specified for second projector",
        "negative_energy_second_projector",
    ),
    ("read_error: ERROR data file line  12", "input_data_read_error"),
    ("read_srror: ERROR data file line  12", "input_data_read_error"),
    (
        "ERROR: No spin-orbit potential can be calculated for l = lloc = 1ERROR Only lloc=0",
        "fr_lloc_not_supported",
    ),
    ("dp3int: interpolation ERROR - out of range", "dpnint_interpolation_error"),
    (
        "optimize: ERROR redisual-energy eigenvalue error, info=   1",
        "optimize_eigenvalue_error",
    ),
    (
        "optimize : ERROR interval-halving search not converged",
        "optimize_not_converged",
    ),
    ("wellstate: lschfb convergence ERROR, n=  5 l=  2", "wellstate_no_bound_state"),
    (
        "wellstate: ERROR well potential iteration failed to converge, n=  3 l=  1",
        "wellstate_not_converged",
    ),
    ("vkboutwf: dgesv ERROR, stopping info =   2", "vkbout_dgesv_failed"),
    ("run_vkb: ERROR B matrix eigenvalue error, info=   1", "matrix_eigenvalue_error"),
    ("run_vkb:_r sovl matrix eigenvalue ERROR, info=   1", "matrix_eigenvalue_error"),
    (
        "relatom: ldiracfb convergence ERROR n,l,kap,iter=   3   1  -2   5",
        "ldiracfb_not_converged",
    ),
    ("qroots: ERROR failed to find root", "qroots_no_root"),
    ("qroots: ERROR failed to find nroot roots", "too_few_qroots"),
    (
        "sbf_basis: overlap matrix eigenvalue ERROR, info=   1",
        "sbf_overlap_eigenvalue_error",
    ),
    (
        "run_ghosts: lschpsbar ERROR, ierr=   1  progam will stop",
        "ghost_lschpsbar_error",
    ),
    ("lschpbar: ERROR no classical turning point", "no_classical_turning_point"),
    (
        " pspot: ERROR first pseudo wave function has node,          program will stop",
        "first_pswf_has_node",
    ),
    ("const_basis: ERROR dgesvd info =    1", "const_basis_dgesvd_failed"),
    (
        "wellstate: Negative-energy state for l=  2 found for maximum barrier.",
        "wellstate_negative_energy_state",
    ),
    # METAPSP
    ("ERROR: check_data found   1 ERROR; stopping", "bad_input"),
    ("ERROR - oncvpspmx does not provide psml output - stopping", "psml_not_supported"),
    ("ERROR polysmooth must have odd ncon", "polysmooth_bad_ncon"),
    (
        "ERROR qcut reduction by 0.74 for ll=2 failed to remove node; stopping",
        "qcut_reduction_failed",
    ),
    ("ERROR vout_m: bad input iexc =   9", "vout_bad_input"),
    (
        "oncvpsp-530: lschfb convergence ERROR n,l,iter=   3   1  12",
        "lschfb_not_converged",
    ),
    ("oncvpsp-752: lschvkbb_m ERROR   6   1   3", "vkb_bound_state_not_converged"),
    (
        "oncvpsp-768: lschfb_m convergence ERROR n,l,iter=   3   1  12",
        "lschfb_not_converged",
    ),
    (
        "run_diag_m: lschfb_m convergence ERROR n,l,ierr=   3   1   1",
        "lschfb_not_converged",
    ),
    (
        "run_diag_m: lschvkbbe_m ERROR   2   6   1  60  9.32315070E+00  1.03054658E+01",
        "lschvkbbe_not_converged",
    ),
    (
        "run_diag_m: lschvkbb_m ERROR   1   1   2  -1.00000000E-01 -2.00000000E-01",
        "diag_lschvkbb_not_converged",
    ),
    (
        "run_ghosts 150: lschpsbar_m ERROR ierr,nn,ll,ee,emin,emax",
        "ghost_lschpsbar_error",
    ),
    (
        "run_ghosts: lschvkbb_m ERROR   2   1   1   -0.1  -0.2",
        "diag_lschvkbb_not_converged",
    ),
    (
        "sratom_m: lschfb_m convergence error n,l,iter,ierr   3   1  12   1",
        "lschfb_not_converged",
    ),
    ("sratom_m: WARNING failed to converge", "atom_not_converged"),
    (
        " WARNING pspot_m: first ll=2 PSWF has node, reducing qcut to   5.00000",
        "first_pswf_has_node_qcut_reduced",
    ),
    # Messages without a version-specific variant
    (" WARNING: not fully converged in 100 steps", "icmod4_nm_not_converged"),
    (
        "WARNING wellstate: moderately localized bound state found for n= 5 l=  2",
        "moderately_localized_bound_state_for_projector",
    ),
    (
        "WARNING wellstate: negative energy specified for n= 6 l=  1",
        "scattering_first_projector",
    ),
    ("Error reading input data file", "input_file_read_error"),
    ("optimize: ERROR ps0norm > uunorm, code will stop", "ps0norm_gt_uunorm"),
]

# Lines which resemble errors or warnings but are not reported as such
NON_MESSAGES = [
    "lschfb: ERROR ll =   3 > nn =   2",
    "ldiracfb: ERROR kap =  -2 ll =   3",
    "B matrix Hermiticity error, ll=   0",
    "    Residual kinetic energy error=  1.71D+00 Ha",
    "  all-electron total energy (Ha)   -2.09589946D+03",
]


def _matching(line: str) -> list[str]:
    return [
        str(definition["name"])
        for definition in DEFINITIONS
        if str(definition["match_string"]) in line
        and re.search(str(definition["pattern"]), line)
    ]


@pytest.mark.parametrize("definition", DEFINITIONS, ids=lambda d: d["name"])
def test_definition_fields(definition):
    assert set(definition) == {
        "name",
        "pattern",
        "match_string",
        "line_count",
        "subroutine",
        "description",
    }
    re.compile(definition["pattern"])
    assert definition["line_count"] >= 1
    assert definition["description"]


def test_definition_names_unique_per_kind():
    for definitions in (WARNINGS, ERRORS):
        names = [definition["name"] for definition in definitions]
        assert len(names) == len(set(names))


def test_every_definition_has_a_message():
    covered = {name for _, name in MESSAGES}
    assert covered == {definition["name"] for definition in DEFINITIONS}


@pytest.mark.parametrize(("line", "name"), MESSAGES)
def test_message_matches_one_definition(line, name):
    assert _matching(line) == [name]


@pytest.mark.parametrize("line", NON_MESSAGES)
def test_non_message_matches_nothing(line):
    assert _matching(line) == []


def test_multiline_message_lines_are_removed():
    text = "\n".join(
        [
            "header",
            "WARNING wellstate: moderately localized bound state found for n= 5 l=  2",
            "second line",
            "third line",
            "data",
        ]
    )
    parser = OncvpspTextParser(text=text, check=False)
    (warning,) = parser.warnings
    assert warning["name"] == "moderately_localized_bound_state_for_projector"
    assert warning["line_numbers"] == [2, 3, 4]
    assert parser.clean_lines == ["header", "data"]
