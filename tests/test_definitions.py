"""Tests for the ONCVPSP/METAPSP error and warning definitions."""

import re

import pytest

from pyoncvpsp.io.output import ERRORS, WARNINGS
from pyoncvpsp.io.output._text import OncvpspTextParser

DEFINITIONS = [*WARNINGS, *ERRORS]

# Messages as written by the ONCVPSP 4.0.1, 3.3.1 and METAPSP sources, and the definition each
# should match. Similar messages from different subroutines must be told apart.
MESSAGES = [
    # azadoks/oncvpsp
    (
        "optimize: WARNING E_resid flat to round-off at qcut, e1, |f1|=  5.88E-19  2.28E-17",
        "eresid_flat",
    ),
    # azadoks/oncvpsp (branch smooooth, 1effe97)
    (
        "run_ghosts: WARNING barrier state n=4 l=1 near    12.345678 Ha not converged;",
        "ghost_barrier_state_not_converged",
    ),
    (
        "run_ghosts: WARNING basis full below the cutoff, l=2, nbmax=40",
        "ghost_basis_full",
    ),
    (
        "qroots: WARNING bisection failed, skipping, ll=1, qlow=  1.2345E+00, qhi=  2.3456E+00",
        "qroots_bisection_failed",
    ),
    (
        "modcore_budget: WARNING the radius    1.234567 of the least mismatch does not fit the budget",
        "modcore_budget_radius_misfit",
    ),
    (
        "modcore_budget: WARNING no minimum of the mismatch bracketed about    1.234567",
        "modcore_budget_no_minimum",
    ),
    (
        "modcore_budget: WARNING root at a jump of the function, x=  1.2345E+00, f from  -1.0000E-03 to   2.0000E-03",
        "modcore_budget_root_at_jump",
    ),
    (
        "ae_converged: WARNING scf stopped on the round-off floor of the residual, res=  1.2345E-12, tolres=  1.0000E-12",
        "ae_scf_residual_floor",
    ),
    (
        "run_config: WARNING self-consistency failed to converge",
        "test_config_ae_scf_not_converged",
    ),
    (
        "run_config_r: WARNING self-consistency failed to converge",
        "test_config_ae_scf_not_converged",
    ),
    (
        "run_config_r: WARNING no classical turning point error, iteration  12",
        "test_config_ae_no_turning_point",
    ),
    (
        "no classical turning point error, iteration  12",
        "test_config_ae_no_turning_point",
    ),
    (
        "run_phsft: WARNING rxpsh inside the core radius, l=1, rxpsh=    1.500000, rc=    2.000000",
        "rxpsh_inside_rc",
    ),
    (
        "run_phsft_r: WARNING rxpsh inside the local potential core radius, l=4, rxpsh=    1.500000, rc=    2.000000",
        "rxpsh_inside_local_rc",
    ),
    (
        "oncvpsp: ERROR node of the local channel l= 2 all-electron wave near rc, at r=  2.345678",
        "local_channel_ae_node",
    ),
    (
        "run_vkb: ERROR node of the local channel's pseudo wave within reach of rc, l= 2 r=  2.345678",
        "local_channel_ps_node",
    ),
    (
        "ode_rc_der: ERROR uu and up disagree for l= 1 at rc=  3.340590",
        "ode_rc_der_mismatch",
    ),
    (
        "match_point: ERROR matching window at mch=  1500 leaves the mesh, mmax=  1501",
        "match_point_failed",
    ),
    ("match_point: ERROR no sign change below mch=   800", "match_point_failed"),
    (
        "mesh_tail: ERROR wave function vanishes too close to rc: rc=  2.000000, last nonzero point  2.100000",
        "mesh_tail_too_short",
    ),
    (
        "free_tail_width: ERROR qcut=  4.000000 is not above the wave vector  5.000000",
        "free_tail_qcut_too_low",
    ),
    ("om_local_values: ERROR the pieces end inside the mesh", "output_mesh_error"),
    (
        "outmesh: ERROR more than one output point below the log mesh",
        "output_mesh_error",
    ),
    (
        "outmesh: ERROR output mesh to   12.0000 bohr, beyond the log mesh,    10.0000",
        "output_mesh_beyond_log_mesh",
    ),
    (
        "outwf_setup: ERROR the potential's pieces end inside the mesh",
        "outwf_setup_error",
    ),
    ("vploc_fit: ERROR lpopt 7", "vploc_bad_lpopt"),
    (
        "check_data: ERROR icmod=4 needs 0<fcfact_min<=fcfact_max and fcfact_step>0, fcfact_min=    0.000000, fcfact_max=    1.000000, fcfact_step=    0.100000",
        "input_teter_grid_invalid",
    ),
    (
        "check_data: ERROR icmod=4 needs 0<rcfact_min<=rcfact_max and rcfact_step>0, rcfact_min=    1.000000, rcfact_max=    0.500000, rcfact_step=    0.100000",
        "input_teter_grid_invalid",
    ),
    (
        "check_data: ERROR rxpsh not beyond every core radius, rxpsh=    1.500000, rcmax=    2.000000",
        "input_rxpsh_inside_rc",
    ),
    (
        "run_phsft: ERROR rxpsh beyond radial mesh, rxpsh=   60.000000, rr(mmax)=   50.000000",
        "rxpsh_beyond_mesh",
    ),
    ("read_ae_file: ERROR c_light differs from the input", "ae_file_mismatch"),
    ("ERROR STOP functionals: ERROR bad nspin definition", "functionals_bad_nspin"),
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
        "runconfig: WARNING ldiracfb convergence ERROR n,l,kap,ierr=   2   1  -2   1",
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
    ("psatom: potential failed to converge", "psatom_not_converged"),
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
    # azadoks/oncvpsp
    (
        "modcore_budget: ERROR stop - no model core inside r =  2.703023 needs less than"
        "    5.0000 Ha for an error of  1.00E-04 Ha",
        "modcore_budget_no_model",
    ),
    (
        "modcore_budget: ERROR stop - no model core inside rc =  3.200622 needs less than"
        "   20.0000 Ha for an error of  1.00E-04 Ha",
        "modcore_budget_no_model",
    ),
    ("modcore_budget: ERROR stop - no core charge", "modcore_budget_no_core_charge"),
    (
        "modcore_budget: ERROR stop - no core charge at rc",
        "modcore_budget_no_core_charge",
    ),
    (
        "modcore_budget_ecut: ERROR stop - residual kinetic energy above  1.00E-04 Ha at"
        "    112.50 Ha, the end of its tables",
        "modcore_budget_waves_unconverged",
    ),
    (
        "modcore_budget_ecut: ERROR stop - no residual energy curves",
        "modcore_budget_no_waves",
    ),
    (
        "modcore_budget_table: ERROR stop - no pseudo-wave reaches  1.00E-05 Ha per electron in its table",
        "modcore_budget_table_unreached",
    ),
    (
        "modcore_budget: ERROR stop - ecut beyond    200.00 Ha",
        "modcore_budget_ecut_too_large",
    ),
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
    ("oncvpsp: ERROR oncvpsp_ps.x requires --ae-file", "ae_file_required"),
    ("write_ae_file: ERROR cannot open si.ae", "ae_file_write_failed"),
    ("read_ae_file: ERROR cannot open si.ae", "ae_file_read_failed"),
    ("read_ae_file: ERROR not an all-electron data file: si.ae", "ae_file_read_failed"),
    (
        "read_ae_file: ERROR file format version   2, expected   1",
        "ae_file_read_failed",
    ),
    (
        "read_ae_file: ERROR atsym, z, nc, nv or iexc differ from the input",
        "ae_file_mismatch",
    ),
    (
        "read_ae_file: ERROR reference configuration differs from the input",
        "ae_file_mismatch",
    ),
    ("read_ae_file: ERROR radial mesh differs from the input", "ae_file_mismatch"),
    (
        "read_ae_file: ERROR test configuration 2 differs from the input",
        "ae_file_mismatch",
    ),
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
    # check_data complaints (ONCVPSP 4.0.1 check_data.f90), each followed by the `bad_input` summary
    ("test_data: must have rc>0.0, l=   1", "input_rc_nonpositive"),
    ("test_data: must have 3 <= ncon <= 5, l=   2", "input_ncon_out_of_range"),
    (
        "test_data: must have ncon+3 <= nbas <= ncon+5, l=   1",
        "input_nbas_out_of_range",
    ),
    (
        "test_data: must have ncon+2 <= nbas <= ncon+5, l=   1",
        "input_nbas_out_of_range",
    ),
    ("test_data: must have qcut>0.0, l=   0", "input_qcut_nonpositive"),
    ("test_data: rc < rc(lloc) for l = 0.  Not allowed.", "input_rc_lt_rcloc"),
    ("test_data: must have debl>0.0)   2", "input_debl_negative"),
    (
        "test_data: must have nproj in [1,5] (0 OK lloc=4 or   2)",
        "input_nproj_out_of_range",
    ),
    (
        "test_data: must have 0 <= lloc <= lmax or lloc == 4",
        "input_local_potential_invalid",
    ),
    ("test_data: for lloc==4, must have rc > 0.0", "input_local_potential_invalid"),
    ("test_data: must have 1 <= lpopt <= 5", "input_local_potential_invalid"),
    ("test_data: must have 0<= icmod <=4", "input_model_core_invalid"),
    ("test_data: must have fcfact>0.0 for icmod= 1 or 3", "input_model_core_invalid"),
    # azadoks/oncvpsp
    ("test_data: must have 0<= icmod <=4 or icmod=6", "input_model_core_invalid"),
    (
        "test_data: must have fcfact>0.0 and rcfact>0.0 for icmod= 6",
        "input_model_core_invalid",
    ),
    ("test_data: must have rlmax>rcmax", "input_rlmax_too_small"),
    ("test_data: must have positive atomic number z", "input_invalid"),
    ("test_data: l > n, reference configuration line   3", "input_invalid"),
    ("test_data: negative ion, test configuration   1", "input_invalid"),
    # Fortran runtime errors (stderr, when merged into the parsed text)
    (
        "Fortran runtime error: Expected REAL for item 10 in formatted transfer, got CHARACTER",
        "fortran_runtime_error",
    ),
    (
        "forrtl: severe (174): SIGSEGV, segmentation fault occurred",
        "fortran_runtime_error",
    ),
    (
        "Program received signal SIGFPE: Floating-point exception - erroneous arithmetic operation.",
        "fortran_runtime_error",
    ),
]

# Lines which resemble errors or warnings but are not reported as such
NON_MESSAGES = [
    "lschfb: ERROR ll =   3 > nn =   2",
    "ldiracfb: ERROR kap =  -2 ll =   3",
    "B matrix Hermiticity error, ll=   0",
    "    Residual kinetic energy error=  1.71D+00 Ha",
    "  all-electron total energy (Ha)   -2.09589946D+03",
    "At line 436 of file oncvpsp.F90 (unit = 6, file = 'stdout')",
    "Note: The following floating-point exceptions are signalling: IEEE_UNDERFLOW_FLAG",
    "Error termination. Backtrace:",
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


def test_ghost_basis_cut_is_one_two_line_warning():
    text = "\n".join(
        [
            "header",
            "",
            "run_ghosts: WARNING barrier state n=4 l=1 near    12.345678 Ha not converged;",
            "run_ghosts: WARNING basis cut there, ghost test to      2.50 Ha",
            "data",
        ]
    )
    parser = OncvpspTextParser(text=text, check=False)
    (warning,) = parser.warnings
    assert warning["line_numbers"] == [3, 4]
    assert parser.clean_lines == ["header", "", "data"]
    assert parser.ghost_basis_cuts == [
        {"n": 4, "l": 1, "e_barrier": 12.345678, "ecut": 2.5}
    ]
