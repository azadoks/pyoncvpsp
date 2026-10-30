"""ONCVPSP warning and error definitions.

A line is attributed to a definition if it contains `match_string` and `pattern` matches it (`re.search`).
`line_count` is the number of lines (starting from the matched line) that make up the message.
The wording of many messages differs between ONCVPSP v3, v4, and METAPSP; the patterns cover all known variants.
"""

from typing import Any

# the exponent with a letter, or of 3 digits without one (an E or D edit descriptor with a 2-digit exponent field
# at |exponent| > 99: "2.1869480064996404-119"; `fort_float` reads both)
RE_FLOAT: str = r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][+-]?\d+|[+-]\d{3}(?![\d.]))?"

WARNINGS: list[dict[str, Any]] = [
    {
        "name": "icmod1_not_converged",
        # ONCVPSP prints "conveged" (sic)
        "pattern": r"WARNING - modcore not conver?ged",
        "match_string": "WARNING - modcore not conve",
        "line_count": 1,
        "subroutine": "modcore",
        "description": "Optimization of polynomial model core charge did not converge",
    },
    {
        "name": "icmod4_nm_not_converged",
        "pattern": r"WARNING: not fully converged in 100 steps",
        "match_string": "WARNING: not fully converged in 100 steps",
        "line_count": 1,
        "subroutine": "modcore3",
        "description": "Nelder-Mead optimization of model core charge Teter parameters did not converge",
    },
    {
        "name": "atom_not_converged",
        "pattern": r"(?P<rel>sr|rel)atom(?:_m)?: WARNING failed to converge",
        "match_string": "atom",
        "line_count": 1,
        "subroutine": "(sr|rel)atom",
        "description": "All-electron calculation did not converge",
    },
    {
        "name": "ldiracfb_not_converged",
        # ONCVPSP v4 prints "n,l,kap,ierr=" for the j = l + 1/2 state and "n,l,kap,iter=" for j = l - 1/2
        "pattern": r"runconfig: WARNING ldiracfb convergence ERROR n,l,kap,(?:iter|ierr)=\s*(?P<n>\d+)\s+(?P<l>\d+)",
        "match_string": "runconfig: WARNING ldiracfb convergence ERROR",
        "line_count": 1,
        "subroutine": "run_config_r",
        "description": "All-electron bound state did not converge for a test configuration (fully-relativistic).",
    },
    {
        "name": "runconfig_lschfb_not_converged",
        "pattern": r"runconfig: WARNING lschfb convergence ERROR n,l,iter=\s*(?P<n>\d+)\s+(?P<l>\d+)",
        "match_string": "runconfig: WARNING lschfb convergence ERROR",
        "line_count": 1,
        "subroutine": "run_config",
        "description": "All-electron bound state did not converge for a test configuration.",
    },
    {
        "name": "ae_atom_no_solution",
        # Followed by " WARNING no output for configuration" (v4) or " no output for this configuration" (v3),
        # then by the reason
        "pattern": r"run_config(?:_r)?: WARNING\s+for AE atom,",
        "match_string": "for AE atom,",
        "line_count": 3,
        "subroutine": "run_config(_r)",
        "description": "All-electron calculation failed for a test configuration; no output for that configuration.",
    },
    {
        "name": "fully_non_local_ps_atom_no_solution",
        # ONCVPSP v3 prints "run_config: ERROR for fully non-local  PS atom,"
        # Followed by " WARNING no output for configuration k" (v4), where k counts the reference
        # configuration as 1 (the block is headed "Test configuration k-1"), or by
        # " no output for this configuration" (v3)
        "pattern": r"run_config(?:_r)?: (?:WARNING|ERROR) for fully non-local\s+PS atom,",
        "match_string": "for fully non-local",
        "line_count": 2,
        "subroutine": "run_config(_r)",
        "description": "Pseudo-atom calculation failed for a test configuration; no output for that configuration.",
    },
    {
        "name": "psatom_not_converged",
        "pattern": r"psatom: potential failed to converge",
        "match_string": "psatom: potential failed to converge",
        "line_count": 1,
        # Also printed by `psatom_r` of ONCVPSP v3
        "subroutine": "psatom(_r)",
        "description": "Pseudo-atom self-consistent potential did not converge for a test configuration.",
    },
    {
        "name": "psatom_r_not_converged",
        "pattern": r"psatom_r: WARNING potential failed to converge",
        "match_string": "psatom_r: WARNING potential failed to converge",
        "line_count": 1,
        "subroutine": "psatom_r",
        "description": "Pseudo-atom self-consistent potential did not converge for a test configuration.",
    },
    {
        "name": "localized_bound_state_for_projector",
        "pattern": r"WARNING wellstate: localized bound state found for n=\s*(?P<n>\d+)\s+l=\s*(?P<l>\d+)",
        "match_string": "WARNING wellstate: localized bound state found for n=",
        "line_count": 3,
        "subroutine": "wellstate[_r]",
        "description": "Unoccupied bound state is sufficiently localized to be used for the projector instead of a wellstate.",
    },
    {
        "name": "moderately_localized_bound_state_for_projector",
        "pattern": r"WARNING wellstate: moderately localized bound state found for n=\s*(?P<n>\d+)\s+l=\s*(?P<l>\d+)",
        "match_string": "WARNING wellstate: moderately localized bound state found for n=",
        "line_count": 3,
        "subroutine": "wellstate[_r]",
        "description": "Unoccupied bound state is moderately localized; a low-energy scattering state is used for the projector.",
    },
    {
        "name": "scattering_first_projector",
        "pattern": r"WARNING wellstate: negative energy specified for n=\s*(?P<n>\d+) l=\s*(?P<l>\d+)",
        "match_string": "WARNING wellstate: negative energy specified for n=",
        "line_count": 3,
        "subroutine": "wellstate",
        "description": "Negative energy specified for a wellstate; a scattering state is used for the projector.",
    },
    {
        "name": "wellstate_negative_energy_state",
        "pattern": r"wellstate: Negative-energy state for l=\s*(?P<l>\d+) found for maximum barrier",
        "match_string": "wellstate: Negative-energy state for l=",
        "line_count": 2,
        "subroutine": "wellstate[_r]",
        "description": "Bound state found for the maximum well barrier; the normal bound state is used for the projector (ONCVPSP v3).",
    },
    {
        "name": "negative_energy_second_projector",
        "pattern": rf"WARNING: negative energy of\s*(?P<ep>{RE_FLOAT}) specified for second projector",
        "match_string": "WARNING: negative energy of",
        "line_count": 3,
        "subroutine": "oncvpsp",
        "description": "Negative energy specified for the second projector; the actual bound-state energy is used instead (ONCVPSP v3).",
    },
    {
        "name": "lschvkbbe_not_converged",
        # run_diag: lschvkbbe ERROR iprj, nnp, ll, ierr, epa, etest
        "pattern": (
            r"(?P<subroutine>\w+): lschvkbbe(?:_m)? ERROR\s+(?P<iproj>\d+)\s+(?P<n>\d+)\s+(?P<l>\d+)\s+(?P<ierr>-?\d+)"
            rf"\s+({RE_FLOAT})\s+({RE_FLOAT})"
        ),
        "match_string": "lschvkbbe",
        "line_count": 1,
        "subroutine": "run_diag(_r)",
        "description": "Unable to converge fixed-energy solution in local+non-local potential.",
    },
    {
        "name": "diag_lschvkbb_not_converged",
        # run_diag: lschvkbb ERROR iprj, ll, ierr, ...; run_ghosts: lschvkbb ERROR nn, ll, ierr, ...
        "pattern": (
            r"(?P<subroutine>run_diag(?:_r|_m|_sr_so_r)?|run_ghosts): lschvkbb(?:_m)? ERROR"
            r"(?:\s+(?P<first>\d+)\s+(?P<l>\d+)\s+(?P<ierr>-?\d+))?"
        ),
        "match_string": "lschvkbb",
        "line_count": 1,
        "subroutine": "run_diag(_r)|run_ghosts",
        "description": "Unable to converge bound-state solution in local+non-local potential during diagnostics.",
    },
    {
        "name": "first_pswf_has_node",
        # Not printed by released ONCVPSP versions, which stop instead (see the error of the same name)
        "pattern": r"WARNING pspot:\s+first pseudo wave function has node",
        "match_string": "first pseudo wave function has node",
        "line_count": 1,
        "subroutine": "pspot",
        "description": "First pseudo wavefunction has a node.",
    },
    {
        "name": "first_pswf_has_node_qcut_reduced",
        "pattern": rf"WARNING pspot_m: first ll=\s*(?P<l>\d) PSWF has node, reducing qcut to\s*(?P<qcut>{RE_FLOAT})",
        "match_string": "WARNING pspot_m: first ll=",
        "line_count": 1,
        "subroutine": "oncvpsp_mx",
        "description": "First pseudo wavefunction has a node; qcut is reduced (METAPSP).",
    },
    {
        "name": "eresid_flat",
        # azadoks/oncvpsp (branch smooooth): the smallest eigenvalue e1 of E_resid(qcut) in the
        # null space and |f1| both below 1e-12 Ha, so the optimized wave is set by round-off
        "pattern": (
            rf"optimize: WARNING E_resid flat to round-off at qcut, e1, \|f1\|=\s*"
            rf"(?P<e1>{RE_FLOAT})\s+(?P<f1>{RE_FLOAT})"
        ),
        "match_string": "E_resid flat to round-off",
        "line_count": 2,
        "subroutine": "optimize",
        "description": "Residual energy at qcut flat to round-off; the pseudo wave function is not "
        "determined (wave_regularization).",
    },
]

ERRORS: list[dict[str, Any]] = [
    {
        "name": "lschvkbb_not_converged",
        # ONCVPSP v3 prints "psatom: lschvkbb convergence error n,l,okb,iter="
        "pattern": r"psatom(?:_r)?: (?:WARNING )?lschvkbb convergence error n,l,(?:kap,|okb,)?iter=\s*(?P<n>\d+)\s+(?P<l>\d+)",
        "match_string": "lschvkbb convergence error n,l,",
        "line_count": 1,
        "subroutine": "psatom(_r)",
        "description": "Unable to converge bound-state solution in local+non-local potential.",
    },
    {
        "name": "wellstate_not_converged",
        "pattern": r"well potential iteration failed to converge, n=\s*(?P<n>\d+)\s+l=\s*(?P<l>\d+)",
        "match_string": "well potential iteration failed to converge, n=",
        "line_count": 1,
        "subroutine": "wellstate(_r)",
        "description": "Unable to converge well-state solution in full potential.",
    },
    {
        "name": "bad_input",
        "pattern": r"ERROR: (?:test|check)_data found\s*(?P<nerr>\d+)\s*(?:ERROR|errors); stopping",
        "match_string": "_data found",
        "line_count": 1,
        "subroutine": "check_data",
        "description": "Invalid input data.",
    },
    {
        "name": "input_file_read_error",
        "pattern": r"Error reading input data file",
        "match_string": "Error reading input data file",
        "line_count": 1,
        "subroutine": "input",
        "description": "Unable to read input data file.",
    },
    {
        "name": "input_data_read_error",
        # "Read ERROR, input data file line" (v4), "read_error: ERROR data file line" (v3)
        "pattern": r"(?:Read (?:ERROR|error), input|read_[es]rror: ERROR) data file line\s*(?P<line>\d+)",
        "match_string": "data file line",
        "line_count": 1,
        "subroutine": "read_error",
        "description": "Unable to read a line of the input data file.",
    },
    {
        "name": "input_file_read_failed",
        "pattern": r"oncvpsp: ERROR reading input file",
        "match_string": "oncvpsp: ERROR reading input file",
        "line_count": 1,
        "subroutine": "oncvpsp",
        "description": "Unable to read input file.",
    },
    {
        "name": "ae_file_required",
        "pattern": r"oncvpsp\S*: ERROR \S+ requires --ae-file",
        "match_string": "requires --ae-file",
        "line_count": 1,
        "subroutine": "oncvpsp_main",
        "description": "A stage of the split run (`oncvpsp*_ae.x`, `oncvpsp*_ps.x`) was not given its all-electron data file.",
    },
    {
        "name": "ae_file_write_failed",
        "pattern": r"write_ae_file: ERROR (?P<reason>.*)",
        "match_string": "write_ae_file: ERROR",
        "line_count": 1,
        "subroutine": "write_ae_file",
        "description": "The all-electron stage could not write its data file.",
    },
    {
        "name": "ae_file_mismatch",
        "pattern": r"read_ae_file: ERROR (?P<reason>(?:atsym|reference configuration|radial mesh|test configuration|c_light).*)",
        "match_string": "read_ae_file: ERROR",
        "line_count": 1,
        "subroutine": "read_ae_file",
        "description": "The all-electron data file was written for a different atom, reference or test configuration, radial mesh or c_light than the input.",
    },
    {
        "name": "ae_file_read_failed",
        "pattern": r"read_ae_file: ERROR (?P<reason>(?:cannot open|not an all-electron|file).*)",
        "match_string": "read_ae_file: ERROR",
        "line_count": 1,
        "subroutine": "read_ae_file",
        "description": "The all-electron data file could not be opened or read (missing, another format version, relativistic treatment or byte order).",
    },
    {
        "name": "no_classical_turning_point",
        "pattern": r"ERROR no classical turning point",
        "match_string": "ERROR no classical turning point",
        "line_count": 1,
        "subroutine": "lsch*",
        "description": "No classical turning point found.",
    },
    {
        "name": "no_core_valence_crossing",
        "pattern": r"ERROR ircc \(core-valence charge crossover",
        "match_string": "ERROR ircc (core-valence charge crossover",
        "line_count": 1,
        "subroutine": "modcore*",
        "description": "No core-valence charge density crossover radius found.",
    },
    {
        "name": "first_pswf_has_node",
        # " ERROR pspot:  first pseudo wave function has node," (v4), " pspot: ERROR first ..." (v3)
        "pattern": r"(?:ERROR pspot:|pspot: ERROR)\s+first pseudo wave function has node",
        "match_string": "first pseudo wave function has node",
        "line_count": 2,
        "subroutine": "pspot",
        "description": "First pseudo wavefunction has a node.",
    },
    {
        "name": "qcut_reduction_failed",
        "pattern": r"ERROR qcut reduction by [\d.]+ for ll=\s*(?P<l>\d) failed to remove node",
        "match_string": "ERROR qcut reduction by",
        "line_count": 1,
        "subroutine": "oncvpsp_m",
        "description": "Reducing qcut failed to remove a node from the first pseudo wavefunction (METAPSP).",
    },
    {
        "name": "ps0norm_gt_uunorm",
        "pattern": r"optimize: ERROR ps0norm > uunorm, code will stop",
        "match_string": "optimize: ERROR ps0norm > uunorm, code will stop",
        "line_count": 1,
        "subroutine": "optimize",
        "description": "Norm of unoptimized pseudo wavefunction exceeds that of all-electron wavefunction.",
    },
    {
        "name": "optimize_eigenvalue_error",
        # "optimize: residual-energy eigenvalue ERROR" (v4), "optimize: ERROR redisual-energy eigenvalue error" (v3)
        "pattern": r"optimize: (?:ERROR )?re(?:sid|dis)ual-energy eigenvalue (?:ERROR|error), info=\s*(?P<info>-?\d+)",
        "match_string": "ual-energy eigenvalue",
        "line_count": 1,
        "subroutine": "optimize",
        "description": "Diagonalization of the residual kinetic energy matrix failed.",
    },
    {
        "name": "optimize_not_converged",
        "pattern": r"optimize ?: ERROR interval-halving search not converged",
        "match_string": "ERROR interval-halving search not converged",
        "line_count": 1,
        "subroutine": "optimize",
        "description": "Interval-halving search for the optimized pseudo wavefunction did not converge.",
    },
    {
        "name": "optimize_newton_not_converged",
        # azadoks/oncvpsp smooooth (solve_norm_constraint); the last step can be NaN
        "pattern": r"optimize: ERROR Newton polish not converged, last step\s*(?P<step>\S+)",
        "match_string": "optimize: ERROR Newton polish not converged",
        "line_count": 1,
        "subroutine": "optimize (solve_norm_constraint)",
        "description": "The Newton polish of the norm constraint of the optimized pseudo "
        "wavefunction did not converge.",
    },
    {
        "name": "rxpsh_too_small",
        "pattern": rf"ERROR: run_phsft: rxpsh=\s*(?P<rxpsh>{RE_FLOAT}) < rc\((?P<l>\d)\)=\s*(?P<rc>{RE_FLOAT})",
        "match_string": "ERROR: run_phsft: rxpsh=",
        "line_count": 1,
        "subroutine": "run_phsft",
        "description": "Matching radius is smaller than pseudopotential core radius for given angular momentum.",
    },
    {
        "name": "wellstate_no_bound_state",
        # "wellstate: ERROR lschfb convergence error, n=" (v4), "wellstate: lschfb convergence ERROR, n=" (v3)
        "pattern": r"wellstate(?:_r)?: (?:ERROR )?lschfb convergence (?:error|ERROR), n=\s*(?P<n>\d+)\s+l=\s*(?P<l>\d+)",
        "match_string": "lschfb convergence",
        "line_count": 1,
        "subroutine": "wellstate",
        "description": "No bound state found in the well potential for given n and l.",
    },
    {
        "name": "lschfb_not_converged",
        "pattern": r"^\s*(?!wellstate|runconfig)(?P<subroutine>\S+(?: \d+)?): lschfb(?:_m)? convergence (?:ERROR|error)",
        "match_string": "lschfb",
        "line_count": 1,
        "subroutine": "sratom|oncvpsp|run_plot|run_diag",
        "description": "Unable to converge all-electron bound state (scalar-relativistic).",
    },
    {
        "name": "ldiracfb_not_converged",
        "pattern": r"^\s*(?!runconfig)(?P<subroutine>\S+(?: \d+)?): ldiracfb convergence ERROR",
        "match_string": ": ldiracfb convergence ERROR",
        "line_count": 1,
        "subroutine": "relatom|oncvpsp_r|run_plot_r|run_diag_r",
        "description": "Unable to converge all-electron bound state (fully-relativistic).",
    },
    {
        "name": "ae_reference_not_converged",
        "pattern": r"oncvpsp(?:_r)?: ERROR all-electron reference atom not converged",
        "match_string": "ERROR all-electron reference atom not converged",
        "line_count": 1,
        "subroutine": "oncvpsp(_r)",
        "description": "All-electron reference configuration did not converge.",
    },
    {
        "name": "too_few_qroots",
        "pattern": r"qroots: ERROR (?:- )?failed to find nroot roots",
        "match_string": "failed to find nroot roots",
        "line_count": 1,
        "subroutine": "qroots",
        "description": "Unable to find required number of roots in qroots subroutine.",
    },
    {
        "name": "qroots_no_root",
        "pattern": r"qroots: ERROR (?:- )?failed to find root",
        "match_string": "failed to find root",
        "line_count": 1,
        "subroutine": "qroots",
        "description": "Unable to find a root in qroots subroutine.",
    },
    {
        "name": "qroots_newton_not_converged",
        # azadoks/oncvpsp smooooth (polish, after the bisection of a root)
        "pattern": (
            rf"qroots: ERROR Newton polish not converged, ll=\s*(?P<l>\d+), q=\s*(?P<q>{RE_FLOAT}),"
            r" last step\s*(?P<step>\S+)"
        ),
        "match_string": "qroots: ERROR Newton polish not converged",
        "line_count": 1,
        "subroutine": "qroots (polish)",
        "description": "The Newton polish of a root q of the basis functions' matching condition "
        "did not converge.",
    },
    {
        "name": "vkb_bound_state_not_converged",
        # "oncvpsp: lschvkbb ERROR" (v3, v4), "oncvpsp-752: lschvkbb_m ERROR" (METAPSP)
        # ierr 200 is vkb_bound_ghost
        "pattern": r"oncvpsp(?:-\d+)?: lschvkbb(?:_m)? ERROR\s+(?P<n>\d+)\s+(?P<l>\d+)\s+(?P<ierr>(?!200\b)-?\d+)",
        "match_string": "lschvkbb",
        "line_count": 1,
        "subroutine": "oncvpsp",
        "description": "Unable to solve for bound state in separable pseudopotential when constructing final pseudo-states and densities.",
    },
    {
        "name": "vkb_bound_ghost",
        # azadoks/oncvpsp from 61e586a: lschvkbb's ierr 200, followed by
        # "a state of l= 0 lies below   -0.132031 Ha that the node count does not expect:" and
        # "a bound ghost of the projectors"
        "pattern": r"oncvpsp: lschvkbb ERROR\s+(?P<n>\d+)\s+(?P<l>\d+)\s+(?P<ierr>200)\b",
        "match_string": "lschvkbb",
        "line_count": 3,
        "subroutine": "oncvpsp",
        "description": "A valence state could not be solved in the separable pseudopotential because "
        "the projectors bind a state of the same l below it that the node count does not "
        "expect: a bound ghost.",
    },
    {
        "name": "vkb_bound_state_not_converged_r",
        "pattern": r"oncvpsp_r: lschvkbb convergence ERROR, n=\s*(?P<n>\d+)\s+l=\s*(?P<l>\d+)",
        "match_string": "oncvpsp_r: lschvkbb convergence ERROR",
        "line_count": 1,
        "subroutine": "oncvpsp_r",
        "description": "Unable to solve for bound state in separable pseudopotential when constructing final pseudo-states and densities.",
    },
    {
        "name": "vkbout_dgesv_failed",
        "pattern": r"vkbout(?:wf)?: dgesv ERROR, stopping info =\s*(?P<info>-?\d+)",
        "match_string": "dgesv ERROR, stopping info",
        "line_count": 1,
        "subroutine": "vkboutwf",
        "description": "Linear solve for VKB pseudo wavefunction coefficients failed while writing plotting data.",
    },
    {
        "name": "debl_not_allowed",
        "pattern": r"oncvpsp: ERROR debl =",
        "match_string": "oncvpsp: ERROR debl =",
        "line_count": 3,
        "subroutine": "oncvpsp(_r)",
        "description": "debl <= 0 is not allowed with 2 or more scattering states for a given l.",
    },
    {
        "name": "fr_lloc_not_supported",
        "pattern": r"ERROR: No spin-orbit potential can be calculated for l = lloc =\s*(?P<lloc>\d)",
        "match_string": "ERROR: No spin-orbit potential can be calculated",
        "line_count": 1,
        "subroutine": "oncvpsp_r",
        "description": "Only lloc = 0 or lloc = 4 are supported for fully-relativistic pseudopotentials (ONCVPSP v3).",
    },
    {
        "name": "matrix_eigenvalue_error",
        # e.g. "run_vkb: B matrix eigenvalue ERROR" (v4), "run_vkb: ERROR B matrix eigenvalue error" (v3)
        "pattern": r"^(?!.*overlap matrix).*matrix eigenvalue (?:ERROR|error)",
        "match_string": "matrix eigenvalue",
        "line_count": 1,
        "subroutine": "run_vkb(_r)|sr_so_r|run_ghosts",
        "description": "Matrix diagonalization (LAPACK dsyev) failed.",
    },
    {
        "name": "sovl_matrix_error",
        "pattern": r"run_vkb: sovl matrix ERROR error, info=\s*(?P<info>-?\d+)",
        "match_string": "run_vkb: sovl matrix ERROR error",
        "line_count": 1,
        "subroutine": "run_vkb",
        "description": "Diagonalization of the projector overlap matrix failed.",
    },
    {
        "name": "sbf_overlap_eigenvalue_error",
        "pattern": r"sbf_basis: (?:ERROR )?overlap matrix eigenvalue (?:ERROR|error), info=\s*(?P<info>-?\d+)",
        "match_string": "overlap matrix eigenvalue",
        "line_count": 1,
        "subroutine": "sbf_basis(_con)",
        "description": "Diagonalization of the spherical Bessel function basis overlap matrix failed.",
    },
    {
        "name": "sbf_negative_overlap_eigenvalue",
        "pattern": r"sbfbasis: ERROR negative eigenvalue of overlap matrix",
        "match_string": "sbfbasis: ERROR negative eigenvalue of overlap matrix",
        "line_count": 1,
        "subroutine": "sbf_basis",
        "description": "Spherical Bessel function basis overlap matrix has a negative eigenvalue.",
    },
    {
        "name": "sbf_rc_der_bad_argument",
        "pattern": r"sbf_rc_der: argument ERROR, llin =\s*(?P<l>\d+)",
        "match_string": "sbf_rc_der: argument ERROR",
        "line_count": 1,
        "subroutine": "sbf_rc_der",
        "description": "Angular momentum greater than 5 is not supported.",
    },
    {
        "name": "const_basis_dgesvd_failed",
        "pattern": r"const_basis: ERROR dgesvd info =\s*(?P<info>-?\d+)",
        "match_string": "const_basis: ERROR dgesvd",
        "line_count": 1,
        "subroutine": "const_basis",
        "description": "Singular value decomposition of the constraint matrix failed (ONCVPSP v3).",
    },
    {
        "name": "modcore_dgesv_failed",
        "pattern": r"modcore: ?ERROR stop - (?:singular polym matrix|dgesv input error)",
        "match_string": "ERROR stop - ",
        "line_count": 1,
        "subroutine": "modcore",
        "description": "Linear solve for polynomial model core charge coefficients failed.",
    },
    {
        "name": "modcore_budget_no_model",
        "pattern": (
            rf"modcore_budget: ERROR stop - no model core inside r(?:c)? =\s*(?P<r>{RE_FLOAT})"
            rf" needs less than\s*(?P<ecut>{RE_FLOAT}) Ha for an error of\s*(?P<error>{RE_FLOAT}) Ha"
        ),
        "match_string": "modcore_budget: ERROR stop - no model core",
        "line_count": 1,
        "subroutine": "modcore_budget",
        "description": "Even the largest budgeted model core (icmod 6) needs more than the cutoff budget.",
    },
    {
        "name": "modcore_budget_no_core_charge",
        # azadoks/oncvpsp 8fef166 prints "no core charge at rc"
        "pattern": r"modcore_budget: ERROR stop - no core charge",
        "match_string": "modcore_budget: ERROR stop - no core charge",
        "line_count": 1,
        "subroutine": "modcore_budget",
        "description": "Budgeted model core (icmod 6) of an atom without core charge.",
    },
    {
        "name": "modcore_budget_waves_unconverged",
        "pattern": (
            rf"modcore_budget_ecut: ERROR stop - residual kinetic energy above\s*(?P<error>{RE_FLOAT})"
            rf" Ha at\s*(?P<ecut>{RE_FLOAT}) Ha, the end of its tables"
        ),
        "match_string": "modcore_budget_ecut: ERROR stop - residual kinetic energy above",
        "line_count": 1,
        "subroutine": "modcore_budget_ecut",
        "description": "The pseudo-waves do not converge to the error of the budgeted model core (icmod 6, fcfact <= 0) within their residual energy tables.",
    },
    {
        "name": "modcore_budget_table_unreached",
        "pattern": rf"modcore_budget_table: ERROR stop - no pseudo-wave reaches\s*(?P<error>{RE_FLOAT}) Ha per electron",
        "match_string": "modcore_budget_table: ERROR stop",
        "line_count": 1,
        "subroutine": "modcore_budget_table",
        "description": "No convergence table of the pseudo-waves reaches the error of the budgeted model core (icmod 6, 7 with fcfact <= 0).",
    },
    {
        "name": "modcore_budget_table_wave_no_curve",
        "pattern": r"modcore_budget_table: ERROR stop - pseudo-wave (?P<iprof>\d+) has a weight and no convergence curve",
        "match_string": "modcore_budget_table: ERROR stop - pseudo-wave ",
        "line_count": 1,
        "subroutine": "modcore_budget_table",
        "description": "A pseudo-wave with a weight in the weighted budget (budget_cutoff weighted) has no convergence curve.",
    },
    {
        "name": "modcore_budget_table_no_weighted",
        "pattern": r"modcore_budget_table: ERROR stop - no weighted curves",
        "match_string": "modcore_budget_table: ERROR stop - no weighted curves",
        "line_count": 1,
        "subroutine": "modcore_budget_table",
        "description": "The convergence curves with a weight in the weighted budget (budget_cutoff weighted) have no wave vectors in common.",
    },
    {
        "name": "modcore_budget_table_mean_unreached",
        # azadoks/oncvpsp smooooth, after the last channel's "Orthonormal projector coefficients"
        "pattern": rf"modcore_budget_table: ERROR stop - the weighted mean does not reach\s*(?P<error>{RE_FLOAT}) Ha per electron",
        "match_string": "modcore_budget_table: ERROR stop - the weighted mean does not",
        "line_count": 1,
        "subroutine": "modcore_budget_table",
        "description": "The weighted mean of the pseudo-waves' convergence curves (budget_cutoff weighted) does not fall to the error of the budgeted model core within the curves.",
    },
    {
        "name": "modcore_budget_table_mean_below",
        "pattern": (
            rf"modcore_budget_table: ERROR stop - the weighted mean is below\s*(?P<error>{RE_FLOAT})"
            r" Ha per electron where the curves start"
        ),
        "match_string": "modcore_budget_table: ERROR stop - the weighted mean is below",
        "line_count": 1,
        "subroutine": "modcore_budget_table",
        "description": "The weighted mean of the pseudo-waves' convergence curves (budget_cutoff weighted) is below the error of the budgeted model core already where the curves start.",
    },
    {
        "name": "modcore_budget_table_bracket_lost",
        "pattern": r"modcore_budget_table: ERROR the bracket lost its sign change",
        "match_string": "modcore_budget_table: ERROR the bracket lost its sign change",
        "line_count": 1,
        "subroutine": "modcore_budget_table",
        "description": "Root search for the cutoff of the weighted mean (budget_cutoff weighted) lost its bracket.",
    },
    {
        "name": "modcore_budget_transform_nodes",
        "pattern": rf"modcore_budget: ERROR stop - r_s up to\s*(?P<r>{RE_FLOAT}) bohr needs more nodes of the transform",
        "match_string": "modcore_budget: ERROR stop - r_s up to",
        "line_count": 1,
        "subroutine": "modcore_budget",
        "description": "The radii of the budgeted model core (icmod 6, 7) need more quadrature nodes for its Fourier transform than the program has.",
    },
    {
        "name": "modcore_budget_rs_beyond_table",
        "pattern": r"modcore_budget: ERROR stop - r_s beyond the table",
        "match_string": "modcore_budget: ERROR stop - r_s beyond the table",
        "line_count": 1,
        "subroutine": "modcore_budget",
        "description": "A radius of the budgeted model core (icmod 6, 7) beyond the mesh of its transform.",
    },
    {
        "name": "modcore_budget_dgglse_failed",
        "pattern": r"modcore_budget: ERROR stop - dgglse info =\s*(?P<info>-?\d+)",
        "match_string": "modcore_budget: ERROR stop - dgglse info",
        "line_count": 1,
        "subroutine": "modcore_budget",
        "description": "Constrained least-squares fit (LAPACK dgglse) of the budgeted model core's polynomial failed.",
    },
    {
        "name": "modcore_budget_no_root",
        "pattern": rf"modcore_budget: ERROR stop - no root between\s*(?P<xa>{RE_FLOAT})\s+(?P<xb>{RE_FLOAT})",
        "match_string": "modcore_budget: ERROR stop - no root between",
        "line_count": 1,
        "subroutine": "modcore_budget (zeroin)",
        "description": "Root search of the budgeted model core (icmod 6, 7) given an interval without a sign change.",
    },
    {
        "name": "modcore_budget_rs_out_of_range",
        # azadoks/oncvpsp smooooth api 9 (msg_budget_rs_range): icmod 8 and 9, r_lo = r_top/50
        "pattern": (
            rf"modcore_budget: ERROR r_s=\s*(?P<r_s>{RE_FLOAT}) bohr outside \[r_lo, r_top\] ="
            rf"\s*\[\s*(?P<r_lo>{RE_FLOAT}),\s*(?P<r_top>{RE_FLOAT})\]"
        ),
        "match_string": "modcore_budget: ERROR r_s=",
        "line_count": 1,
        "subroutine": "modcore_budget",
        "description": "The given radius of the budgeted model core (icmod 8, 9) is outside "
        "[r_top/50, r_top], r_top the smaller of the smallest rc and the last radius with core charge.",
    },
    {
        "name": "modcore_budget_fill_not_monotone",
        # azadoks/oncvpsp smooooth api 9 (msg_budget_fill_monotone)
        "pattern": rf"modcore_budget: ERROR the fill is not monotone at r_s=\s*(?P<r_s>{RE_FLOAT}) bohr",
        "match_string": "modcore_budget: ERROR the fill is not monotone",
        "line_count": 1,
        "subroutine": "modcore_budget",
        "description": "The fill of the budgeted model core (icmod 8, 9) rises inside the given radius r_s.",
    },
    {
        "name": "modcore_budget_share_out_of_range",
        # azadoks/oncvpsp smooooth api 9 (msg_budget_share_range)
        "pattern": (
            rf"modcore_budget: ERROR the error share s=\s*(?P<share>{RE_FLOAT}) has its cutoff beyond"
            rf"\s*(?P<ecut_max>{RE_FLOAT}) Ha"
        ),
        "match_string": "modcore_budget: ERROR the error share",
        "line_count": 1,
        "subroutine": "modcore_budget",
        "description": "An error share of the budgeted model core puts its cutoff beyond the largest "
        "density cutoff it checks.",
    },
    {
        "name": "modcore_budget_no_waves",
        "pattern": r"modcore_budget_ecut: ERROR stop - no residual energy curves",
        "match_string": "modcore_budget_ecut: ERROR stop - no residual energy curves",
        "line_count": 1,
        "subroutine": "modcore_budget_ecut",
        "description": "No pseudo-wave residual energy curves for the budget of the budgeted model core (icmod 6, fcfact <= 0).",
    },
    {
        "name": "modcore_budget_ecut_too_large",
        "pattern": rf"modcore_budget: ERROR stop - ecut beyond\s*(?P<ecut_max>{RE_FLOAT}) Ha",
        "match_string": "modcore_budget: ERROR stop - ecut beyond",
        "line_count": 1,
        "subroutine": "modcore_budget",
        "description": "Cutoff budget of the budgeted model core (icmod 6) beyond the largest density cutoff it checks.",
    },
    {
        "name": "dpnint_interpolation_error",
        "pattern": r"dp[n3]int: interpolation ERROR",
        "match_string": ": interpolation ERROR",
        "line_count": 1,
        "subroutine": "dpnint",
        "description": "Polynomial interpolation failed (too few points or out of range).",
    },
    {
        "name": "ghost_lschpsbar_error",
        # v4: "run_ghosts 143: lschpsbar ERROR ierr,nn,ll,ee,emin,emax" followed by the values
        "pattern": r"run_ghosts(?: \d+)?: lschpsbar(?:_m)? ERROR(?: ierr,nn,ll,ee,emin,emax\s+(?P<ierr>-?\d+)\s+(?P<n>\d+)\s+(?P<l>\d+))?",
        "match_string": "lschpsbar",
        "line_count": 1,
        "subroutine": "run_ghosts",
        "description": "Unable to solve for barrier-confined state during ghost analysis.",
    },
    {
        "name": "psatom_bad_argument",
        "pattern": r"psatom: argument ERROR - okb=\s*(?P<okb>-?\d+)",
        "match_string": "psatom: argument ERROR",
        "line_count": 1,
        "subroutine": "psatom(_r)",
        "description": "Invalid argument to psatom (ONCVPSP v3).",
    },
    {
        "name": "upf_unsupported_functional",
        "pattern": r"upfout: ERROR iexc =",
        "match_string": "upfout: ERROR iexc =",
        "line_count": 1,
        "subroutine": "upfout(_r)",
        "description": "Exchange-correlation functional is not supported for UPF output.",
    },
    {
        "name": "psml_not_supported",
        "pattern": r"ERROR - oncvpspmx does not provide psml output",
        "match_string": "ERROR - oncvpspmx does not provide psml output",
        "line_count": 1,
        "subroutine": "oncvpsp_mx",
        "description": "PSML output is not supported by this METAPSP executable.",
    },
    {
        "name": "libxc_not_available",
        "pattern": r"exc_libxc_stub: ERROR iexc =\s*(?P<iexc>-?\d+) requires libxc",
        "match_string": "requires libxc",
        "line_count": 3,
        "subroutine": "exc_libxc_stub",
        "description": "Exchange-correlation functional requires libxc, but ONCVPSP was built without it.",
    },
    {
        "name": "vout_bad_input",
        # "vout: ERROR bad input iexc =" (ONCVPSP), "ERROR vout_m: bad input iexc =" (METAPSP)
        "pattern": r"(?:vout: ERROR|ERROR vout_m:) bad input (?:mode|iexc) =",
        "match_string": "bad input",
        "line_count": 1,
        "subroutine": "vout",
        "description": "Invalid mode or exchange-correlation functional in potential calculation.",
    },
    {
        "name": "polysmooth_bad_ncon",
        "pattern": r"ERROR polysmooth must have odd ncon",
        "match_string": "ERROR polysmooth must have odd ncon",
        "line_count": 1,
        "subroutine": "gauss_smooth",
        "description": "Polynomial smoothing requires an odd number of constraints (METAPSP).",
    },
]

# Specific `check_data` complaints (each is followed by the summary `bad_input` error)
_CHECK_DATA_SPECIFIC = (
    r"must have rc>0\.0|must have 3 <= ncon|must have ncon\+|must have qcut>0\.0|rc < rc\(lloc\)"
    r"|must have debl>0\.0\)|must have nproj in|must have 0 <= lloc|for lloc==4|must have 1 <= lpopt"
    r"|must have 0<= icmod|must have fcfact|must have rcfact|must have rlmax>rcmax"
    r"|must have 0\.0<slack|must have 0\.0<r_s|must have error_shares"
)

ERRORS += [
    {
        "name": "input_rc_nonpositive",
        "pattern": r"test_data: must have rc>0\.0, l=\s*(?P<l>\d)",
        "match_string": "test_data: must have rc>0.0",
        "line_count": 1,
        "subroutine": "check_data",
        "description": "Core radius rc must be positive.",
    },
    {
        "name": "input_ncon_out_of_range",
        "pattern": r"test_data: must have 3 <= ncon <= 5, l=\s*(?P<l>\d)",
        "match_string": "test_data: must have 3 <= ncon",
        "line_count": 1,
        "subroutine": "check_data",
        "description": "Number of constraints ncon must be in [3, 5].",
    },
    {
        "name": "input_nbas_out_of_range",
        "pattern": r"test_data: must have ncon\+[23] <= nbas <= ncon\+5, l=\s*(?P<l>\d)",
        "match_string": "test_data: must have ncon+",
        "line_count": 1,
        "subroutine": "check_data",
        "description": "Basis size nbas must be in [ncon+2 (or 3), ncon+5].",
    },
    {
        "name": "input_qcut_nonpositive",
        "pattern": r"test_data: must have qcut>0\.0, l=\s*(?P<l>\d)",
        "match_string": "test_data: must have qcut>0.0",
        "line_count": 1,
        "subroutine": "check_data",
        "description": "Wavevector cutoff qcut must be positive.",
    },
    {
        "name": "input_rc_lt_rcloc",
        "pattern": r"test_data: rc < rc\(lloc\) for l =\s*(?P<l>\d)",
        "match_string": "test_data: rc < rc(lloc)",
        "line_count": 1,
        "subroutine": "check_data",
        "description": "Core radius rc is smaller than the local-potential radius rc(lloc).",
    },
    {
        "name": "input_debl_negative",
        "pattern": r"test_data: must have debl>0\.0\)\s*(?P<l>\d)",
        "match_string": "test_data: must have debl>0.0)",
        "line_count": 1,
        "subroutine": "check_data",
        "description": "Energy increment debl must not be negative.",
    },
    {
        "name": "input_nproj_out_of_range",
        "pattern": r"test_data: must have nproj in \[1,5\] \(0 OK lloc=4 or\s*(?P<l>\d)\)",
        "match_string": "test_data: must have nproj in",
        "line_count": 1,
        "subroutine": "check_data",
        "description": "Number of projectors nproj must be in [1, 5] (0 only for the local channel).",
    },
    {
        "name": "input_local_potential_invalid",
        "pattern": r"test_data: (?:must have 0 <= lloc <= lmax or lloc == 4|for lloc==4, must have rc > 0\.0|must have 1 <= lpopt <= 5)",
        "match_string": "test_data: ",
        "line_count": 1,
        "subroutine": "check_data",
        "description": "Invalid local-potential parameters (lloc, lpopt, or rc(lloc)).",
    },
    {
        "name": "input_model_core_invalid",
        # azadoks/oncvpsp: "0<= icmod <=4 or icmod=6", "fcfact>0.0 and rcfact>0.0 for icmod= 6";
        # smooooth api 9: "0.0<slack<=1.0 for icmod= 6 to 9", "0.0<r_s<=min(rc) for icmod= 8 or 9",
        # "error_shares>0.0"
        "pattern": (
            r"test_data: must have (?:0<= icmod <=4|fcfact>0\.0 (?:and rcfact>0\.0 )?for icmod"
            r"|rcfact>0\.0 for icmod|0\.0<slack<=1\.0 for icmod|0\.0<r_s<=min\(rc\) for icmod"
            r"|error_shares>0\.0)"
        ),
        "match_string": "test_data: must have",
        "line_count": 1,
        "subroutine": "check_data",
        "description": "Invalid model-core-charge parameters (icmod, fcfact, rcfact, slack, r_s "
        "or error_shares).",
    },
    {
        "name": "input_rlmax_too_small",
        "pattern": r"test_data: must have rlmax>rcmax",
        "match_string": "test_data: must have rlmax>rcmax",
        "line_count": 1,
        "subroutine": "check_data",
        "description": "Output grid extent rlmax must exceed the largest core radius.",
    },
    {
        "name": "input_invalid",
        "pattern": rf"^\s*test_data: (?!{_CHECK_DATA_SPECIFIC})\S",
        "match_string": "test_data: ",
        "line_count": 1,
        "subroutine": "check_data",
        "description": "Invalid input data (see message).",
    },
    {
        "name": "fortran_runtime_error",
        # Written to stderr by the Fortran runtime (gfortran: "Fortran runtime error:", ifort: "forrtl: severe");
        # only seen when stderr is merged into the parsed text
        "pattern": r"^(?:Fortran runtime error:|forrtl: severe|Program received signal)",
        "match_string": "r",
        "line_count": 1,
        "subroutine": "runtime",
        "description": "Fortran runtime error (I/O or formatted-transfer error, floating-point exception, or signal).",
    },
]

# azadoks/oncvpsp (branch smooooth)
WARNINGS += [
    {
        "name": "ghost_barrier_state_not_converged",
        # run_ghosts: the basis is cut below a barrier state it cannot converge, and the ghost test
        # covers energies only up to the cut (`OncvpspTextParser.ghost_basis_cuts`)
        "pattern": (
            rf"run_ghosts: WARNING barrier state n=\s*(?P<n>\d+) l=\s*(?P<l>\d+) near\s*"
            rf"(?P<e_barrier>{RE_FLOAT}) Ha not converged;"
        ),
        "match_string": "run_ghosts: WARNING barrier state",
        "line_count": 2,
        "subroutine": "run_ghosts",
        "description": "A barrier state of the ghost test's basis did not converge; the basis is cut "
        "there, and the ghost test covers energies only up to the cut.",
    },
    {
        "name": "ghost_basis_full",
        "pattern": r"run_ghosts: WARNING basis full below the cutoff, l=\s*(?P<l>\d+), nbmax=\s*(?P<nbmax>\d+)",
        "match_string": "run_ghosts: WARNING basis full",
        "line_count": 1,
        "subroutine": "run_ghosts",
        "description": "The ghost test's basis reached its largest size below the cutoff.",
    },
    {
        "name": "qroots_bisection_failed",
        "pattern": (
            rf"qroots: WARNING bisection failed, skipping, ll=\s*(?P<l>\d+), qlow=\s*(?P<qlow>{RE_FLOAT}),"
            rf" qhi=\s*(?P<qhi>{RE_FLOAT})"
        ),
        "match_string": "qroots: WARNING bisection failed",
        "line_count": 1,
        "subroutine": "qroots",
        "description": "Bisection for a root of the spherical Bessel function matching failed; the "
        "root is skipped.",
    },
    {
        "name": "modcore_budget_radius_misfit",
        "pattern": rf"modcore_budget: WARNING the radius\s*(?P<r_s>{RE_FLOAT}) of the least mismatch does not fit the budget",
        "match_string": "modcore_budget: WARNING the radius",
        "line_count": 1,
        "subroutine": "modcore_budget",
        "description": "The model-core radius of the least hardness mismatch does not fit the cutoff budget.",
    },
    {
        "name": "modcore_budget_deprecated",
        # azadoks/oncvpsp smooooth api 9 (wrn_budget_deprecated), before the model core block
        "pattern": r"modcore_budget: WARNING icmod 6 and 7 are deprecated",
        "match_string": "modcore_budget: WARNING icmod 6 and 7 are deprecated",
        "line_count": 1,
        "subroutine": "modcore_budget",
        "description": "The budgeted model cores that search r_s (icmod 6, 7) are deprecated: icmod 8 "
        "and 9 take r_s.",
    },
    {
        "name": "modcore_budget_no_minimum",
        "pattern": rf"modcore_budget: WARNING no minimum of the mismatch bracketed about\s*(?P<r>{RE_FLOAT})",
        "match_string": "modcore_budget: WARNING no minimum",
        "line_count": 1,
        "subroutine": "modcore_budget",
        "description": "No minimum of the hardness mismatch is bracketed about the radius.",
    },
    {
        "name": "modcore_budget_root_at_jump",
        "pattern": rf"modcore_budget: WARNING root at a jump of the function, x=\s*(?P<x>{RE_FLOAT})",
        "match_string": "modcore_budget: WARNING root at a jump",
        "line_count": 1,
        "subroutine": "modcore_budget (zeroin)",
        "description": "The root found is at a jump of the function, not a zero of it.",
    },
    {
        "name": "ae_scf_residual_floor",
        "pattern": (
            rf"ae_converged: WARNING scf stopped on the round-off floor of the residual, "
            rf"res=\s*(?P<res>{RE_FLOAT}), tolres=\s*(?P<tolres>{RE_FLOAT})"
        ),
        "match_string": "ae_converged: WARNING scf stopped",
        "line_count": 1,
        "subroutine": "ae_converged",
        # tolres 1e-11 from sratom/relatom, 1e-12 from psatom (azadoks/oncvpsp 360a78a), each
        # after its test configuration's header
        "description": "All-electron or pseudo-atom self-consistency stopped on the round-off "
        "floor of the residual, above its tolerance (benign).",
    },
    {
        "name": "test_config_ae_scf_not_converged",
        # The reason, after `ae_atom_no_solution`, why a test configuration has no output
        "pattern": r"(?P<subroutine>run_config(?:_r)?): WARNING self-consistency failed to converge",
        "match_string": "WARNING self-consistency failed to converge",
        "line_count": 1,
        "subroutine": "run_config(_r)",
        "description": "All-electron self-consistency failed for a test configuration.",
    },
    {
        "name": "test_config_ae_no_turning_point",
        "pattern": r"(?:run_config_r: WARNING )?no classical turning point error, iteration\s*(?P<iteration>\d+)",
        "match_string": "no classical turning point error, iteration",
        "line_count": 1,
        "subroutine": "run_config(_r)",
        "description": "No classical turning point in the all-electron atom of a test configuration.",
    },
    {
        "name": "rxpsh_inside_rc",
        "pattern": (
            rf"(?P<subroutine>run_phsft(?:_r)?): WARNING rxpsh inside the core radius, l=\s*(?P<l>\d+), "
            rf"rxpsh=\s*(?P<rxpsh>{RE_FLOAT}), rc=\s*(?P<rc>{RE_FLOAT})"
        ),
        "match_string": "WARNING rxpsh inside the core radius",
        "line_count": 1,
        "subroutine": "run_phsft(_r)",
        "description": "The log-derivative radius is inside the core radius of the channel; the "
        "phases are not comparable there.",
    },
    {
        "name": "rxpsh_inside_local_rc",
        "pattern": (
            rf"(?P<subroutine>run_phsft(?:_r)?): WARNING rxpsh inside the local potential core radius, "
            rf"l=\s*(?P<l>\d+), rxpsh=\s*(?P<rxpsh>{RE_FLOAT}), rc=\s*(?P<rc>{RE_FLOAT})"
        ),
        "match_string": "WARNING rxpsh inside the local potential core radius",
        "line_count": 1,
        "subroutine": "run_phsft(_r)",
        "description": "The log-derivative radius is inside the core radius of the local potential.",
    },
    {
        "name": "well_tail_share",
        # 74b1c54: E_resid at qcut of a well state confined by its well (well_tail "well" or
        # "confined"), with the well's tail against the free tail; `diagnostics` well_share has
        # both
        "pattern": rf"WARNING the well's tail changes it by\s*(?P<share_percent>{RE_FLOAT}) %:",
        "match_string": "WARNING the well's tail changes it by",
        "line_count": 2,
        "subroutine": "run_optimize",
        "description": "The well's tail changes the residual kinetic energy at qcut of a well state "
        "by more than 10 % against the free tail; its cutoffs carry the arbitrary well.",
    },
    {
        "name": "well_state_qcut_below_k",
        "pattern": rf"WARNING qcut is not above this well state's wave vector\s*(?P<k>{RE_FLOAT})",
        "match_string": "WARNING qcut is not above this well state's wave vector",
        "line_count": 2,
        "subroutine": "run_optimize",
        "description": "qcut is not above the wave vector of a well state, which oscillates faster "
        "than the cutoff: its cutoffs are not meaningful with any tail.",
    },
    {
        "name": "well_state_free_tail_too_fast",
        # azadoks/oncvpsp from e781800, well_tail = "well": the free tail of the well's share
        # grows too fast for its window (kappa L > 12), followed by
        # "    WARNING its window at qcut (kappa L > 12): no share"
        "pattern": r"WARNING this well state's free tail grows too fast for",
        "match_string": "WARNING this well state's free tail grows too fast for",
        "line_count": 2,
        "subroutine": "run_optimize",
        "description": "The free tail of a well state, at a negative energy, grows too fast for "
        "its window at qcut (kappa L > 12), so the well's share of its E_resid is not defined.",
    },
]

ERRORS += [
    {
        "name": "local_channel_ae_node",
        # lloc < 4: the local potential is inverted from the all-electron wave of channel lloc
        "pattern": (
            rf"oncvpsp: ERROR node of the local channel l=\s*(?P<l>\d+) all-electron wave near rc, "
            rf"at r=\s*(?P<r>{RE_FLOAT})"
        ),
        "match_string": "oncvpsp: ERROR node of the local channel",
        "line_count": 1,
        "subroutine": "oncvpsp",
        "description": "The all-electron wave of the local channel has a node near rc, where the "
        "local potential is inverted from it.",
    },
    {
        "name": "local_channel_ps_node",
        "pattern": (
            rf"run_vkb: ERROR node of the local channel's pseudo wave within reach of rc, "
            rf"l=\s*(?P<l>\d+) r=\s*(?P<r>{RE_FLOAT})"
        ),
        "match_string": "run_vkb: ERROR node of the local channel",
        "line_count": 1,
        "subroutine": "run_vkb",
        "description": "The pseudo wave of the local channel has a node within reach of rc.",
    },
    {
        "name": "ode_rc_der_mismatch",
        # The second line gives uu' from up and from stencils on uu
        "pattern": (
            rf"ode_rc_der: ERROR uu and up disagree for l=\s*(?P<l>\d+) at rc=\s*(?P<rc>{RE_FLOAT})"
        ),
        "match_string": "ode_rc_der: ERROR",
        "line_count": 2,
        "subroutine": "ode_rc_der",
        "description": "The derivative of the all-electron wave at rc from the ODE disagrees with "
        "its finite differences.",
    },
    {
        "name": "optimize_q_grid_too_long",
        "pattern": r"run_optimize: ERROR q grid longer than the convergence curve, nq=\s*(?P<nq>\d+), ncurve=\s*(?P<ncurve>\d+)",
        "match_string": "run_optimize: ERROR q grid longer",
        "line_count": 1,
        "subroutine": "run_optimize",
        "description": "The q grid of the optimization is longer than the convergence curve.",
    },
    {
        "name": "eresid_tail_too_short",
        # azadoks/oncvpsp smooooth; nr can be an overflowed negative integer
        "pattern": r"eresid: ERROR tail too short, nr=\s*(?P<nr>-?\d+)",
        "match_string": "eresid: ERROR tail too short",
        "line_count": 1,
        "subroutine": "eresid",
        "description": "The linear mesh of the residual kinetic energy's tail beyond rc has too "
        "few points.",
    },
    {
        "name": "match_point_failed",
        "pattern": r"match_point: ERROR (?P<reason>matching window at mch=.*leaves the mesh|no sign change below mch=)",
        "match_string": "match_point: ERROR",
        "line_count": 1,
        "subroutine": "match_point",
        "description": "The matching point of an inward/outward wave solution could not be located "
        "on the mesh.",
    },
    {
        "name": "mesh_tail_too_short",
        "pattern": (
            rf"mesh_tail: ERROR wave function vanishes too close to rc: rc=\s*(?P<rc>{RE_FLOAT}), "
            rf"last nonzero point\s*(?P<r>{RE_FLOAT})"
        ),
        "match_string": "mesh_tail: ERROR",
        "line_count": 1,
        "subroutine": "mesh_tail (free_tail_m)",
        "description": "With well_tail = free, the wave function vanishes too close to rc for its tail.",
    },
    {
        "name": "free_tail_qcut_too_low",
        "pattern": rf"free_tail_width: ERROR qcut=\s*(?P<qcut>{RE_FLOAT}) is not above the wave vector",
        "match_string": "free_tail_width: ERROR",
        "line_count": 2,
        "subroutine": "free_tail_width (free_tail_m)",
        "description": "With well_tail = free, qcut is not above the wave vector of the well state.",
    },
    {
        "name": "free_tail_negative_ep",
        # azadoks/oncvpsp from fd5aaf0, after the Wellstate header of the state, followed by
        # "the well state of l= 2, n=  5 at ep=   -1.608320 Ha has a growing tail;" and
        # "raise its ep (debl), or use well_tail = "confined" or "well""
        "pattern": r'oncvpsp: ERROR well_tail = "free" needs ep >= 0:',
        "match_string": 'oncvpsp: ERROR well_tail = "free" needs ep',
        "line_count": 3,
        "subroutine": "oncvpsp",
        "description": "With well_tail = free, a well state is below 0 Ha, where its free tail "
        "grows and its residual kinetic energy is lost to round-off.",
    },
    {
        "name": "free_tail_too_fast",
        # azadoks/oncvpsp from e781800, followed by
        # "grows too fast for its window (kappa L =   20.74 > 12): raise qcut or the state's energy"
        "pattern": rf"free_tail_width: ERROR the free tail of the well state at\s*(?P<e>{RE_FLOAT}) Ha with qcut=\s*(?P<qcut>{RE_FLOAT})",
        "match_string": "free_tail_width: ERROR the free tail",
        "line_count": 2,
        "subroutine": "free_tail_width (free_tail_m)",
        "description": "With well_tail = free, the free tail of a well state at a negative energy "
        "grows too fast for its window at qcut (kappa L > 12).",
    },
    {
        "name": "output_mesh_error",
        "pattern": r"(?P<subroutine>om_local_values|outmesh): ERROR (?!output mesh to)(?P<reason>.*)",
        "match_string": ": ERROR",
        "line_count": 1,
        "subroutine": "outmesh_m",
        "description": "The pseudopotential could not be put on the output mesh.",
    },
    {
        "name": "output_mesh_beyond_log_mesh",
        "pattern": (
            rf"outmesh: ERROR output mesh to\s*(?P<rlmax>{RE_FLOAT}) bohr, beyond the log mesh,\s*"
            rf"(?P<rmax>{RE_FLOAT})"
        ),
        "match_string": "outmesh: ERROR output mesh to",
        "line_count": 1,
        "subroutine": "outmesh_m",
        "description": "The output mesh extends beyond the log mesh (output_grid rlmax and log_mesh rmax).",
    },
    {
        "name": "outwf_setup_error",
        "pattern": r"outwf_setup: ERROR (?P<reason>.*)",
        "match_string": "outwf_setup: ERROR",
        "line_count": 1,
        "subroutine": "outwf_setup",
        "description": "The piecewise potential could not be set up for the outward wave solution.",
    },
    {
        "name": "vploc_bad_lpopt",
        "pattern": r"vploc_fit: ERROR lpopt\s*(?P<lpopt>-?\d+)",
        "match_string": "vploc_fit: ERROR lpopt",
        "line_count": 1,
        "subroutine": "vploc_fit",
        "description": "Unsupported lpopt for the local potential polynomial.",
    },
    {
        "name": "input_teter_grid_invalid",
        "pattern": r"check_data: ERROR icmod=4 needs 0<(?P<parameter>fcfact|rcfact)_min",
        "match_string": "check_data: ERROR icmod=4 needs",
        "line_count": 1,
        "subroutine": "check_data",
        "description": "The icmod 4 grid of fcfact or rcfact is invalid (min, max, step).",
    },
    {
        "name": "input_rxpsh_inside_rc",
        "pattern": rf"check_data: ERROR rxpsh not beyond every core radius, rxpsh=\s*(?P<rxpsh>{RE_FLOAT})",
        "match_string": "check_data: ERROR rxpsh",
        "line_count": 1,
        "subroutine": "check_data",
        "description": "The log-derivative radius rxpsh is not beyond every core radius.",
    },
    {
        "name": "rxpsh_beyond_mesh",
        "pattern": rf"(?P<subroutine>run_phsft(?:_r)?): ERROR rxpsh beyond radial mesh, rxpsh=\s*(?P<rxpsh>{RE_FLOAT})",
        "match_string": "ERROR rxpsh beyond radial mesh",
        "line_count": 1,
        "subroutine": "run_phsft(_r)",
        "description": "The log-derivative radius rxpsh is beyond the radial mesh.",
    },
    {
        "name": "functionals_bad_nspin",
        # `error stop` with the message: written to stderr, so only seen when it is merged
        "pattern": r"functionals: ERROR bad nspin definition",
        "match_string": "functionals: ERROR bad nspin",
        "line_count": 1,
        "subroutine": "functionals",
        "description": "Internal error: the potential array does not match the functional's nspin.",
    },
    {
        "name": "wellstate_energy_jump",
        "pattern": (
            rf"wellstate: ERROR E\(rwell\) jumps across ep, n=\s*(?P<n>\d+)\s+l=\s*(?P<l>\d+)\s+"
            rf"E-ep=\s*(?P<e_minus_ep>{RE_FLOAT})"
        ),
        "match_string": "wellstate: ERROR E(rwell) jumps across ep",
        "line_count": 1,
        "subroutine": "wellstate",
        "description": "The energy of the well state jumps across the requested energy ep as the "
        "well radius varies, so no radius gives ep; E-ep is the gap left at the jump.",
    },
    {
        "name": "ionic_valence_lschvkbb_failed",
        # ierr 200 is ionic_valence_bound_ghost
        "pattern": r"ionic_valence: ERROR lschvkbb n,l,ierr=\s*(?P<n>\d+)\s+(?P<l>\d+)\s+(?P<ierr>(?!200\b)-?\d+)",
        "match_string": "ionic_valence: ERROR lschvkbb",
        "line_count": 1,
        "subroutine": "ionic_valence",
        "description": "A pseudo valence state could not be solved in the separable potential with "
        "the ionic tail beyond rcmax.",
    },
    {
        "name": "ionic_valence_bound_ghost",
        # azadoks/oncvpsp from 61e586a: lschvkbb's ierr 200, followed by the two lines of
        # vkb_bound_ghost
        "pattern": r"ionic_valence: ERROR lschvkbb n,l,ierr=\s*(?P<n>\d+)\s+(?P<l>\d+)\s+(?P<ierr>200)\b",
        "match_string": "ionic_valence: ERROR lschvkbb",
        "line_count": 3,
        "subroutine": "ionic_valence",
        "description": "A pseudo valence state could not be solved in the separable potential with "
        "the ionic tail because the projectors bind a state of the same l below it that the "
        "node count does not expect: a bound ghost.",
    },
    {
        "name": "ionic_screening_not_converged",
        # azadoks/oncvpsp smooooth, after the model core
        "pattern": (
            r"ionic_screening: ERROR the ionic tail did not converge in\s*(?P<maxiter>\d+) steps, "
            rf"change =\s*(?P<change>{RE_FLOAT})"
        ),
        "match_string": "ionic_screening: ERROR the ionic tail did not converge",
        "line_count": 1,
        "subroutine": "ionic_screening (ionic_tail_m)",
        "description": "The fixed point of the screened local potential with the ionic tail "
        "beyond rcmax did not converge.",
    },
    {
        "name": "well_n_not_confined",
        "pattern": r'oncvpsp: ERROR well_n is for well_tail = "confined" only',
        "match_string": "oncvpsp: ERROR well_n is for well_tail",
        "line_count": 1,
        "subroutine": "oncvpsp",
        "description": 'well_n is set for some l while well_tail is not "confined".',
    },
    {
        "name": "well_n_below_first_well_state",
        "pattern": (
            r"oncvpsp: ERROR well_n for l=\s*(?P<l>\d+) is\s*(?P<well_n>\d+), below the first "
            r"well state's n,\s*(?P<n>\d+)"
        ),
        "match_string": "oncvpsp: ERROR well_n for l=",
        "line_count": 1,
        "subroutine": "oncvpsp",
        "description": "well_n is below the n that ONCVPSP gives the channel's first well state, "
        "one above the last bound state's.",
    },
    {
        "name": "well_norm_space_free_tail",
        "pattern": r'oncvpsp: ERROR well_norm = "space" \(1\) is not defined for well_tail = "free"',
        "match_string": 'oncvpsp: ERROR well_norm = "space"',
        "line_count": 1,
        "subroutine": "oncvpsp",
        "description": 'well_norm = "space" with well_tail = "free": the free tail has no finite norm.',
    },
    {
        "name": "ecut_weight_occupied_channel",
        "pattern": (
            r"oncvpsp: ERROR ecut_weight for l=\s*(?P<l>\d+) is for a channel with no occupied state"
        ),
        "match_string": "oncvpsp: ERROR ecut_weight for l=",
        "line_count": 1,
        "subroutine": "oncvpsp",
        "description": "ecut_weight is set for a channel with an occupied valence state; it is for "
        "the channels with none.",
    },
]

# A line of azadoks/oncvpsp's `<routine>: ERROR ...` form. A run which stops without a recognized
# error is attributed to its last such line (see `analyze_failure`); in a completed run these are
# not errors (e.g. `lschfb: ERROR ll > nn`, which ONCVPSP recovers from)
UNRECOGNIZED_ERROR: dict[str, Any] = {
    "name": "unrecognized_error",
    "pattern": r"^\s*(?P<subroutine>[A-Za-z_][\w-]*(?: \d+)?): (?:\w+ )?ERROR\b",
    "match_string": "ERROR",
    "line_count": 1,
    "subroutine": "Unknown",
    "description": "Error message of the program not in the catalogue.",
}
