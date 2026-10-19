"""ONCVPSP warning and error definitions.

A line is attributed to a definition if it contains `match_string` and `pattern` matches it (`re.search`).
`line_count` is the number of lines (starting from the matched line) that make up the message.
The wording of many messages differs between ONCVPSP v3, v4, and METAPSP; the patterns cover all known variants.
"""

from typing import Any

RE_FLOAT: str = r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][+-]?\d+)?"

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
        "pattern": r"read_ae_file: ERROR (?P<reason>(?:atsym|reference configuration|radial mesh|test configuration).*)",
        "match_string": "read_ae_file: ERROR",
        "line_count": 1,
        "subroutine": "read_ae_file",
        "description": "The all-electron data file was written for a different atom, reference or test configuration, or radial mesh than the input.",
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
        "name": "vkb_bound_state_not_converged",
        # "oncvpsp: lschvkbb ERROR" (v3, v4), "oncvpsp-752: lschvkbb_m ERROR" (METAPSP)
        "pattern": r"oncvpsp(?:-\d+)?: lschvkbb(?:_m)? ERROR\s+(?P<n>\d+)\s+(?P<l>\d+)\s+(?P<ierr>-?\d+)",
        "match_string": "lschvkbb",
        "line_count": 1,
        "subroutine": "oncvpsp",
        "description": "Unable to solve for bound state in separable pseudopotential when constructing final pseudo-states and densities.",
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
        # azadoks/oncvpsp: "0<= icmod <=4 or icmod=6", "fcfact>0.0 and rcfact>0.0 for icmod= 6"
        "pattern": (
            r"test_data: must have (?:0<= icmod <=4|fcfact>0\.0 (?:and rcfact>0\.0 )?for icmod"
            r"|rcfact>0\.0 for icmod)"
        ),
        "match_string": "test_data: must have",
        "line_count": 1,
        "subroutine": "check_data",
        "description": "Invalid model-core-charge parameters (icmod, fcfact, or rcfact).",
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
