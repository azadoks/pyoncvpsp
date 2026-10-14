"""Tests for the attribution of ONCVPSP messages to stages, channels and input parameters.

The outputs in `data/attribution` were generated with ONCVPSP 4.0.1 from the pseudo-dojo Si input by
perturbing one parameter at a time (see the test names); the truncated ones are a successful run cut
short to mimic a killed process.
"""

import gzip
from pathlib import Path

import pytest

from pyoncvpsp.io.output import (
    ATTRIBUTION,
    ERRORS,
    WARNINGS,
    Attribution,
    Level,
    OncvpspTextParser,
    Scope,
    Stage,
)
from pyoncvpsp.io.output._attribution import LineContext, attribution_for, line_contexts

DATA_DIR = Path(__file__).parent / "data" / "attribution"


def read(name: str) -> str:
    with gzip.open(DATA_DIR / f"{name}.out.gz", "rt", encoding="latin-1") as f:
        return f.read()


def parse(name: str) -> OncvpspTextParser:
    return OncvpspTextParser(text=read(name), check=False)


# --- Definitions ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("level", "name"),
    [(Level.WARNING, d["name"]) for d in WARNINGS]
    + [(Level.ERROR, d["name"]) for d in ERRORS],
    ids=lambda x: x if isinstance(x, str) else x.value,
)
def test_every_definition_has_attribution(level, name):
    meta = attribution_for(level, name)
    assert "stage" in meta and "scope" in meta, (
        f"{level.value}:{name} lacks attribution metadata"
    )
    assert meta["stage"] is None or isinstance(meta["stage"], Stage)
    assert meta["scope"] is None or isinstance(meta["scope"], Scope)
    assert meta.get("l_rule", "context") in ("pending", "context", "none")


def test_attribution_has_no_stale_entries():
    names = {d["name"] for d in WARNINGS} | {d["name"] for d in ERRORS}
    for key in ATTRIBUTION:
        assert key.split(":", 1)[-1] in names, (
            f"attribution for unknown definition {key}"
        )


# --- Line contexts ----------------------------------------------------------------------------------

V4_HEADER = ["Calculating optimized projector #   1", "", " for l=   {l}"]


def test_line_contexts_follow_stage_headers():
    lines = [
        "# ATOM AND REFERENCE CONFIGURATION",
        "# TEST CONFIGURATIONS",
        "# ncnf",
        "    0",
        "# nvcnf",
        "#   n    l    f",
        "",
        "Reference configufation results",
        "Begin loop to  construct optimized pseudo wave functions",
        *[s.format(l=0) for s in V4_HEADER],
        "   Wellstate for l =  1  n =  3",
        *[s.format(l=1) for s in V4_HEADER],
        "Calculating optimized projector #   2",
        "",
        " for l=   1",
        "Construct Vanderbilt / Kleinmman-Bylander projectors",
        "B matrix Hermiticity error, ll=   1",
        "Model core correction analysis",
        "Pseudoatom total energy   -3.760249",
        "Diagnostic tests using Vanderbilt-Kleinman-Bylander pseudopotentials",
        "   1   1.7155494   1.7676372  -0.1499815   0.0000014   1.0000187   1.0000126",
        "Test configuration 1",
        "DATA FOR PLOTTING",
        "scattering, iprj= 1,  l= 2, all-electron wave function, pseudo w-f",
        "log derivativve data for plotting, l= 0",
        "GNUSCRIPT",
        "Begin PSP_UPF",
    ]
    ctx = line_contexts(lines)
    stages = [c.stage for c in ctx]
    assert stages[0] == Stage.INPUT
    assert stages[5] == Stage.AE_REFERENCE  # after the input echo
    assert stages[8] == Stage.SEMILOCAL
    assert ctx[11].header_l == 0 and ctx[11].iproj == 1 and ctx[11].pending_l == 1
    assert ctx[12].well_l == 1 and ctx[12].pending_l == 1 and ctx[12].current_l == 0
    assert ctx[15].header_l == 1 and ctx[15].pending_l == 2
    assert ctx[18].iproj == 2
    assert ctx[19].stage == Stage.VKB and ctx[19].marker_l is None
    assert ctx[20].marker_l == 1
    assert ctx[21].stage == Stage.MODEL_CORE
    assert ctx[23].stage == Stage.DIAGNOSTICS and ctx[24].marker_l == 1
    assert ctx[25].stage == Stage.TEST_CONFIG and ctx[25].config_index == 1
    assert (
        ctx[26].stage == Stage.LOGDER_PLOT
        and ctx[27].marker_l == 2
        and ctx[28].marker_l == 0
    )
    assert ctx[-1].stage == Stage.OUTPUT


def test_pending_l_before_any_header():
    assert LineContext(stage=Stage.SEMILOCAL).pending_l == 0


# --- Synthetic messages -----------------------------------------------------------------------------


def _semilocal_prefix(l_done: int) -> list[str]:
    lines = [
        "Reference configufation results",
        "Begin loop to  construct optimized pseudo wave functions",
    ]
    for l in range(l_done + 1):  # noqa: E741
        lines += [s.format(l=l) for s in V4_HEADER]
        lines += ["    Residual kinetic energy error=  4.00D-06 Ha"]
    return lines


def test_qroots_error_is_attributed_to_the_next_channel():
    # `qroots` runs before the per-l header, so its error follows the block of the previous l
    lines = _semilocal_prefix(1) + ["qroots: ERROR - failed to find nroot roots"]
    parser = OncvpspTextParser(lines=lines, check=False)
    failure = parser.failure
    assert failure is not None
    assert failure.name == "too_few_qroots"
    assert (failure.stage, failure.scope, failure.l) == (Stage.SEMILOCAL, Scope.ELL, 2)
    assert failure.attribution == Attribution.INFERRED
    assert "pseudopotentials__qcut__2" in failure.implicated_parameters
    assert failure.last_l_reached == 1


def test_qroots_error_for_first_channel():
    lines = _semilocal_prefix(-1) + ["qroots: ERROR failed to find root"]
    failure = OncvpspTextParser(lines=lines, check=False).failure
    assert (
        failure is not None
        and failure.l == 0
        and failure.attribution == Attribution.INFERRED
    )


def test_matrix_eigenvalue_error_stage_from_position():
    vkb = _semilocal_prefix(0) + [
        "Construct Vanderbilt / Kleinmman-Bylander projectors",
        "B matrix Hermiticity error, ll=   0",
        "run_vkb: B matrix eigenvalue ERROR, info=   1",
    ]
    failure = OncvpspTextParser(lines=vkb, check=False).failure
    assert failure is not None
    assert (failure.stage, failure.scope, failure.l) == (Stage.VKB, Scope.ELL, 0)
    ghosts = vkb[:-1] + [
        "Diagnostic tests using Vanderbilt-Kleinman-Bylander pseudopotentials",
        " Testing for bound ghosts",
        "run_ghosts: hamiltonian matrix eigenvalue ERROR, info=   1",
    ]
    failure = OncvpspTextParser(lines=ghosts, check=False).failure
    assert failure is not None
    assert (failure.stage, failure.scope, failure.l) == (
        Stage.DIAGNOSTICS,
        Scope.LOCAL,
        None,
    )


def test_rxpsh_error_in_completed_run_is_not_a_failure():
    parser = parse("testcfg_bad")
    assert parser.completed
    assert parser.failure is None
    assert parser.last_stage_reached == Stage.OUTPUT


def test_stderr_runtime_error_appended_to_stdout():
    stdout = read("debl_zero_l2")
    stderr = (DATA_DIR / "debl_zero_l2.stderr").read_text(encoding="latin-1")
    parser = OncvpspTextParser(text=stdout + "\n" + stderr, check=False)
    names = [m.name for m in parser.messages if m.level == Level.ERROR]
    assert names == ["debl_not_allowed", "fortran_runtime_error"]
    failure = parser.failure
    assert failure is not None and failure.name == "debl_not_allowed" and failure.l == 2
    runtime = [m for m in parser.messages if m.name == "fortran_runtime_error"][0]
    assert runtime.stage == Stage.SEMILOCAL  # context of the last stdout line


# --- Generated failure outputs ----------------------------------------------------------------------

FAILURES = {
    # name: (primary, stage, scope, l, attribution, last_stage_reached)
    "rcloc_gt_rc": (
        "input_rc_lt_rcloc",
        Stage.INPUT,
        Scope.ELL_AND_LOCAL,
        0,
        Attribution.EXPLICIT,
        Stage.INPUT,
    ),
    "lloc_bad": (
        "input_local_potential_invalid",
        Stage.INPUT,
        Scope.LOCAL,
        None,
        Attribution.NONE,
        Stage.INPUT,
    ),
    "rc_tiny_l1": (
        "vkb_bound_state_not_converged",
        Stage.VKB_SOLVE,
        Scope.ELL_AND_LOCAL,
        1,
        Attribution.EXPLICIT,
        Stage.VKB,
    ),
    "qcut_high_l1": (
        "first_pswf_has_node",
        Stage.SEMILOCAL,
        Scope.ELL,
        1,
        Attribution.INFERRED,
        Stage.SEMILOCAL,
    ),
    "debl_zero_l2": (
        "debl_not_allowed",
        Stage.SEMILOCAL,
        Scope.ELL,
        2,
        Attribution.INFERRED,
        Stage.SEMILOCAL,
    ),
    "debl_zero_l0": (
        "debl_not_allowed",
        Stage.SEMILOCAL,
        Scope.ELL,
        0,
        Attribution.INFERRED,
        Stage.SEMILOCAL,
    ),
    "ep_huge_l2": (
        "wellstate_not_converged",
        Stage.SEMILOCAL,
        Scope.ELL,
        2,
        Attribution.EXPLICIT,
        Stage.SEMILOCAL,
    ),
    "truncated_semilocal_l1": (
        "incomplete_output",
        Stage.SEMILOCAL,
        Scope.ELL,
        1,
        Attribution.INFERRED,
        Stage.SEMILOCAL,
    ),
    "truncated_plot": (
        "incomplete_output",
        Stage.LOGDER_PLOT,
        Scope.ELL,
        1,
        Attribution.INFERRED,
        Stage.LOGDER_PLOT,
    ),
}


@pytest.mark.parametrize("name", sorted(FAILURES), ids=sorted(FAILURES))
def test_failure_attribution(name):
    primary, stage, scope, l, attribution, last_stage = FAILURES[name]  # noqa: E741
    parser = parse(name)
    assert not parser.completed
    failure = parser.failure
    assert failure is not None
    assert failure.name == primary
    assert failure.stage == stage
    assert failure.scope == scope
    assert failure.l == l
    assert failure.attribution == attribution
    assert failure.last_stage_reached == last_stage
    assert parser.last_stage_reached == last_stage
    if failure.name == "incomplete_output":
        assert failure.errors == []
    else:
        assert failure.primary is failure.errors[0]
    d = failure.as_dict()
    assert d["stage"] == stage.value and d["l"] == l
    assert all(isinstance(k, str) for k in d["implicated_parameters"])


def test_implicated_parameters_follow_scope():
    semilocal = parse("qcut_high_l1").failure
    assert semilocal is not None
    assert semilocal.implicated_parameters == [
        "pseudopotentials__rc__1",
        "pseudopotentials__qcut__1",
        "pseudopotentials__ncon__1",
        "pseudopotentials__nbas__1",
        "vkb_projectors__nproj__1",
    ]  # 3p is bound (no ep); the node is in the first projector (no debl)
    assert semilocal.primary.iproj == 1

    vkb_solve = parse("rc_tiny_l1").failure
    assert vkb_solve is not None
    params = vkb_solve.implicated_parameters
    assert "pseudopotentials__rc__1" in params and "vkb_projectors__debl__1" in params
    assert "local_potential__rcloc" in params and "local_potential__lloc" in params
    assert not any(k.startswith("pseudopotentials__rc__0") for k in params)

    debl = parse("debl_zero_l0").failure
    assert debl is not None
    assert "vkb_projectors__debl__0" in debl.implicated_parameters
    assert "pseudopotentials__ep__0" not in debl.implicated_parameters  # 3s is bound

    inp = parse("lloc_bad").failure
    assert inp is not None
    assert inp.implicated_parameters == [
        "local_potential__lloc",
        "local_potential__lpopt",
        "local_potential__rcloc",
        "local_potential__dvloc0",
    ]


def test_test_configuration_messages_point_at_the_input_configuration():
    parser = parse("testcfg_bad")
    messages = {m.name: m for m in parser.messages}
    # ONCVPSP numbers test configurations from 0 = reference configuration
    skipped = messages["lschvkbb_not_converged"]
    assert skipped.stage == Stage.TEST_CONFIG and skipped.config_index == 1
    assert skipped.implicated_parameters == ["test_configurations__0"]
    assert skipped.l == 1 and skipped.n == 2  # from the message itself (pseudo-atom n)
    assert messages["fully_non_local_ps_atom_no_solution"].config_index == 1
    diag = messages["lschvkbbe_not_converged"]
    assert diag.stage == Stage.DIAGNOSTICS and diag.l == 0 and diag.iproj == 2


def test_messages_sorted_and_serializable():
    parser = parse("debl_zero_l2")
    lines = [m.line_numbers[0] for m in parser.messages]
    assert lines == sorted(lines)
    for message in parser.messages:
        d = message.as_dict()
        assert d["level"] in ("warning", "error")
        assert isinstance(d["stage"], str) and isinstance(d["scope"], str)
    assert [m.name for m in parser.messages if m.level == Level.ERROR] == [
        e["name"] for e in parser.errors
    ]
