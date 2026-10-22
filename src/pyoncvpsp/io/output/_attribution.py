"""Attribute ONCVPSP messages to the program stage and the input parameters that produced them.

Messages that do not print `l` themselves are attributed using the context of their line (see
`_sections`). The main program's loop over `l` calls `wellstate`, the `debl` check and `qroots`
*before* printing the per-`l` header `Calculating optimized projector`, so their messages appear
after the block of the previous `l` (`l_rule = "pending"`).
"""

import re
from typing import TYPE_CHECKING, Any, Mapping

from ._data import ERRORS, UNRECOGNIZED_ERROR, WARNINGS
from ._messages import (
    DEFAULT_SCOPE,
    Attribution,
    Level,
    OncvpspFailure,
    OncvpspMessage,
    Scope,
    Stage,
    implicated_parameters,
)
from ._sections import LineContext

if TYPE_CHECKING:
    from pyoncvpsp.projectors import ProjectorLayout

    from ._text import OncvpspTextParser

# Attribution metadata per message name (keys `"warning:<name>"` / `"error:<name>"` take precedence
# over `"<name>"` where the same name is used for both a warning and an error).
#   stage: Stage | None  -- None: infer from the position in the output
#   scope: Scope | None  -- None: default for the stage
#   l_rule: "pending" | "context" (default) | "none"
#   iproj: int  -- projector the message is always about, when it does not print one
ATTRIBUTION: dict[str, dict[str, Any]] = {
    # --- warnings
    "icmod1_not_converged": {"stage": Stage.MODEL_CORE, "scope": Scope.MODEL_CORE},
    "icmod4_nm_not_converged": {"stage": Stage.MODEL_CORE, "scope": Scope.MODEL_CORE},
    "atom_not_converged": {"stage": None, "scope": None, "l_rule": "none"},
    "warning:ldiracfb_not_converged": {
        "stage": Stage.TEST_CONFIG,
        "scope": Scope.TEST_CONFIG,
    },
    "runconfig_lschfb_not_converged": {
        "stage": Stage.TEST_CONFIG,
        "scope": Scope.TEST_CONFIG,
    },
    "ae_atom_no_solution": {"stage": Stage.TEST_CONFIG, "scope": Scope.TEST_CONFIG},
    "fully_non_local_ps_atom_no_solution": {
        "stage": Stage.TEST_CONFIG,
        "scope": Scope.TEST_CONFIG,
    },
    "psatom_not_converged": {"stage": Stage.TEST_CONFIG, "scope": Scope.TEST_CONFIG},
    "psatom_r_not_converged": {"stage": Stage.TEST_CONFIG, "scope": Scope.TEST_CONFIG},
    "localized_bound_state_for_projector": {
        "stage": Stage.SEMILOCAL,
        "scope": Scope.ELL,
    },
    "moderately_localized_bound_state_for_projector": {
        "stage": Stage.SEMILOCAL,
        "scope": Scope.ELL,
    },
    "scattering_first_projector": {"stage": Stage.SEMILOCAL, "scope": Scope.ELL},
    "wellstate_negative_energy_state": {"stage": Stage.SEMILOCAL, "scope": Scope.ELL},
    "negative_energy_second_projector": {
        "stage": Stage.SEMILOCAL,
        "scope": Scope.ELL,
        "l_rule": "pending",
        "iproj": 2,
    },
    "lschvkbbe_not_converged": {
        "stage": Stage.DIAGNOSTICS,
        "scope": Scope.ELL_AND_LOCAL,
    },
    "diag_lschvkbb_not_converged": {
        "stage": Stage.DIAGNOSTICS,
        "scope": Scope.ELL_AND_LOCAL,
    },
    "first_pswf_has_node": {"stage": Stage.SEMILOCAL, "scope": Scope.ELL},
    "first_pswf_has_node_qcut_reduced": {"stage": Stage.SEMILOCAL, "scope": Scope.ELL},
    "eresid_flat": {"stage": Stage.SEMILOCAL, "scope": Scope.ELL},
    "ghost_barrier_state_not_converged": {
        "stage": Stage.DIAGNOSTICS,
        "scope": Scope.ELL_AND_LOCAL,
    },
    "ghost_basis_full": {"stage": Stage.DIAGNOSTICS, "scope": Scope.ELL_AND_LOCAL},
    "qroots_bisection_failed": {"stage": Stage.SEMILOCAL, "scope": Scope.ELL},
    "modcore_budget_radius_misfit": {
        "stage": Stage.MODEL_CORE,
        "scope": Scope.MODEL_CORE,
    },
    "modcore_budget_no_minimum": {"stage": Stage.MODEL_CORE, "scope": Scope.MODEL_CORE},
    "modcore_budget_root_at_jump": {
        "stage": Stage.MODEL_CORE,
        "scope": Scope.MODEL_CORE,
    },
    # The reference atom, or the all-electron atom of a test configuration
    "ae_scf_residual_floor": {"stage": None, "scope": None, "l_rule": "none"},
    "test_config_ae_scf_not_converged": {
        "stage": Stage.TEST_CONFIG,
        "scope": Scope.TEST_CONFIG,
    },
    "test_config_ae_no_turning_point": {
        "stage": Stage.TEST_CONFIG,
        "scope": Scope.TEST_CONFIG,
    },
    "rxpsh_inside_rc": {"stage": Stage.LOGDER_PLOT, "scope": Scope.ELL},
    # l is that of the local potential, above lmax
    "rxpsh_inside_local_rc": {"stage": Stage.LOGDER_PLOT, "scope": Scope.LOCAL},
    # --- errors: input
    "bad_input": {"stage": Stage.INPUT, "scope": Scope.INPUT},
    "input_rc_nonpositive": {"stage": Stage.INPUT, "scope": Scope.ELL},
    "input_ncon_out_of_range": {"stage": Stage.INPUT, "scope": Scope.ELL},
    "input_nbas_out_of_range": {"stage": Stage.INPUT, "scope": Scope.ELL},
    "input_qcut_nonpositive": {"stage": Stage.INPUT, "scope": Scope.ELL},
    "input_rc_lt_rcloc": {"stage": Stage.INPUT, "scope": Scope.ELL_AND_LOCAL},
    "input_debl_negative": {"stage": Stage.INPUT, "scope": Scope.ELL},
    "input_nproj_out_of_range": {"stage": Stage.INPUT, "scope": Scope.ELL},
    "input_local_potential_invalid": {"stage": Stage.INPUT, "scope": Scope.LOCAL},
    "input_model_core_invalid": {"stage": Stage.INPUT, "scope": Scope.MODEL_CORE},
    "input_rlmax_too_small": {"stage": Stage.INPUT, "scope": Scope.GLOBAL},
    "input_invalid": {"stage": Stage.INPUT, "scope": Scope.INPUT},
    "input_file_read_error": {"stage": Stage.INPUT, "scope": Scope.INPUT},
    "input_data_read_error": {"stage": Stage.INPUT, "scope": Scope.INPUT},
    "input_file_read_failed": {"stage": Stage.INPUT, "scope": Scope.INPUT},
    "ae_file_required": {"stage": Stage.INPUT, "scope": Scope.INPUT},
    "ae_file_write_failed": {"stage": Stage.AE_REFERENCE, "scope": Scope.INPUT},
    "ae_file_mismatch": {"stage": Stage.AE_REFERENCE, "scope": Scope.INPUT},
    "ae_file_read_failed": {"stage": Stage.AE_REFERENCE, "scope": Scope.INPUT},
    "fr_lloc_not_supported": {"stage": Stage.INPUT, "scope": Scope.LOCAL},
    "libxc_not_available": {"stage": Stage.INPUT, "scope": Scope.INPUT},
    "vout_bad_input": {"stage": Stage.INPUT, "scope": Scope.INPUT},
    # --- errors: all-electron reference atom
    "ae_reference_not_converged": {"stage": Stage.AE_REFERENCE, "scope": Scope.GLOBAL},
    # The main program prints stale (n, l) here (upstream bug): l is inferred from the position
    "lschfb_not_converged": {"stage": None, "scope": Scope.GLOBAL, "l_rule": "pending"},
    "error:ldiracfb_not_converged": {
        "stage": None,
        "scope": Scope.GLOBAL,
        "l_rule": "pending",
    },
    "no_classical_turning_point": {"stage": None, "scope": None},
    # --- errors: semilocal pseudopotentials
    "wellstate_not_converged": {"stage": Stage.SEMILOCAL, "scope": Scope.ELL},
    "wellstate_no_bound_state": {"stage": Stage.SEMILOCAL, "scope": Scope.ELL},
    "debl_not_allowed": {
        "stage": Stage.SEMILOCAL,
        "scope": Scope.ELL,
        "l_rule": "pending",
    },
    "too_few_qroots": {
        "stage": Stage.SEMILOCAL,
        "scope": Scope.ELL,
        "l_rule": "pending",
    },
    "qroots_no_root": {
        "stage": Stage.SEMILOCAL,
        "scope": Scope.ELL,
        "l_rule": "pending",
    },
    "qcut_reduction_failed": {"stage": Stage.SEMILOCAL, "scope": Scope.ELL},
    "ps0norm_gt_uunorm": {"stage": Stage.SEMILOCAL, "scope": Scope.ELL},
    "optimize_eigenvalue_error": {"stage": Stage.SEMILOCAL, "scope": Scope.ELL},
    "optimize_not_converged": {"stage": Stage.SEMILOCAL, "scope": Scope.ELL},
    "sbf_overlap_eigenvalue_error": {"stage": Stage.SEMILOCAL, "scope": Scope.ELL},
    "sbf_negative_overlap_eigenvalue": {"stage": Stage.SEMILOCAL, "scope": Scope.ELL},
    "sbf_rc_der_bad_argument": {"stage": Stage.SEMILOCAL, "scope": Scope.ELL},
    "const_basis_dgesvd_failed": {"stage": Stage.SEMILOCAL, "scope": Scope.ELL},
    "polysmooth_bad_ncon": {"stage": Stage.SEMILOCAL, "scope": Scope.ELL},
    # --- errors: VKB projectors (run_vkb) or diagnostics (run_ghosts), depending on position
    "matrix_eigenvalue_error": {"stage": None, "scope": None},
    "sovl_matrix_error": {"stage": Stage.VKB, "scope": Scope.ELL},
    # --- errors: valence states in the separable potential
    "vkb_bound_state_not_converged": {
        "stage": Stage.VKB_SOLVE,
        "scope": Scope.ELL_AND_LOCAL,
    },
    "vkb_bound_state_not_converged_r": {
        "stage": Stage.VKB_SOLVE,
        "scope": Scope.ELL_AND_LOCAL,
    },
    # --- errors: model core charge
    "no_core_valence_crossing": {"stage": Stage.MODEL_CORE, "scope": Scope.MODEL_CORE},
    "modcore_dgesv_failed": {"stage": Stage.MODEL_CORE, "scope": Scope.MODEL_CORE},
    "modcore_budget_no_model": {"stage": Stage.MODEL_CORE, "scope": Scope.MODEL_CORE},
    "modcore_budget_no_core_charge": {
        "stage": Stage.MODEL_CORE,
        "scope": Scope.MODEL_CORE,
    },
    "modcore_budget_waves_unconverged": {
        "stage": Stage.MODEL_CORE,
        "scope": Scope.MODEL_CORE,
    },
    "modcore_budget_no_waves": {"stage": Stage.MODEL_CORE, "scope": Scope.MODEL_CORE},
    "modcore_budget_table_unreached": {
        "stage": Stage.MODEL_CORE,
        "scope": Scope.MODEL_CORE,
    },
    "modcore_budget_ecut_too_large": {
        "stage": Stage.MODEL_CORE,
        "scope": Scope.MODEL_CORE,
    },
    # --- errors: diagnostics
    "ghost_lschpsbar_error": {"stage": Stage.DIAGNOSTICS, "scope": Scope.ELL_AND_LOCAL},
    # --- errors: test configurations (ONCVPSP 4 continues after these; fatal only if incomplete)
    "lschvkbb_not_converged": {"stage": Stage.TEST_CONFIG, "scope": Scope.TEST_CONFIG},
    "psatom_bad_argument": {"stage": Stage.TEST_CONFIG, "scope": Scope.TEST_CONFIG},
    # --- errors: log derivatives and plotting
    "rxpsh_too_small": {"stage": Stage.LOGDER_PLOT, "scope": Scope.ELL},
    "vkbout_dgesv_failed": {"stage": Stage.LOGDER_PLOT, "scope": Scope.ELL},
    # --- errors: pseudopotential file output
    "upf_unsupported_functional": {"stage": Stage.OUTPUT, "scope": Scope.GLOBAL},
    "psml_not_supported": {"stage": Stage.OUTPUT, "scope": Scope.GLOBAL},
    # --- errors: azadoks/oncvpsp
    "local_channel_ae_node": {"stage": Stage.VKB, "scope": Scope.ELL_AND_LOCAL},
    "local_channel_ps_node": {"stage": Stage.VKB, "scope": Scope.ELL_AND_LOCAL},
    "ode_rc_der_mismatch": {"stage": Stage.SEMILOCAL, "scope": Scope.ELL},
    "optimize_q_grid_too_long": {"stage": Stage.SEMILOCAL, "scope": Scope.ELL},
    "mesh_tail_too_short": {"stage": Stage.SEMILOCAL, "scope": Scope.ELL},
    "free_tail_qcut_too_low": {"stage": Stage.SEMILOCAL, "scope": Scope.ELL},
    "output_mesh_error": {"stage": Stage.OUTPUT, "scope": Scope.GLOBAL},
    "output_mesh_beyond_log_mesh": {"stage": Stage.OUTPUT, "scope": Scope.INPUT},
    "vploc_bad_lpopt": {"stage": Stage.INPUT, "scope": Scope.LOCAL},
    "input_teter_grid_invalid": {"stage": Stage.INPUT, "scope": Scope.MODEL_CORE},
    "input_rxpsh_inside_rc": {"stage": Stage.INPUT, "scope": Scope.INPUT},
    "rxpsh_beyond_mesh": {"stage": Stage.LOGDER_PLOT, "scope": Scope.INPUT},
    "functionals_bad_nspin": {"stage": None, "scope": Scope.GLOBAL, "l_rule": "none"},
    # --- errors: anywhere
    "dpnint_interpolation_error": {"stage": None, "scope": None},
    # Called by the wave solvers, in any stage
    "match_point_failed": {"stage": None, "scope": None},
    "outwf_setup_error": {"stage": None, "scope": None},
    "unrecognized_error": {"stage": None, "scope": None},
    "fortran_runtime_error": {"stage": None, "scope": None},
}

_DEFINITIONS: dict[tuple[Level, str], dict] = {
    **{(Level.WARNING, d["name"]): d for d in WARNINGS},
    **{(Level.ERROR, d["name"]): d for d in ERRORS},
    (Level.ERROR, UNRECOGNIZED_ERROR["name"]): UNRECOGNIZED_ERROR,
}
_PATTERNS: dict[tuple[Level, str], re.Pattern] = {
    key: re.compile(d["pattern"]) for key, d in _DEFINITIONS.items()
}


def attribution_for(level: Level, name: str) -> dict[str, Any]:
    """Attribution metadata for a message definition."""
    return ATTRIBUTION.get(f"{level.value}:{name}") or ATTRIBUTION.get(name) or {}


# --- Message attribution ---------------------------------------------------------------------------


def _int_group(groups: Mapping[str, str], key: str) -> int | None:
    value = groups.get(key)
    return int(value) if value is not None else None


def _debl_candidates(
    input_data: Mapping[str, Any] | None, layout: "ProjectorLayout | None"
) -> list[int]:
    """Channels for which ONCVPSP's `debl <= 0 with 2+ scattering states` check fails: a projector
    state energy is set by `debl`, and `debl <= 0`."""
    if not input_data or layout is None:
        return []
    try:
        debl = input_data["vkb_projectors"]["debl"]
        lmax = input_data["pseudopotentials"]["lmax"]
    except (KeyError, TypeError):
        return []
    return [
        l
        for l in range(lmax + 1)  # noqa: E741
        if debl[l] <= 0.0 and layout.uses_debl(l)
    ]


def attribute_message(
    message: Mapping[str, Any],
    level: Level,
    lines: list[str],
    contexts: list[LineContext],
    input_data: Mapping[str, Any] | None = None,
    layout: "ProjectorLayout | None" = None,
) -> OncvpspMessage:
    """Attribute a message dictionary (as produced by `OncvpspTextParser._parse_errors`)."""
    name = message["name"]
    meta = attribution_for(level, name)
    first = message["line_numbers"][0] - 1
    line = lines[first] if 0 <= first < len(lines) else ""
    ctx = (
        contexts[first]
        if 0 <= first < len(contexts)
        else (contexts[-1] if contexts else LineContext())
    )
    pattern = _PATTERNS.get((level, name))
    groups: dict[str, str] = {}
    if pattern and (m := pattern.search(line)):
        groups = {k: v for k, v in m.groupdict().items() if v is not None}

    # Stage
    stage: Stage | None = meta.get("stage")
    if stage is None:
        stage = ctx.stage
        if stage == Stage.INPUT:
            # Messages with explicit INPUT stage are all listed; anything else printed before the
            # reference-atom header comes from the all-electron reference calculation
            stage = Stage.AE_REFERENCE
        if name == "lschfb_not_converged" or name == "ldiracfb_not_converged":
            sub = groups.get("subroutine", "")
            if sub.startswith(("sratom", "relatom")) and stage != Stage.TEST_CONFIG:
                stage = Stage.AE_REFERENCE
    scope: Scope | None = meta.get("scope")
    if scope is None:
        scope = DEFAULT_SCOPE[stage]

    # Angular momentum
    l_rule = meta.get("l_rule", "context")
    l: int | None = None  # noqa: E741
    attribution = Attribution.NONE
    if "l" in groups:
        l = int(groups["l"])  # noqa: E741
        attribution = Attribution.EXPLICIT
    elif (
        l_rule == "pending"
        and ctx.stage == Stage.SEMILOCAL
        and stage == Stage.SEMILOCAL
    ):
        l = ctx.pending_l  # noqa: E741
        attribution = Attribution.INFERRED
    elif (
        l_rule != "none"
        and scope in (Scope.ELL, Scope.ELL_AND_LOCAL)
        and stage == ctx.stage
    ):
        if ctx.current_l is not None:
            l = ctx.current_l  # noqa: E741
            attribution = Attribution.INFERRED
    if name == "debl_not_allowed":
        candidates = _debl_candidates(input_data, layout)
        if candidates and l not in candidates:
            l = candidates[0] if len(candidates) == 1 else l  # noqa: E741
            attribution = Attribution.INFERRED if l is not None else Attribution.NONE

    iproj = _int_group(groups, "iproj")
    if iproj is None:
        iproj = meta.get("iproj")
    # A message printed before the header of its channel is not about the last header's projector
    if (
        iproj is None
        and l_rule != "pending"
        and stage == Stage.SEMILOCAL
        and ctx.stage == Stage.SEMILOCAL
        and l == ctx.header_l
    ):
        iproj = ctx.iproj
    n = _int_group(groups, "n")
    config_index = _int_group(groups, "config")
    if (
        config_index is None
        and stage == Stage.TEST_CONFIG
        and ctx.stage == Stage.TEST_CONFIG
    ):
        config_index = ctx.config_index

    # Degrade the scope when the channel could not be determined
    if l is None:
        if scope == Scope.ELL:
            scope = Scope.GLOBAL
        elif scope == Scope.ELL_AND_LOCAL:
            scope = Scope.LOCAL

    return OncvpspMessage(
        name=name,
        level=level,
        subroutine=message.get("subroutine", ""),
        description=message.get("description", ""),
        line_numbers=list(message["line_numbers"]),
        context_lines=list(message.get("context_lines", [])),
        context_line_numbers=list(message.get("context_line_numbers", [])),
        stage=stage,
        scope=scope,
        attribution=attribution,
        l=l,
        iproj=iproj,
        n=n,
        config_index=config_index,
        implicated_parameters=implicated_parameters(
            stage, scope, l, iproj, config_index, input_data, layout
        ),
        groups=groups,
    )


def _safe_input(parser: "OncvpspTextParser") -> dict | None:
    try:
        return parser.input or None
    except Exception:  # pylint: disable=broad-except
        return None


def _safe_layout(parser: "OncvpspTextParser") -> "ProjectorLayout | None":
    try:
        return parser.projector_layout
    except Exception:  # pylint: disable=broad-except
        return None


def attribute_messages(parser: "OncvpspTextParser") -> list[OncvpspMessage]:
    """All warnings and errors of a parsed output, attributed and sorted by line."""
    lines = parser.raw_lines
    contexts = parser.line_contexts
    input_data = _safe_input(parser)
    layout = _safe_layout(parser)
    messages = [
        attribute_message(m, Level.WARNING, lines, contexts, input_data, layout)
        for m in parser.warnings
    ] + [
        attribute_message(m, Level.ERROR, lines, contexts, input_data, layout)
        for m in parser.errors
    ]
    return sorted(messages, key=lambda m: m.line_numbers[0])


def _last_unrecognized_error(parser: "OncvpspTextParser") -> OncvpspMessage | None:
    """The last `<routine>: ERROR ...` line which is not part of a recognized message, as an
    `unrecognized_error` attributed by its position (`None` if there is none)."""
    lines = parser.raw_lines
    known = {i for m in parser.warnings + parser.errors for i in m["line_numbers"]}
    pattern = _PATTERNS[(Level.ERROR, UNRECOGNIZED_ERROR["name"])]
    for i in range(len(lines) - 1, -1, -1):
        if i + 1 in known or "ERROR" not in lines[i]:
            continue
        if match := pattern.search(lines[i]):
            start, stop = max(0, i - 3), min(len(lines), i + 4)
            message = {
                "name": UNRECOGNIZED_ERROR["name"],
                "line_numbers": [i + 1],
                "context_lines": lines[start:stop],
                "context_line_numbers": list(range(start + 1, stop + 1)),
                "subroutine": match["subroutine"],
                "description": f"{UNRECOGNIZED_ERROR['description']}: {lines[i].strip()}",
            }
            return attribute_message(
                message,
                Level.ERROR,
                lines,
                parser.line_contexts,
                _safe_input(parser),
                _safe_layout(parser),
            )
    return None


def analyze_failure(parser: "OncvpspTextParser") -> OncvpspFailure | None:
    """Why the run failed, or `None` if ONCVPSP wrote all of its output.

    A run which stops without a recognized error, but after a line of the form
    `<routine>: ERROR ...`, fails with that line as an `unrecognized_error`.

    Error messages in a completed run (e.g. a skipped test configuration, or `rxpsh < rc`) are not
    failures of the pseudization; they remain available via `parser.errors` / `parser.messages`.
    """
    if parser.completed:
        return None
    lines = parser.raw_lines
    contexts = parser.line_contexts
    last = contexts[-1] if contexts else LineContext()
    errors = [m for m in attribute_messages(parser) if m.level == Level.ERROR]
    if not errors and (unrecognized := _last_unrecognized_error(parser)) is not None:
        errors = [unrecognized]
    if errors:
        primary = errors[0]
    else:
        stage = last.stage
        scope = DEFAULT_SCOPE[stage]
        l = last.current_l  # noqa: E741
        if stage == Stage.SEMILOCAL and l is None:
            l = last.pending_l  # noqa: E741
        if l is None:
            scope = (
                Scope.GLOBAL
                if scope == Scope.ELL
                else (Scope.LOCAL if scope == Scope.ELL_AND_LOCAL else scope)
            )
        primary = OncvpspMessage(
            name="incomplete_output",
            level=Level.ERROR,
            subroutine="main",
            description="Output ends without any recognized error message (program killed, crashed, "
            "or still running). Missing or truncated plotting data: "
            + ", ".join(parser.missing_plot_data),
            line_numbers=[len(lines)],
            context_lines=lines[-5:],
            context_line_numbers=list(range(max(1, len(lines) - 4), len(lines) + 1)),
            stage=stage,
            scope=scope,
            attribution=Attribution.INFERRED if l is not None else Attribution.NONE,
            l=l,
            iproj=last.iproj
            if stage == Stage.SEMILOCAL and l == last.header_l
            else None,
            config_index=last.config_index if stage == Stage.TEST_CONFIG else None,
            implicated_parameters=implicated_parameters(
                stage,
                scope,
                l,
                None,
                last.config_index if stage == Stage.TEST_CONFIG else None,
                _safe_input(parser),
                _safe_layout(parser),
            ),
        )
    return OncvpspFailure(
        primary=primary,
        errors=errors,
        last_stage_reached=last.stage,
        last_l_reached=last.last_l,
        completed=False,
    )


__all__ = [
    "ATTRIBUTION",
    "analyze_failure",
    "attribute_message",
    "attribute_messages",
    "attribution_for",
]
