"""Attribute ONCVPSP messages to the program stage and the input parameters that produced them.

ONCVPSP's stdout is stage-ordered, and within the semilocal stage it is ordered by angular momentum
`l`. A state machine over the stage and per-`l` headers gives every line a context, which is used to
attribute messages that do not print `l` themselves.

The main program's loop over `l` calls `wellstate`, the `debl` check and `qroots` *before* printing
the per-`l` header `Calculating optimized projector`, so their messages appear after the block of the
previous `l` (`l_rule = "pending"`).
"""

from dataclasses import dataclass, replace
import re
from typing import TYPE_CHECKING, Any, Mapping

from ._data import ERRORS, WARNINGS
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

if TYPE_CHECKING:
    from ._text import OncvpspTextParser

# Attribution metadata per message name (keys `"warning:<name>"` / `"error:<name>"` take precedence
# over `"<name>"` where the same name is used for both a warning and an error).
#   stage: Stage | None  -- None: infer from the position in the output
#   scope: Scope | None  -- None: default for the stage
#   l_rule: "pending" | "context" (default) | "none"
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
    # --- errors: anywhere
    "dpnint_interpolation_error": {"stage": None, "scope": None},
    "fortran_runtime_error": {"stage": None, "scope": None},
}

_DEFINITIONS: dict[tuple[Level, str], dict] = {
    **{(Level.WARNING, d["name"]): d for d in WARNINGS},
    **{(Level.ERROR, d["name"]): d for d in ERRORS},
}
_PATTERNS: dict[tuple[Level, str], re.Pattern] = {
    key: re.compile(d["pattern"]) for key, d in _DEFINITIONS.items()
}


def attribution_for(level: Level, name: str) -> dict[str, Any]:
    """Attribution metadata for a message definition."""
    return ATTRIBUTION.get(f"{level.value}:{name}") or ATTRIBUTION.get(name) or {}


# --- Line contexts -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class LineContext:
    """Program state at a line of the output."""

    stage: Stage = Stage.INPUT
    header_l: int | None = (
        None  # l of the last `Calculating optimized projector` header (semilocal)
    )
    well_l: int | None = (
        None  # l of the last `Wellstate for l =` block (printed before the header)
    )
    iproj: int | None = None  # projector of the last header (semilocal)
    marker_l: int | None = (
        None  # l of the last per-l marker in the VKB/diagnostics/plot stages
    )
    config_index: int | None = (
        None  # test configuration as numbered by ONCVPSP (0 = reference configuration)
    )

    @property
    def pending_l(self) -> int:
        """Channel currently being processed by the main program *before* its header is printed."""
        if self.well_l is not None and (
            self.header_l is None or self.well_l > self.header_l
        ):
            return self.well_l
        return self.header_l + 1 if self.header_l is not None else 0

    @property
    def current_l(self) -> int | None:
        """Channel of the enclosing per-l block, if any."""
        if self.stage == Stage.SEMILOCAL:
            return self.header_l
        return self.marker_l

    @property
    def last_l(self) -> int | None:
        """Highest channel for which any per-l output was seen."""
        candidates = [
            x for x in (self.header_l, self.well_l, self.marker_l) if x is not None
        ]
        return max(candidates) if candidates else None


_RE = {
    "test_config_echo": re.compile(r"^# TEST CONFIGURATIONS"),
    "echo_nlf": re.compile(r"^#\s+n\s+l\s+f\s*$"),
    "ae_reference": re.compile(r"^Reference configu[fr]ation results"),
    "semilocal": re.compile(
        r"^Begin loop to\s+construct optimized pseudo wave functions"
    ),
    "wellstate": re.compile(r"^\s*Wellstate for l =\s*(?P<l>\d+)\s+n =\s*(?P<n>\d+)"),
    "header_v4": re.compile(r"^Calculating optimized projector #\s*(?P<iproj>\d+)"),
    "header_v4_l": re.compile(r"^\s*for l=\s*(?P<l>\d+)"),
    "header_v3": re.compile(
        r"^Calculating (?P<which>first|second) optimized projector for l=\s*(?P<l>\d+)"
    ),
    "vkb": re.compile(r"^Construct Vanderbilt / Klein+m+an+-Bylander projectors"),
    "hermiticity": re.compile(r"^B matrix Hermiticity error, ll=\s*(?P<l>\d+)"),
    "model_core": re.compile(
        r"^(?:Model core correction analysis|Teter function model core charge|Polynomial model core charge|Model core charge)"
    ),
    "pseudoatom": re.compile(r"^Pseudoatom total energy"),
    "diagnostics": re.compile(r"^Diagnostic tests using"),
    "diag_row": re.compile(
        r"^\s{2,}(?P<l>\d)\s+\d+\.\d{4,}\s+\d+\.\d{4,}\s+-?\d+\.\d{4,}"
    ),
    "test_config": re.compile(r"^Test configuration\s*(?P<index>\d+)"),
    "plot": re.compile(r"^DATA FOR PLOTTING"),
    "plot_wf": re.compile(r"^n=\s*\d+,\s+l=\s*(?P<l>\d+)"),
    "plot_scat": re.compile(r"^scattering, iprj=\s*(?P<iproj>\d+),\s+l=\s*(?P<l>\d+)"),
    "plot_logder": re.compile(r"^log derivativ+e data for plotting, l=\s*(?P<l>\d+)"),
    "output": re.compile(
        r"^(?:Begin PSP_UPF|Begin PSPCODE8|Begin PSP8|Begin PSML|<UPF |<PSML|<psml)"
    ),
}


def line_contexts(lines: list[str]) -> list[LineContext]:
    """Context of every line of the output (the state *after* processing the line's own markers)."""
    ctx = LineContext()
    contexts: list[LineContext] = []
    seen_test_config_echo = False
    for i, line in enumerate(lines):
        if ctx.stage == Stage.INPUT:
            if _RE["test_config_echo"].match(line):
                seen_test_config_echo = True
            elif seen_test_config_echo and _RE["echo_nlf"].match(line):
                ctx = replace(ctx, stage=Stage.AE_REFERENCE)
        if _RE["ae_reference"].match(line):
            ctx = replace(ctx, stage=Stage.AE_REFERENCE)
        elif _RE["semilocal"].match(line):
            ctx = replace(ctx, stage=Stage.SEMILOCAL)
        elif ctx.stage in (Stage.AE_REFERENCE, Stage.SEMILOCAL) and (
            m := _RE["wellstate"].match(line)
        ):
            ctx = replace(ctx, stage=Stage.SEMILOCAL, well_l=int(m["l"]))
        elif m := _RE["header_v4"].match(line):
            # The format `(/a,i4)` reverts, so ` for l=` lands on its own line after a blank one
            l = None  # noqa: E741
            for j in range(i + 1, min(i + 4, len(lines))):
                if ml := _RE["header_v4_l"].match(lines[j]):
                    l = int(ml["l"])  # noqa: E741
                    break
            ctx = replace(ctx, stage=Stage.SEMILOCAL, header_l=l, iproj=int(m["iproj"]))
        elif m := _RE["header_v3"].match(line):
            ctx = replace(
                ctx,
                stage=Stage.SEMILOCAL,
                header_l=int(m["l"]),
                iproj=1 if m["which"] == "first" else 2,
            )
        elif _RE["vkb"].match(line):
            ctx = replace(ctx, stage=Stage.VKB, marker_l=None, iproj=None)
        elif ctx.stage == Stage.VKB and (m := _RE["hermiticity"].match(line)):
            ctx = replace(ctx, marker_l=int(m["l"]))
        elif ctx.stage in (Stage.VKB, Stage.VKB_SOLVE) and (
            _RE["model_core"].match(line) or _RE["pseudoatom"].match(line)
        ):
            ctx = replace(ctx, stage=Stage.MODEL_CORE, marker_l=None)
        elif _RE["diagnostics"].match(line):
            ctx = replace(ctx, stage=Stage.DIAGNOSTICS, marker_l=None)
        elif ctx.stage == Stage.DIAGNOSTICS and (m := _RE["diag_row"].match(line)):
            ctx = replace(ctx, marker_l=int(m["l"]))
        elif m := _RE["test_config"].match(line):
            ctx = replace(
                ctx,
                stage=Stage.TEST_CONFIG,
                config_index=int(m["index"]),
                marker_l=None,
            )
        elif _RE["plot"].match(line):
            ctx = replace(
                ctx, stage=Stage.LOGDER_PLOT, marker_l=None, config_index=None
            )
        elif ctx.stage == Stage.LOGDER_PLOT and (
            m := (_RE["plot_wf"].match(line) or _RE["plot_logder"].match(line))
        ):
            ctx = replace(ctx, marker_l=int(m["l"]))
        elif ctx.stage == Stage.LOGDER_PLOT and (m := _RE["plot_scat"].match(line)):
            ctx = replace(ctx, marker_l=int(m["l"]))
        elif ctx.stage in (Stage.LOGDER_PLOT, Stage.OUTPUT) and _RE["output"].match(
            line
        ):
            ctx = replace(ctx, stage=Stage.OUTPUT, marker_l=None)
        contexts.append(ctx)
    return contexts


# --- Message attribution ---------------------------------------------------------------------------


def _int_group(groups: Mapping[str, str], key: str) -> int | None:
    value = groups.get(key)
    return int(value) if value is not None else None


def _debl_candidates(input_data: Mapping[str, Any] | None) -> list[int]:
    """Channels for which ONCVPSP's `debl <= 0 with 2+ scattering states` check fails."""
    if not input_data:
        return []
    try:
        from pyoncvpsp.io.input._models import OncvpspInput

        model = OncvpspInput(**input_data)
    except Exception:  # pylint: disable=broad-except
        return []
    candidates = []
    for l in range(model.pseudopotentials.lmax + 1):  # noqa: E741
        nproj = model.vkb_projectors.nproj[l]
        debl = model.vkb_projectors.debl[l]
        if nproj >= 2 and debl <= 0.0 and model.debl_is_active(l):
            candidates.append(l)
    return candidates


def attribute_message(
    message: Mapping[str, Any],
    level: Level,
    lines: list[str],
    contexts: list[LineContext],
    input_data: Mapping[str, Any] | None = None,
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
        candidates = _debl_candidates(input_data)
        if candidates and l not in candidates:
            l = candidates[0] if len(candidates) == 1 else l  # noqa: E741
            attribution = Attribution.INFERRED if l is not None else Attribution.NONE

    iproj = _int_group(groups, "iproj")
    if (
        iproj is None
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
        spin_sign=None,
        config_index=config_index,
        implicated_parameters=implicated_parameters(
            stage, scope, l, iproj, config_index, input_data
        ),
        groups=groups,
    )


def _safe_input(parser: "OncvpspTextParser") -> dict | None:
    try:
        return parser.input or None
    except Exception:  # pylint: disable=broad-except
        return None


def attribute_messages(parser: "OncvpspTextParser") -> list[OncvpspMessage]:
    """All warnings and errors of a parsed output, attributed and sorted by line."""
    lines = parser.raw_lines
    contexts = line_contexts(lines)
    input_data = _safe_input(parser)
    messages = [
        attribute_message(m, Level.WARNING, lines, contexts, input_data)
        for m in parser.warnings
    ] + [
        attribute_message(m, Level.ERROR, lines, contexts, input_data)
        for m in parser.errors
    ]
    return sorted(messages, key=lambda m: m.line_numbers[0])


def analyze_failure(parser: "OncvpspTextParser") -> OncvpspFailure | None:
    """Why the run failed, or `None` if ONCVPSP wrote all of its output.

    Error messages in a completed run (e.g. a skipped test configuration, or `rxpsh < rc`) are not
    failures of the pseudization; they remain available via `parser.errors` / `parser.messages`.
    """
    if parser.completed:
        return None
    lines = parser.raw_lines
    contexts = line_contexts(lines)
    last = contexts[-1] if contexts else LineContext()
    errors = [m for m in attribute_messages(parser) if m.level == Level.ERROR]
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
    "LineContext",
    "analyze_failure",
    "attribute_message",
    "attribute_messages",
    "attribution_for",
    "line_contexts",
]
