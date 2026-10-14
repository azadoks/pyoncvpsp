"""Structured ONCVPSP messages with stage, scope, and parameter attribution.

ONCVPSP stops with exit status 0 on every error, so failures must be read from stdout. Stdout is
stage-ordered, which lets a message be attributed to the stage of the program that produced it, and
often to the angular-momentum channel (and thus the input parameters) that caused it.
"""

from dataclasses import dataclass, field, asdict
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Mapping

from pyoncvpsp.channels import ChannelKey

if TYPE_CHECKING:
    from pyoncvpsp.projectors import ProjectorLayout


class Stage(StrEnum):
    """Stage of the ONCVPSP program, in execution order."""

    INPUT = "input"  # reading and checking input (`check_data`)
    AE_REFERENCE = "ae_reference"  # all-electron reference atom (`sratom`/`relatom`)
    SEMILOCAL = (
        "semilocal"  # wellstates, optimized pseudo wavefunctions, semilocal potentials
    )
    VKB = "vkb"  # Vanderbilt-Kleinman-Bylander projector construction (`run_vkb`)
    VKB_SOLVE = (
        "vkb_solve"  # valence states solved in the separable (VKB + local) potential
    )
    MODEL_CORE = "model_core"  # model core charge (`modcore*`)
    DIAGNOSTICS = "diagnostics"  # `run_diag`, `run_ghosts`
    TEST_CONFIG = "test_config"  # `run_config`
    LOGDER_PLOT = "logder_plot"  # `run_plot`, `run_phsft`, gnuplot script
    OUTPUT = "output"  # pseudopotential file output (`upfout`, `psp8out`, ...)
    UNKNOWN = "unknown"

    @property
    def order(self) -> int:
        """Position of the stage in the execution order (`UNKNOWN` sorts last)."""
        return STAGE_ORDER.index(self)


STAGE_ORDER: list[Stage] = [
    Stage.INPUT,
    Stage.AE_REFERENCE,
    Stage.SEMILOCAL,
    Stage.VKB,
    Stage.VKB_SOLVE,
    Stage.MODEL_CORE,
    Stage.DIAGNOSTICS,
    Stage.TEST_CONFIG,
    Stage.LOGDER_PLOT,
    Stage.OUTPUT,
    Stage.UNKNOWN,
]


class Scope(StrEnum):
    """Which input parameters a message implicates."""

    GLOBAL = "global"  # cannot be narrowed down
    ELL = "ell"  # parameters of one angular-momentum channel
    ELL_AND_LOCAL = "ell_and_local"  # one channel plus the local potential
    LOCAL = "local"  # local potential only
    MODEL_CORE = "model_core"
    TEST_CONFIG = "test_config"
    INPUT = "input"  # malformed / unreadable input, independent of the physics


class Attribution(StrEnum):
    """How the angular momentum `l` of a message was determined."""

    EXPLICIT = "explicit"  # ONCVPSP printed it
    INFERRED = "inferred"  # from the position in the output and/or the input parameters
    NONE = "none"


class Level(StrEnum):
    WARNING = "warning"
    ERROR = "error"


# Default scope for messages whose definition does not specify one
DEFAULT_SCOPE: dict[Stage, Scope] = {
    Stage.INPUT: Scope.INPUT,
    Stage.AE_REFERENCE: Scope.GLOBAL,
    Stage.SEMILOCAL: Scope.ELL,
    Stage.VKB: Scope.ELL,
    Stage.VKB_SOLVE: Scope.ELL_AND_LOCAL,
    Stage.MODEL_CORE: Scope.MODEL_CORE,
    Stage.DIAGNOSTICS: Scope.ELL_AND_LOCAL,
    Stage.TEST_CONFIG: Scope.TEST_CONFIG,
    Stage.LOGDER_PLOT: Scope.ELL,
    Stage.OUTPUT: Scope.GLOBAL,
    Stage.UNKNOWN: Scope.GLOBAL,
}

# Flattened-key naming convention shared with `OncvpspInput.as_dict(flat=True)` and oncvpsp-opt features
SEP = "__"


def _key(*parts: Any) -> str:
    return SEP.join(str(p) for p in parts)


def ell_parameters(
    l: int,  # noqa: E741
    input_data: Mapping[str, Any] | None = None,
    iproj: int | None = None,
    include_vkb: bool = True,
    layout: "ProjectorLayout | None" = None,
) -> list[str]:
    """Flattened input keys of the parameters that control channel `l`.

    `ep` and `debl` are included only when the projector `layout` uses them to set a projector
    state energy (of projector `iproj`, if given). Without a layout, both are included.
    """
    keys = [
        _key("pseudopotentials", "rc", l),
        _key("pseudopotentials", "qcut", l),
        _key("pseudopotentials", "ncon", l),
        _key("pseudopotentials", "nbas", l),
    ]
    ep_active = debl_active = True
    if layout is not None and layout.l_channel(l):
        ep_active = layout.uses_ep(l)
        debl_active = layout.uses_debl(l, iproj)
    nproj = None
    try:
        nproj = (input_data or {})["vkb_projectors"]["nproj"][l]
    except (KeyError, IndexError, TypeError):
        pass
    if ep_active:
        keys.append(_key("pseudopotentials", "ep", l))
    if include_vkb:
        if nproj is None or nproj >= 1:
            keys.append(_key("vkb_projectors", "nproj", l))
        if debl_active:
            keys.append(_key("vkb_projectors", "debl", l))
    return keys


def local_parameters(
    input_data: Mapping[str, Any] | None = None,
    layout: "ProjectorLayout | None" = None,
) -> list[str]:
    """Flattened input keys of the local-potential parameters (including the semilocal channel
    used as local potential when `lloc != 4`)."""
    keys = [_key("local_potential", "lloc"), _key("local_potential", "lpopt")]
    lloc = None
    if input_data:
        lloc = (input_data.get("local_potential") or {}).get("lloc")
    if lloc is None or lloc == 4:
        keys += [_key("local_potential", "rcloc"), _key("local_potential", "dvloc0")]
    if lloc is not None and lloc != 4:
        keys += ell_parameters(lloc, input_data, include_vkb=False, layout=layout)
    return keys


def model_core_parameters() -> list[str]:
    return [
        _key("model_core_charge", "icmod"),
        _key("model_core_charge", "fcfact"),
        _key("model_core_charge", "rcfact"),
    ]


def implicated_parameters(
    stage: Stage,
    scope: Scope,
    l: int | None = None,  # noqa: E741
    iproj: int | None = None,
    config_index: int | None = None,
    input_data: Mapping[str, Any] | None = None,
    layout: "ProjectorLayout | None" = None,
) -> list[str]:
    """Flattened input keys implicated by a message with the given attribution."""
    keys: list[str] = []
    if scope in (Scope.ELL, Scope.ELL_AND_LOCAL) and l is not None:
        keys += ell_parameters(
            l, input_data, iproj, include_vkb=stage != Stage.LOGDER_PLOT, layout=layout
        )
        if stage == Stage.LOGDER_PLOT:
            keys.append(_key("log_derivative_analysis", "rxpsh"))
    if scope in (Scope.ELL_AND_LOCAL, Scope.LOCAL):
        keys += local_parameters(input_data, layout)
    if scope == Scope.MODEL_CORE:
        keys += model_core_parameters()
    # ONCVPSP numbers test configurations from 0 = the reference configuration, so configuration
    # `k >= 1` is `test_configurations[k - 1]` of the input
    if scope == Scope.TEST_CONFIG and config_index is not None and config_index >= 1:
        keys.append(_key("test_configurations", config_index - 1))
    # de-duplicate, preserving order
    return list(dict.fromkeys(keys))


@dataclass
class OncvpspMessage:
    """A warning or error reported by ONCVPSP, with its source attribution."""

    name: str
    level: Level
    subroutine: str
    description: str
    line_numbers: list[int]
    context_lines: list[str]
    context_line_numbers: list[int]
    stage: Stage = Stage.UNKNOWN
    scope: Scope = Scope.GLOBAL
    attribution: Attribution = Attribution.NONE
    l: int | None = None  # noqa: E741
    iproj: int | None = None
    n: int | None = None
    j: float | None = None  # total angular momentum, if attributed to a j-channel
    config_index: int | None = (
        None  # as numbered by ONCVPSP: 0 = reference, k = test_configurations[k - 1]
    )
    implicated_parameters: list[str] = field(default_factory=list)
    groups: dict[str, str] = field(default_factory=dict)

    @property
    def channel(self) -> ChannelKey | None:
        """l-channel or j-channel the message is attributed to, if any."""
        return (
            None
            if self.l is None
            else ChannelKey.from_record({"l": self.l, "j": self.j})
        )

    @property
    def message_lines(self) -> list[str]:
        """Lines which make up the message itself."""
        return [
            line
            for ln, line in zip(self.context_line_numbers, self.context_lines)
            if ln in self.line_numbers
        ]

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable dictionary (enums as their string values)."""
        d = asdict(self)
        for key in ("level", "stage", "scope", "attribution"):
            d[key] = str(d[key].value) if d[key] is not None else None
        return d

    def as_legacy_dict(self) -> dict[str, Any]:
        """Dictionary with the keys used by `OncvpspTextParser.errors`/`.warnings` before attribution
        existed, plus the attribution fields."""
        d = self.as_dict()
        d.pop("level")
        return d


@dataclass
class OncvpspFailure:
    """Why an ONCVPSP run did not produce a pseudopotential.

    `primary` is the first fatal message in the output (ONCVPSP stops at the first error, so later
    errors are secondary). Runs which stop without any recognized error message have a synthetic
    `incomplete_output` primary attributed to the last stage reached.
    """

    primary: OncvpspMessage
    errors: list[OncvpspMessage]
    last_stage_reached: Stage
    last_l_reached: int | None
    completed: bool = False

    @property
    def name(self) -> str:
        return self.primary.name

    @property
    def stage(self) -> Stage:
        return self.primary.stage

    @property
    def scope(self) -> Scope:
        return self.primary.scope

    @property
    def attribution(self) -> Attribution:
        return self.primary.attribution

    @property
    def l(self) -> int | None:  # noqa: E741,E743
        return self.primary.l

    @property
    def implicated_parameters(self) -> list[str]:
        return self.primary.implicated_parameters

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable dictionary."""
        return {
            "name": self.name,
            "stage": str(self.stage.value),
            "scope": str(self.scope.value),
            "attribution": str(self.attribution.value),
            "l": self.l,
            "iproj": self.primary.iproj,
            "config_index": self.primary.config_index,
            "implicated_parameters": list(self.implicated_parameters),
            "last_stage_reached": str(self.last_stage_reached.value),
            "last_l_reached": self.last_l_reached,
            "completed": self.completed,
            "primary": self.primary.as_dict(),
            "errors": [e.as_dict() for e in self.errors],
        }


__all__ = [
    "Attribution",
    "DEFAULT_SCOPE",
    "Level",
    "OncvpspFailure",
    "OncvpspMessage",
    "STAGE_ORDER",
    "Scope",
    "Stage",
    "ell_parameters",
    "implicated_parameters",
    "local_parameters",
    "model_core_parameters",
]
