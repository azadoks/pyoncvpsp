"""Locate the program stages and the blocks within them in ONCVPSP's stdout.

ONCVPSP's stdout is stage-ordered, and within the semilocal stage it is ordered by angular momentum
`l`. A state machine over the stage and block headers gives every line a context: the stage, the
channel being processed, and the enclosing block (e.g. one optimized projector, one test
configuration). The context is used to attribute messages and to bound the search for each quantity.

The main program's loop over `l` calls `wellstate`, the `debl` check and `qroots` *before* printing
the per-`l` header `Calculating optimized projector`, so `LineContext.pending_l` gives the channel
being processed before its header appears. Fully-relativistic ONCVPSP v3 repeats that loop for each
j-channel of `l`, and its `cnorm` lines (printed while computing the projector states) announce `l`.
"""

from dataclasses import dataclass, replace
import re
from typing import Iterator

from pyoncvpsp.channels import ChannelKey

from ._messages import Stage


class BlockKind:
    """Kinds of `Block`."""

    PROJECTOR = "projector"  # one optimized projector (`optimize`), semilocal stage
    VKB = "vkb"  # B matrix and projector coefficients of one channel (`run_vkb`)
    DIAG_SEMILOCAL = (
        "diag_semilocal"  # diagnostic tests in the semilocal potentials (`run_diag`)
    )
    DIAG_VKB = "diag_vkb"  # diagnostic tests in the VKB potentials (`run_diag`)
    GHOSTS = "ghosts"  # ghost tests (`run_ghosts`)
    TEST_CONFIG = "test_config"  # one test configuration (`run_config`)


@dataclass(frozen=True)
class Block:
    """Block of output opened by a header line, which ends where the next block or stage begins."""

    kind: str
    line: int  # index of the header line in the raw lines
    l: int | None = None  # noqa: E741
    iproj: int | None = None
    channel: ChannelKey | None = (
        None  # VKB blocks: the l-channel or j-channel (from `kap`)
    )
    config_index: int | None = None


@dataclass(frozen=True)
class LineContext:
    """Program state at a line of the output."""

    stage: Stage = Stage.INPUT
    header_l: int | None = (
        None  # l of the last `Calculating optimized projector` header (semilocal)
    )
    announced_l: int | None = (
        None  # l of the last `Wellstate for l =` or v3 `cnorm` line since the last header
    )
    iproj: int | None = None  # projector of the last header (semilocal)
    marker_l: int | None = (
        None  # l of the last per-l marker in the VKB/diagnostics/plot stages
    )
    config_index: int | None = (
        None  # test configuration as numbered by ONCVPSP (0 = reference configuration)
    )
    block: Block | None = None  # enclosing block, if any

    @property
    def pending_l(self) -> int:
        """Channel currently being processed by the main program *before* its header is printed."""
        if self.announced_l is not None:
            return self.announced_l
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
            x for x in (self.header_l, self.announced_l, self.marker_l) if x is not None
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
    "cnorm": re.compile(r"^\s+cnorm=\s*\S+\s+l{1,2}=\s*(?P<l>\d+)"),
    "header_v4": re.compile(r"^Calculating optimized projector #\s*(?P<iproj>\d+)"),
    "header_v4_l": re.compile(r"^\s*for l=\s*(?P<l>\d+)"),
    "header_v3": re.compile(
        r"^Calculating (?P<which>first|second) optimized projector for l=\s*(?P<l>\d+)"
    ),
    "vkb": re.compile(r"^Construct Vanderbilt / Klein+m+an+-Bylander projectors"),
    "b_matrix": re.compile(
        r"^B matrix(?: Hermiticity error)?, ll=\s*(?P<l>\d+),?(?:\s+kap=\s*(?P<kappa>-?\d+))?"
    ),
    "model_core": re.compile(
        r"^(?:Model core correction analysis|Teter function model core charge|Polynomial model core charge|Model core charge)"
    ),
    "pseudoatom": re.compile(r"^Pseudoatom total energy"),
    "diag_semilocal": re.compile(r"^Diagnostic tests using semi-local"),
    "diag_vkb": re.compile(r"^Diagnostic tests using Vanderbilt-Kleinman-Bylander"),
    "diag_row": re.compile(
        r"^\s{2,}(?P<l>\d)\s+\d+\.\d{4,}\s+\d+\.\d{4,}\s+-?\d+\.\d{4,}"
    ),
    "ghosts": re.compile(r"^\s*Testing for"),
    "test_config": re.compile(r"^Test configuration\s*(?P<index>\d+)"),
    "plot": re.compile(r"^DATA FOR PLOTTING"),
    "plot_wf": re.compile(r"^n=\s*\d+,\s+l=\s*(?P<l>\d+)"),
    "plot_scat": re.compile(r"^scattering, iprj=\s*(?P<iproj>\d+),\s+l=\s*(?P<l>\d+)"),
    "plot_logder": re.compile(r"^log derivativ+e data for plotting, l=\s*(?P<l>\d+)"),
    "output": re.compile(
        r"^(?:Begin PSP_UPF|Begin PSPCODE8|Begin PSP8|Begin PSML|<UPF |<PSML|<psml)"
    ),
}

# Any of the markers above (named groups dropped, as they repeat): a line it does not match changes
# no context, so the markers need not be tried one by one on it
_RE_ANY = re.compile(
    "|".join(
        "(?:" + re.sub(r"\(\?P<\w+>", "(?:", pattern.pattern) + ")"
        for pattern in _RE.values()
    )
)


def line_contexts(lines: list[str]) -> list[LineContext]:
    """Context of every line of the output (the state *after* processing the line's own markers)."""
    ctx = LineContext()
    contexts: list[LineContext] = []
    seen_test_config_echo = False
    for i, line in enumerate(lines):
        if not _RE_ANY.match(line):
            contexts.append(ctx)
            continue
        if ctx.stage == Stage.INPUT:
            if _RE["test_config_echo"].match(line):
                seen_test_config_echo = True
            elif seen_test_config_echo and _RE["echo_nlf"].match(line):
                ctx = replace(ctx, stage=Stage.AE_REFERENCE)
        if _RE["ae_reference"].match(line):
            ctx = replace(ctx, stage=Stage.AE_REFERENCE, block=None)
        elif _RE["semilocal"].match(line):
            ctx = replace(ctx, stage=Stage.SEMILOCAL, block=None)
        elif ctx.stage in (Stage.AE_REFERENCE, Stage.SEMILOCAL) and (
            m := _RE["wellstate"].match(line)
        ):
            ctx = replace(
                ctx, stage=Stage.SEMILOCAL, announced_l=int(m["l"]), block=None
            )
        elif ctx.stage == Stage.SEMILOCAL and (m := _RE["cnorm"].match(line)):
            ctx = replace(ctx, announced_l=int(m["l"]), block=None)
        elif m := _RE["header_v4"].match(line):
            # The format `(/a,i4)` reverts, so ` for l=` lands on its own line after a blank one
            l = None  # noqa: E741
            for j in range(i + 1, min(i + 4, len(lines))):
                if ml := _RE["header_v4_l"].match(lines[j]):
                    l = int(ml["l"])  # noqa: E741
                    break
            iproj = int(m["iproj"])
            ctx = replace(
                ctx,
                stage=Stage.SEMILOCAL,
                header_l=l,
                iproj=iproj,
                announced_l=None,
                block=Block(BlockKind.PROJECTOR, i, l=l, iproj=iproj),
            )
        elif m := _RE["header_v3"].match(line):
            l = int(m["l"])  # noqa: E741
            iproj = 1 if m["which"] == "first" else 2
            ctx = replace(
                ctx,
                stage=Stage.SEMILOCAL,
                header_l=l,
                iproj=iproj,
                announced_l=None,
                block=Block(BlockKind.PROJECTOR, i, l=l, iproj=iproj),
            )
        elif _RE["vkb"].match(line):
            ctx = replace(ctx, stage=Stage.VKB, marker_l=None, iproj=None, block=None)
        elif ctx.stage == Stage.VKB and (m := _RE["b_matrix"].match(line)):
            l = int(m["l"])  # noqa: E741
            kappa = int(m["kappa"]) if m["kappa"] is not None else None
            channel = ChannelKey.from_printed_kappa(l, kappa)
            ctx = replace(
                ctx, marker_l=l, block=Block(BlockKind.VKB, i, l=l, channel=channel)
            )
        elif ctx.stage in (Stage.VKB, Stage.VKB_SOLVE) and (
            _RE["model_core"].match(line) or _RE["pseudoatom"].match(line)
        ):
            ctx = replace(ctx, stage=Stage.MODEL_CORE, marker_l=None, block=None)
        elif _RE["diag_semilocal"].match(line):
            ctx = replace(
                ctx,
                stage=Stage.DIAGNOSTICS,
                marker_l=None,
                block=Block(BlockKind.DIAG_SEMILOCAL, i),
            )
        elif _RE["diag_vkb"].match(line):
            ctx = replace(
                ctx,
                stage=Stage.DIAGNOSTICS,
                marker_l=None,
                block=Block(BlockKind.DIAG_VKB, i),
            )
        elif ctx.stage == Stage.DIAGNOSTICS and _RE["ghosts"].match(line):
            ctx = replace(ctx, block=Block(BlockKind.GHOSTS, i))
        elif ctx.stage == Stage.DIAGNOSTICS and (m := _RE["diag_row"].match(line)):
            ctx = replace(ctx, marker_l=int(m["l"]))
        elif m := _RE["test_config"].match(line):
            index = int(m["index"])
            ctx = replace(
                ctx,
                stage=Stage.TEST_CONFIG,
                config_index=index,
                marker_l=None,
                block=Block(BlockKind.TEST_CONFIG, i, config_index=index),
            )
        elif _RE["plot"].match(line):
            ctx = replace(
                ctx,
                stage=Stage.LOGDER_PLOT,
                marker_l=None,
                config_index=None,
                block=None,
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
            ctx = replace(ctx, stage=Stage.OUTPUT, marker_l=None, block=None)
        contexts.append(ctx)
    return contexts


def blocks(contexts: list[LineContext], kind: str) -> Iterator[tuple[Block, int, int]]:
    """Blocks of a kind as `(block, start, stop)`, where `contexts[start:stop]` is the block.

    `contexts` may be a subsequence of the contexts of all lines (e.g. with message lines removed),
    in which case `start` and `stop` index into the same subsequence of lines.
    """
    current: Block | None = None
    start = 0
    for i, ctx in enumerate(contexts):
        if ctx.block is current:
            continue
        if current is not None and current.kind == kind:
            yield current, start, i
        current, start = ctx.block, i
    if current is not None and current.kind == kind:
        yield current, start, len(contexts)


__all__ = ["Block", "BlockKind", "LineContext", "blocks", "line_contexts"]
