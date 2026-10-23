"""Which all-electron state each projector is built from, predicted from the input.

ONCVPSP and METAPSP choose the projector state of every projector from the reference configuration
and the input `ep`, `debl` and `nproj`. The rules differ between program versions (see the
`_v3_slots` and `_v4_slots` adapters, each transcribed from the Fortran source cited in it).
`predict_projector_layout` is the only place that knows them: the input model, the output parser
and message attribution all ask the layout.

The input does not fix everything. Whether a `wellstate` call returns a wellstate or a localized
bound state (and at which energy) is decided at run time, so a slot only says *how* its state is
obtained; the output parser resolves the rest.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterator

from pyoncvpsp.channels import ChannelKey, ProjectorKey, j_channels
from pyoncvpsp.io.input._models import OncvpspInput


class Solver(StrEnum):
    """How the projector state of a slot is obtained."""

    BOUND = "bound"  # bound-state solve at the valence state's eigenvalue (`lschfb`)
    WELLSTATE = "wellstate"  # `wellstate`: a wellstate or a localized bound state
    PARTNER_WELL = (
        "partner_well"  # v4 j = l - 1/2: solved in the well of the j = l + 1/2 slot
    )
    BY_ENERGY = "by_energy"  # v2/v3 second projector: bound solve if its energy < 0, else `wellstate`


class Energy(StrEnum):
    """Where the target energy of a slot comes from."""

    VALENCE = "valence"  # all-electron eigenvalue of the slot's valence state
    EP = "ep"  # input `ep` of the l-channel
    DEBL = "debl"  # energy of the previous projector state of the channel plus input `debl`


@dataclass(frozen=True)
class ProjectorSlot:
    """The predicted projector state of one projector."""

    projector: ProjectorKey
    n: int  # principal quantum number requested (a wellstate may end up with more nodes)
    solver: Solver
    energy: Energy
    valence_index: int | None = (
        None  # reference-configuration index of the valence state used
    )

    @property
    def is_valence(self) -> bool:
        """Whether the projector state is a valence state solved as a bound state."""
        return self.solver == Solver.BOUND and self.energy == Energy.VALENCE


@dataclass(frozen=True)
class ProjectorLayout:
    """Predicted projector slots of every channel, in program order. The local channel
    (`lloc <= lmax`) has one slot, used for the semilocal potential only."""

    slots: tuple[ProjectorSlot, ...]

    def __iter__(self) -> Iterator[ProjectorSlot]:
        return iter(self.slots)

    def __len__(self) -> int:
        return len(self.slots)

    def get(self, projector: ProjectorKey) -> ProjectorSlot | None:
        return next((s for s in self.slots if s.projector == projector), None)

    def channel(self, channel: ChannelKey) -> list[ProjectorSlot]:
        """Slots of an l-channel or j-channel, in `iproj` order."""
        return [s for s in self.slots if s.projector.channel == channel]

    def l_channel(self, l: int) -> list[ProjectorSlot]:  # noqa: E741
        """Slots of all channels with orbital angular momentum `l`."""
        return [s for s in self.slots if s.projector.channel.l == l]

    def uses_ep(self, l: int) -> bool:  # noqa: E741
        """Whether the input `ep` of l-channel `l` sets a projector state energy."""
        return any(s.energy == Energy.EP for s in self.l_channel(l))

    def uses_debl(self, l: int, iproj: int | None = None) -> bool:  # noqa: E741
        """Whether the input `debl` of l-channel `l` sets a projector state energy (of projector
        `iproj`, if given)."""
        return any(
            s.energy == Energy.DEBL and iproj in (None, s.projector.iproj)
            for s in self.l_channel(l)
        )


def predict_projector_layout(
    inp: OncvpspInput, program: str, version: str, relativistic: bool
) -> ProjectorLayout:
    """Projector slots that `program` (ONCVPSP or METAPSP) at `version` builds for the input `inp`;
    fully-relativistic if `relativistic` (ONCVPSP only)."""
    if program == "ONCVPSP" and version.startswith(("2", "3")):
        # No 2.x source is available; v2 is assumed to follow the 3.3.1 rules
        slots_of = _v3_slots
    elif (program == "ONCVPSP" and version.startswith("4")) or program == "METAPSP":
        slots_of = _v4_slots
    else:
        raise ValueError(f"Unsupported program version: {program} {version}")
    slots: list[ProjectorSlot] = []
    for l in range(inp.pseudopotentials.lmax + 1):  # noqa: E741
        for k, channel in enumerate(j_channels(l, relativistic)):
            slots += slots_of(inp, channel, partner=k > 0)
    return ProjectorLayout(tuple(slots))


def _states_of(inp: OncvpspInput, l: int) -> tuple[list[int], int]:  # noqa: E741
    """Reference-configuration indices of the valence states of `l` (in input order), and the
    principal quantum number of the first state above the core for `l` (one above the outermost
    core state of `l`, else `l + 1`)."""
    config = list(inp.reference_configuration)
    nc = inp.oncvpsp.nc
    core = [s.n for s in config[:nc] if s.l == l]
    valence = [i for i in range(nc, len(config)) if config[i].l == l]
    return valence, (core[-1] + 1 if core else l + 1)


def _is_local(inp: OncvpspInput, l: int) -> bool:  # noqa: E741
    return inp.local_potential.lloc == l


def _v4_slots(
    inp: OncvpspInput, channel: ChannelKey, partner: bool
) -> list[ProjectorSlot]:
    """ONCVPSP 4.0.1 (`oncvpsp.F90:393-449`, `oncvpsp_r.F90:396-533`) and METAPSP
    (`oncvpsp_m.f90`): every valence state of `l` in input order, up to `nproj`; then wellstates, the first at `ep` if no valence state was used, the others at the
    previous energy plus `debl`. The local channel has `nproj = 1`. In fully-relativistic runs,
    `wellstate` is called for j = l + 1/2 only, and j = l - 1/2 is solved in the same well. With
    the confined well of azadoks/oncvpsp, `well_n` sets the n of the first wellstate."""
    config = list(inp.reference_configuration)
    l = channel.l  # noqa: E741
    nproj = 1 if _is_local(inp, l) else inp.vkb_projectors.nproj[l]
    valence, n_next = _states_of(inp, l)
    slots: list[ProjectorSlot] = []
    for i in valence:
        slots.append(
            ProjectorSlot(
                ProjectorKey(channel, len(slots) + 1),
                config[i].n,
                Solver.BOUND,
                Energy.VALENCE,
                i,
            )
        )
        # The check follows the first valence state, so nproj = 0 (lloc = 4) keeps them all
        if len(slots) == nproj:
            break
    if slots:
        n_next = slots[-1].n + 1
    # azadoks/oncvpsp, well_tail = "confined": the input's well_n, if set, is the n of the first
    # wellstate (an error below ONCVPSP's choice); the next ones follow by one each
    if inp.pseudopotentials.well_tail == "confined" and inp.vkb_projectors.well_n:
        well_n = inp.vkb_projectors.well_n[inp.vkb_projectors.l.index(l)]
        if well_n > 0:
            n_next = well_n
    while len(slots) < nproj:
        slots.append(
            ProjectorSlot(
                ProjectorKey(channel, len(slots) + 1),
                n_next,
                Solver.PARTNER_WELL if partner else Solver.WELLSTATE,
                Energy.DEBL if slots else Energy.EP,
            )
        )
        n_next += 1
    return slots


def _v3_slots(
    inp: OncvpspInput, channel: ChannelKey, partner: bool
) -> list[ProjectorSlot]:
    """ONCVPSP 3.3.1 (`oncvpsp.f90:259-310, 398-445`; per kappa in `oncvpsp_r.f90:330-504`): at
    most two projectors. The first is the first valence state of `l`, solved as a bound state;
    without a valence state of `l`, a wellstate at `ep`. The second (`nproj = 2`) has `n` one above
    the first. Its energy is that of the last further valence state of `l` (which overrides
    `debl`), else the first energy plus `debl`; it is solved as a bound state of that `n` if the
    energy is negative and with `wellstate` otherwise. The first projector state is computed for
    the local channel too.

    The source also handles unoccupied valence states (`fp(l1) > 0`, `fa(ii) > 0`), but
    `check_data` rejects a reference configuration with any `f <= 0`, so every valence state is
    occupied."""
    del partner  # every kappa calls `wellstate` itself
    config = list(inp.reference_configuration)
    l = channel.l  # noqa: E741
    valence, n_first = _states_of(inp, l)
    if valence:
        first = valence[0]
        slots = [
            ProjectorSlot(
                ProjectorKey(channel, 1),
                config[first].n,
                Solver.BOUND,
                Energy.VALENCE,
                first,
            )
        ]
    else:
        slots = [
            ProjectorSlot(
                ProjectorKey(channel, 1), n_first, Solver.WELLSTATE, Energy.EP
            )
        ]
    if inp.vkb_projectors.nproj[l] == 2 and not _is_local(inp, l):
        n_second = slots[0].n + 1
        if further := valence[1:]:
            # The valence state itself only if its n is `n_second`, else another bound state
            # (reported in a warning when its energy differs)
            last = further[-1]
            slots.append(
                ProjectorSlot(
                    ProjectorKey(channel, 2),
                    n_second,
                    Solver.BOUND if config[last].n == n_second else Solver.BY_ENERGY,
                    Energy.VALENCE,
                    last,
                )
            )
        else:
            slots.append(
                ProjectorSlot(
                    ProjectorKey(channel, 2), n_second, Solver.BY_ENERGY, Energy.DEBL
                )
            )
    return slots


__all__ = [
    "Energy",
    "ProjectorLayout",
    "ProjectorSlot",
    "Solver",
    "predict_projector_layout",
]
