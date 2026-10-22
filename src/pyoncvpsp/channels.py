"""Identifiers of l-channels, j-channels and projectors.

An l-channel `(l,)` has no total angular momentum `j`; a j-channel `(l, j)` with `j = l ± 1/2`
exists only in fully-relativistic calculations, which split each l-channel of the input into its
j-channels (only `(0, 1/2)` for `l = 0`). ONCVPSP prints j-channels in several conventions (Dirac
`kappa`, a spin sign, a `-` plot marker); `ChannelKey` is built from any of them and is the only
form used past the point of parsing.

Parsed records carry a channel as the flat fields `l` and `j` (`j` is `None` for an l-channel),
and a projector as `l`, `j` and `iproj`.
"""

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Any, Mapping

HALF = Fraction(1, 2)


@dataclass(frozen=True)
class ChannelKey:
    """An l-channel (`j is None`) or a j-channel (`j = l ± 1/2`)."""

    l: int  # noqa: E741
    j: Fraction | None = None

    def __post_init__(self) -> None:
        if self.l < 0:
            raise ValueError(f"l must be non-negative, got {self.l}")
        if self.j is not None:
            object.__setattr__(self, "j", Fraction(self.j))
            if self.j not in (self.l + HALF, self.l - HALF) or self.j < 0:
                raise ValueError(
                    f"j must be l ± 1/2 >= 1/2, got l={self.l}, j={self.j}"
                )

    @classmethod
    def from_kappa(cls, kappa: int) -> "ChannelKey":
        """j-channel of the Dirac quantum number `kappa`: `-(l+1)` for `j = l + 1/2`, `l` for `j = l - 1/2`."""
        if kappa == 0:
            raise ValueError("kappa must be non-zero")
        if kappa < 0:
            return cls(-kappa - 1, -kappa - HALF)
        return cls(kappa, kappa - HALF)

    @classmethod
    def from_printed_kappa(cls, l: int, kappa: int | None) -> "ChannelKey":  # noqa: E741
        """Channel of a line which prints `l` and, if fully-relativistic, `kappa` (absent or 0 otherwise)."""
        if not kappa:
            return cls(l)
        channel = cls.from_kappa(kappa)
        if channel.l != l:
            raise ValueError(f"kappa={kappa} does not belong to l={l}")
        return channel

    @classmethod
    @lru_cache(maxsize=None)  # one call per plot line; channels are immutable
    def from_plot_marker(cls, l: int, marker: str, relativistic: bool) -> "ChannelKey":  # noqa: E741
        """Channel of a plot line, whose marker column is `-` for `j = l + 1/2` (and for `l = 0`)
        and blank for `j = l - 1/2` in fully-relativistic output."""
        if not relativistic:
            return cls(l)
        if marker == "-" or l == 0:
            return cls(l, l + HALF)
        return cls(l, l - HALF)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ChannelKey":
        """Channel of a parsed record with the fields `l` and `j`."""
        j = record.get("j")
        return cls(record["l"], None if j is None else Fraction(j).limit_denominator(2))

    @property
    def is_j_channel(self) -> bool:
        return self.j is not None

    @property
    def kappa(self) -> int | None:
        """Dirac quantum number, `None` for an l-channel."""
        if self.j is None:
            return None
        return -(self.l + 1) if self.j > self.l else self.l

    @property
    def l_channel(self) -> "ChannelKey":
        """The l-channel this channel belongs to."""
        return ChannelKey(self.l)

    def as_dict(self) -> dict[str, Any]:
        """Flat record fields `l` and `j` (`j` as a float, `None` for an l-channel)."""
        return {"l": self.l, "j": None if self.j is None else float(self.j)}

    def __str__(self) -> str:
        return f"l={self.l}" if self.j is None else f"l={self.l}, j={self.j}"


def j_channels(l: int, relativistic: bool) -> list[ChannelKey]:  # noqa: E741
    """Channels into which the l-channel `l` of the input is split: itself if not fully-relativistic,
    else its j-channels in the order ONCVPSP processes them (`j = l + 1/2` first)."""
    if not relativistic:
        return [ChannelKey(l)]
    if l == 0:
        return [ChannelKey(0, HALF)]
    return [ChannelKey(l, l + HALF), ChannelKey(l, l - HALF)]


@dataclass(frozen=True)
class ProjectorKey:
    """The `iproj`-th projector (1-based) of an l-channel or j-channel."""

    channel: ChannelKey
    iproj: int

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ProjectorKey":
        """Projector of a parsed record with the fields `l`, `j` and `iproj`."""
        return cls(ChannelKey.from_record(record), record["iproj"])

    def as_dict(self) -> dict[str, Any]:
        """Flat record fields `l`, `j` and `iproj`."""
        return {**self.channel.as_dict(), "iproj": self.iproj}

    def __str__(self) -> str:
        return f"{self.channel}, iproj={self.iproj}"


__all__ = ["ChannelKey", "ProjectorKey", "j_channels"]
