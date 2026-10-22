"""Fast candidate lookup for line-by-line regex searches.

Most searches of the output look for a pattern in a few of its tens of thousands of lines. Rather
than run the pattern on every line, `LineIndex` finds the lines containing a substring that every
match must contain (or one of several, for an alternation), searching the joined text, and the
pattern is run only on those. The substrings are only a pre-filter: the pattern decides every match, so the results are those of
running it on every line.
"""

from bisect import bisect_right
from functools import lru_cache
from heapq import merge
from importlib import import_module
from typing import Any, Iterable, Iterator
import re

# The regex parser of the standard library (private: without it, nothing is pre-filtered)
_sre_parser: Any
try:
    _sre_parser = import_module("re._parser")
except ImportError:  # pragma: no cover
    _sre_parser = None


@lru_cache(maxsize=None)
def required_literals(regex: re.Pattern[str]) -> frozenset[str]:
    """Literal substrings one of which every match of `regex` contains (empty if none are found).

    A literal of the top-level sequence of the pattern (or of the groups of that sequence) is
    matched exactly once by every match, and so is one of the alternatives of an alternation of
    the sequence: of these, the set whose shortest literal is longest (then the smallest) is
    returned. Anything else (repeats, classes, case-insensitive matching) ends a literal.
    """
    if _sre_parser is None or not isinstance(regex.pattern, str):
        return frozenset()
    if regex.flags & re.IGNORECASE:
        return frozenset()
    try:
        tree = _sre_parser.parse(regex.pattern, regex.flags)
    except Exception:  # pylint: disable=broad-except
        return frozenset()
    return _sequence_literals(_sre_parser, tree)


def _sequence_literals(sre_parser: Any, items) -> frozenset[str]:
    literal, subpattern = sre_parser.LITERAL, sre_parser.SUBPATTERN
    branch, ignorecase = sre_parser.BRANCH, sre_parser.SRE_FLAG_IGNORECASE
    candidates: list[frozenset[str]] = []
    current: list[str] = []

    def end_literal() -> None:
        if current:
            candidates.append(frozenset(["".join(current)]))
            current.clear()

    def walk(items) -> None:
        for op, av in items:
            if op is literal:
                current.append(chr(av))
            elif op is subpattern and not av[1] & ignorecase:
                walk(av[3])
            else:
                end_literal()
                if op is branch:
                    alternatives = [
                        _sequence_literals(sre_parser, items) for items in av[1]
                    ]
                    if all(alternatives):
                        candidates.append(frozenset().union(*alternatives))

    walk(items)
    end_literal()
    return max(
        candidates, key=lambda c: (min(map(len, c)), -len(c)), default=frozenset()
    )


class LineIndex:
    """Lines joined into one text, to find the lines containing a substring."""

    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.text = "\n".join(lines)
        self.starts = [0] * len(lines)
        offset = 0
        for i, line in enumerate(lines):
            self.starts[i] = offset
            offset += len(line) + 1

    def _offset(self, i: int) -> int:
        return self.starts[i] if i < len(self.starts) else len(self.text)

    def containing(
        self, needle: str, start: int = 0, stop: int | None = None
    ) -> Iterator[int]:
        """Indices of the lines in `lines[start:stop]` which contain `needle`, in order."""
        stop = len(self.lines) if stop is None else min(stop, len(self.lines))
        if start >= stop:
            return
        if not needle:
            yield from range(start, stop)
            return
        if "\n" in needle:
            return
        end = self._offset(stop)
        pos = self.text.find(needle, self._offset(start), end)
        while pos != -1:
            i = bisect_right(self.starts, pos) - 1
            yield i
            pos = self.text.find(needle, self._offset(i + 1), end)

    def containing_any(
        self, needles: Iterable[str], start: int = 0, stop: int | None = None
    ) -> Iterator[int]:
        """Indices of the lines in `lines[start:stop]` which contain any of `needles`, in order."""
        # A `str.find` pass for each needle is much faster than a regex of the alternatives
        merged = merge(
            *(self.containing(needle, start, stop) for needle in set(needles))
        )
        previous = -1
        for i in merged:
            if i != previous:
                yield i
                previous = i

    def finditer(
        self, regex: re.Pattern[str], start: int = 0, stop: int | None = None
    ) -> Iterator[tuple[int, re.Match[str]]]:
        """`(index, match)` of the lines in `lines[start:stop]` which `regex.search` matches."""
        search = regex.search
        needles = required_literals(regex) or {""}
        for i in self.containing_any(needles, start, stop):
            if match := search(self.lines[i]):
                yield i, match


__all__ = ["LineIndex", "required_literals"]
