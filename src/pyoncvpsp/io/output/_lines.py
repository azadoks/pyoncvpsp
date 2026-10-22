"""Fast candidate lookup for line-by-line regex searches.

Most searches of the output look for a pattern in a few of its tens of thousands of lines. Rather
than run the pattern on every line, `LineIndex` finds the lines containing a substring that every
match must contain (with `str.find` on the joined text), and the pattern is run only on those. The
substring is only a pre-filter: the pattern decides every match, so the results are those of
running it on every line.
"""

from bisect import bisect_right
from functools import lru_cache
from importlib import import_module
from typing import Any, Iterator
import re

# The regex parser of the standard library (private: without it, nothing is pre-filtered)
_sre_parser: Any
try:
    _sre_parser = import_module("re._parser")
except ImportError:  # pragma: no cover
    _sre_parser = None


@lru_cache(maxsize=None)
def required_literal(regex: re.Pattern[str]) -> str:
    """Longest literal substring of every match of `regex` (`""` if none is found).

    Only literals in the top-level sequence of the pattern (and in the groups of that sequence)
    are considered, since they are matched exactly once by every match; anything else (repeats,
    alternatives, classes, case-insensitive matching) ends a literal.
    """
    if _sre_parser is None or not isinstance(regex.pattern, str):
        return ""
    literal, subpattern = _sre_parser.LITERAL, _sre_parser.SUBPATTERN
    ignorecase = _sre_parser.SRE_FLAG_IGNORECASE
    if regex.flags & re.IGNORECASE:
        return ""
    try:
        tree = _sre_parser.parse(regex.pattern, regex.flags)
    except Exception:  # pylint: disable=broad-except
        return ""
    literals: list[str] = []
    current: list[str] = []

    def end_literal() -> None:
        if current:
            literals.append("".join(current))
            current.clear()

    def walk(items) -> None:
        for op, av in items:
            if op is literal:
                current.append(chr(av))
            elif op is subpattern and not av[1] & ignorecase:
                walk(av[3])
            else:
                end_literal()

    walk(tree)
    end_literal()
    return max(literals, key=len, default="")


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

    def finditer(
        self, regex: re.Pattern[str], start: int = 0, stop: int | None = None
    ) -> Iterator[tuple[int, re.Match[str]]]:
        """`(index, match)` of the lines in `lines[start:stop]` which `regex.search` matches."""
        search = regex.search
        for i in self.containing(required_literal(regex), start, stop):
            if match := search(self.lines[i]):
                yield i, match


__all__ = ["LineIndex", "required_literal"]
