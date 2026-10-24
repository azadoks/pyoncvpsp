"""Parser for the stdout of `oncvpsp_confine.x` (azadoks/oncvpsp), a pseudopotential and its
all-electron atom confined: in a Wigner-Seitz sphere, and compressed by a wall.

The sphere block is a block per l, j and radius: a line `ws_r l ikap rs` (bohr), then, for each
band edge of the Wigner-Seitz rule in the energy window, in ascending all-electron energy,
either a line `ws l ikap edge n e_ae e_ps` (Ha) followed by its `ws_ecut l ikap edge n tol ecut`
lines (the cutoff, Ha, at which the pseudopotential's state in the sphere is within tol of its
converged energy, for the tolerances reached among 1e-2 to 1e-6 Ha), one
`ws_floor l ikap edge n ecut err` line (the largest cutoff and its error) and, with `-t`, its
`ws_conv l ikap edge n ecut err` lines (the error at every basis size); or a line
`ws_missing l ikap edge n e_ae` for an all-electron edge with no pseudopotential partner. `edge`
is 1 for the bottom of the band (r phi'/phi = 0 at rs), 2 for its centre (-l-1) and 3 for its top
(phi = 0); `n` is the all-electron state's nodes inside rs plus l + 1; `ikap` is 1 for
j = l + 1/2 (and every l of a non- or scalar-relativistic run), 2 for j = l - 1/2.

The wall block is a line `wall_r0 r0` (bohr), the onset of the wall, then for each strength k of
the wall (Ha/bohr**2), the free atom (k = 0) first, a line `wall k etot_ae etot_ps de_ae de_ps`
(Ha), the total energies and their changes from the free atom, followed by a line
`wall_eig k n l ikap e_ae e_ps` per valence state, with `n` the all-electron principal quantum
number; or, where an atom does not converge, a line `wall_failed k atom` (`ae` or `ps`) instead.

Comment lines (`#`) and the `build:` line are ignored. Errors are printed on stdout as
`<routine>: ERROR ...`, with a nonzero exit status.
"""

from collections.abc import Iterable, Sequence
from os import PathLike
import re

import numpy as np

from pyoncvpsp.io._utils import fort_float

__all__ = (
    "CONFINE_COLUMNS",
    "OncvpspConfineError",
    "confine_command",
    "confine_tables",
    "parse_oncvpsp_confine",
)

# Columns of each table of `confine_tables`
CONFINE_COLUMNS: dict[str, tuple[str, ...]] = {
    "edges": ("l", "ikap", "rs", "edge", "n", "e_ae", "e_ps"),
    "ecut": ("l", "ikap", "rs", "edge", "n", "tol", "ecut"),
    "floor": ("l", "ikap", "rs", "edge", "n", "ecut", "err"),
    "conv": ("l", "ikap", "rs", "edge", "n", "ecut", "err"),
    "wall": ("k", "etot_ae", "etot_ps", "de_ae", "de_ps"),
    "wall_eig": ("k", "n", "l", "ikap", "e_ae", "e_ps"),
    "wall_failed": ("k", "atom"),
}

_INT_COLUMNS = frozenset(("l", "ikap", "edge", "n"))

# The table of each line of the sphere block that follows a `ws_r` line
_SPHERE_TABLES = {
    "ws": "edges",
    "ws_missing": "edges",
    "ws_ecut": "ecut",
    "ws_floor": "floor",
    "ws_conv": "conv",
}

RE_ERROR: re.Pattern[str] = re.compile(r"^\s*(?P<routine>\S+): ERROR (?P<message>.*)$")


class OncvpspConfineError(Exception):
    """Error reported by `oncvpsp_confine.x`, or output it did not finish."""


def _column(name: str, values: Sequence) -> np.ndarray:
    if name == "atom":
        return np.array(values, dtype=str)
    return np.array(values, dtype=int if name in _INT_COLUMNS else float)


def confine_tables(lines: Iterable[str]) -> dict:
    """Tidy tables of the lines of `oncvpsp_confine.x`; other lines are ignored.

    Args:
        lines: Lines of its stdout.

    Raises:
        ValueError: If a sphere line comes before any `ws_r` line, or is for another channel.

    Returns:
        dict: The tables of `CONFINE_COLUMNS`, each a dict of columns of equal length, and
            `wall_r0`, the onset of the wall (bohr), or None without a wall block:
            edges: A row per `ws` and `ws_missing` line, `e_ps` NaN for `ws_missing`.
            ecut: A row per `ws_ecut` line, `ecut` the cutoff (Ha) for the tolerance `tol` (Ha).
            floor: A row per `ws_floor` line, the largest cutoff and its error (Ha).
            conv: A row per `ws_conv` line (`-t`), the error at each cutoff (Ha).
            wall: A row per `wall` line, the total energies and their changes (Ha).
            wall_eig: A row per `wall_eig` line, the eigenvalues (Ha).
            wall_failed: A row per `wall_failed` line, `atom` "ae" or "ps".
            The sphere tables carry `rs` (bohr) from the block's `ws_r` line.
    """
    rows: dict[str, list[tuple]] = {name: [] for name in CONFINE_COLUMNS}
    wall_r0: float | None = None
    block: tuple[int, int, float] | None = None
    for line in lines:
        tag, *fields = line.split() or [""]
        if tag == "ws_r":
            block = (int(fields[0]), int(fields[1]), fort_float(fields[2]))
        elif tag in _SPHERE_TABLES:
            ell, ikap = int(fields[0]), int(fields[1])
            if block is None or block[:2] != (ell, ikap):
                raise ValueError(f"`{tag}` line outside its `ws_r` block: {line!r}")
            reals = [fort_float(value) for value in fields[4:]]
            if tag == "ws_missing":
                reals.append(float("nan"))
            rows[_SPHERE_TABLES[tag]].append(
                (*block, int(fields[2]), int(fields[3]), *reals)
            )
        elif tag == "wall_r0":
            wall_r0 = fort_float(fields[0])
        elif tag == "wall":
            rows["wall"].append(tuple(fort_float(value) for value in fields))
        elif tag == "wall_eig":
            rows["wall_eig"].append(
                (
                    fort_float(fields[0]),
                    *(int(value) for value in fields[1:4]),
                    *(fort_float(value) for value in fields[4:]),
                )
            )
        elif tag == "wall_failed":
            rows["wall_failed"].append((fort_float(fields[0]), fields[1]))
    tables: dict = {}
    for name, names in CONFINE_COLUMNS.items():
        columns = list(zip(*rows[name])) if rows[name] else [()] * len(names)
        tables[name] = {
            column: _column(column, values) for column, values in zip(names, columns)
        }
    tables["wall_r0"] = wall_r0
    return tables


def parse_oncvpsp_confine(text: str, returncode: int | None = None) -> dict:
    """Parse the stdout of `oncvpsp_confine.x`.

    Args:
        text: Its stdout.
        returncode: Its exit status, if known; nonzero is an error even without an error line.

    Raises:
        OncvpspConfineError: If the output has an error line or the exit status is nonzero.

    Returns:
        dict: The tables of `confine_tables`. A run with `-s` and `-w none` has every table
            empty.
    """
    lines = text.splitlines()
    errors = [match for line in lines if (match := RE_ERROR.match(line))]
    if errors:
        raise OncvpspConfineError(
            "; ".join(f"{m['routine']}: {m['message'].strip()}" for m in errors)
        )
    if returncode:
        raise OncvpspConfineError(f"oncvpsp_confine.x exited with status {returncode}")
    return confine_tables(lines)


def confine_command(
    ae_file: str | PathLike,
    ps_file: str | PathLike,
    radii: Sequence[float] | None = None,
    channels: Sequence[int] | None = None,
    energies: tuple[float, float, float] | None = None,
    ecut_max: float | None = None,
    table: bool = False,
    sphere: bool = True,
    walls: Sequence[float] | None = None,
    wall_radius: float | None = None,
    executable: str | PathLike = "oncvpsp_confine.x",
) -> list[str]:
    """Command line of `oncvpsp_confine.x`, for `subprocess.run`.

    Args:
        ae_file: All-electron data file of an `_ae.x` (`-a`).
        ps_file: Pseudopotential data file of a run with `--ps-file` (`-p`).
        radii: Radii of the sphere (bohr), each beyond every core radius (`-r`); default 1.25,
            1.5 and 2 times the largest.
        channels: Angular momenta from 0 to 3 (`-l`); default all four.
        energies: `(emin, emax, de)` (Ha), the window of the edges and the step of the scan of
            the phases (`-e`); default the input's epsh1, epsh2, and 0.1.
        ecut_max: The largest cutoff of the basis (Ha) (`-c`); default 200.
        table: Also print the error at every basis size, the `ws_conv` lines (`-t`).
        sphere: False leaves out the Wigner-Seitz sphere (`-s`).
        walls: The strengths of the wall (Ha/bohr**2) (`-w`), empty for none; default 0.01,
            0.03 and 0.1.
        wall_radius: The onset of the wall (bohr) (`-R`); default the largest core radius.
        executable: Path or name of the executable.

    Returns:
        list[str]: The arguments.
    """
    command = [str(executable), "-a", str(ae_file), "-p", str(ps_file)]
    if radii is not None:
        command += ["-r", ",".join(repr(float(r)) for r in radii)]
    if channels is not None:
        command += ["-l", ",".join(str(int(ell)) for ell in channels)]
    if energies is not None:
        command += ["-e", ",".join(repr(float(e)) for e in energies)]
    if ecut_max is not None:
        command += ["-c", repr(float(ecut_max))]
    if table:
        command.append("-t")
    if not sphere:
        command.append("-s")
    if walls is not None:
        command += [
            "-w",
            ",".join(repr(float(k)) for k in walls) if len(walls) else "none",
        ]
    if wall_radius is not None:
        command += ["-R", repr(float(wall_radius))]
    return command
