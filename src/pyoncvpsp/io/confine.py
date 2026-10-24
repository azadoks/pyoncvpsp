"""Parser for the stdout of `oncvpsp_confine.x` (azadoks/oncvpsp), a pseudopotential and its
all-electron atom confined: in a Wigner-Seitz sphere, and compressed by a wall.

The sphere block is a block per l, j and radius: a line `ws_r l ikap rs` (bohr), then, for each
band edge of the Wigner-Seitz rule in the energy window, in ascending all-electron energy,
either a line `ws l ikap edge n e_ae e_ps` (Ha) followed by its `ws_ecut l ikap edge n tol ecut`
lines (the cutoff, Ha, at which the pseudopotential's state in the sphere is within tol of its
converged energy, for the tolerances reached among 1e-2 to 1e-6 Ha), one
`ws_floor l ikap edge n ecut err` line (the largest cutoff and its error) and, with `-t`, its
`ws_conv l ikap edge n ecut err` lines (the error at every basis size); or a line
`ws_missing l ikap edge n e_ae` for an all-electron edge with no pseudopotential partner, or
`ws_core l ikap edge n e_ae` for one that is the band of a core state. A `ws` line may be followed
by `ws_paired_off l ikap edge n e_ps`, a pseudopotential edge of another phase nearer the
all-electron edge than its partner. After the block's edges, `ws_extra l ikap edge e_ps` is a
pseudopotential edge in the window with no all-electron partner (a ghost). `edge` is 1 for the
bottom of the band (r phi'/phi = 0 at rs), 2 for its centre (-l-1) and 3 for its top (phi = 0);
`n` is the all-electron state's nodes inside rs plus l + 1; `ikap` is 1 for j = l + 1/2 (and
every l of a non- or scalar-relativistic run), 2 for j = l - 1/2.

The wall block is a line `wall_r0 r0` (bohr), the onset of the wall, then for each strength k of
the wall (Ha/bohr**2), the free atom (k = 0) first, a line `wall k etot_ae etot_ps de_ae de_ps`
(Ha), the total energies and their changes from the free atom, followed by a line
`wall_eig k n l ikap e_ae e_ps` per valence state, with `n` the all-electron principal quantum
number; or, where an atom does not converge, a line `wall_failed k atom` (`ae` or `ps`) instead.

The last line (azadoks/oncvpsp from 5c91c31) is `confine_summary max_de ecut4 ecut5 n_missing
n_extra n_paired_off n_wall_failed max_dde`: over the edges below 0 Ha, the largest
|e_ps - e_ae| and the largest cutoffs at 1e-4 and 1e-5 Ha (Infinity where one is not reached);
the counts of the `ws_missing`, `ws_extra`, `ws_paired_off` and `wall_failed` lines; and the
largest |de_ps - de_ae| of the walls. A part left out (`-s`, `-w none`) gives NaN and -1.

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
    "SUMMARY_KEYS",
    "OncvpspConfineError",
    "confine_command",
    "confine_tables",
    "parse_oncvpsp_confine",
)

# Columns of each table of `confine_tables`
CONFINE_COLUMNS: dict[str, tuple[str, ...]] = {
    "edges": ("l", "ikap", "rs", "edge", "n", "e_ae", "e_ps", "partner"),
    "ecut": ("l", "ikap", "rs", "edge", "n", "tol", "ecut"),
    "floor": ("l", "ikap", "rs", "edge", "n", "ecut", "err"),
    "conv": ("l", "ikap", "rs", "edge", "n", "ecut", "err"),
    "paired_off": ("l", "ikap", "rs", "edge", "n", "e_ps"),
    "extra": ("l", "ikap", "rs", "edge", "e_ps"),
    "wall": ("k", "etot_ae", "etot_ps", "de_ae", "de_ps"),
    "wall_eig": ("k", "n", "l", "ikap", "e_ae", "e_ps"),
    "wall_failed": ("k", "atom"),
}

# Keys of the `summary` of `confine_tables`, in the order of the `confine_summary` line
SUMMARY_KEYS: tuple[str, ...] = (
    "max_de",
    "ecut4",
    "ecut5",
    "n_missing",
    "n_extra",
    "n_paired_off",
    "n_wall_failed",
    "max_dde",
)

_INT_COLUMNS = frozenset(("l", "ikap", "edge", "n"))
_STR_COLUMNS = frozenset(("atom", "partner"))

# For each line of the sphere block that follows a `ws_r` line: its table, the number of its
# integers after l and ikap, and what it adds to the row after its values
_SPHERE_LINES: dict[str, tuple[str, int, tuple]] = {
    "ws": ("edges", 2, ("ps",)),
    "ws_missing": ("edges", 2, (float("nan"), "missing")),
    "ws_core": ("edges", 2, (float("nan"), "core")),
    "ws_ecut": ("ecut", 2, ()),
    "ws_floor": ("floor", 2, ()),
    "ws_conv": ("conv", 2, ()),
    "ws_paired_off": ("paired_off", 2, ()),
    "ws_extra": ("extra", 1, ()),
}

RE_ERROR: re.Pattern[str] = re.compile(r"^\s*(?P<routine>\S+): ERROR (?P<message>.*)$")


class OncvpspConfineError(Exception):
    """Error reported by `oncvpsp_confine.x`, or output it did not finish."""


def _column(name: str, values: Sequence) -> np.ndarray:
    if name in _STR_COLUMNS:
        return np.array(values, dtype=str)
    return np.array(values, dtype=int if name in _INT_COLUMNS else float)


def confine_tables(lines: Iterable[str]) -> dict:
    """Tidy tables of the lines of `oncvpsp_confine.x`; other lines are ignored.

    Args:
        lines: Lines of its stdout.

    Raises:
        ValueError: If a sphere line comes before any `ws_r` line, or is for another channel.

    Returns:
        dict: The tables of `CONFINE_COLUMNS`, each a dict of columns of equal length;
            `wall_r0`, the onset of the wall (bohr), or None without a wall block; and
            `summary`, the `confine_summary` line as a dict of `SUMMARY_KEYS`, or None
            without one (before azadoks/oncvpsp 5c91c31):
            edges: A row per `ws`, `ws_missing` and `ws_core` line, `partner` "ps", "missing"
                or "core", `e_ps` NaN without a partner.
            ecut: A row per `ws_ecut` line, `ecut` the cutoff (Ha) for the tolerance `tol` (Ha).
            floor: A row per `ws_floor` line, the largest cutoff and its error (Ha).
            conv: A row per `ws_conv` line (`-t`), the error at each cutoff (Ha).
            paired_off: A row per `ws_paired_off` line, `e_ps` the nearer edge of another phase.
            extra: A row per `ws_extra` line, a pseudopotential edge with no partner.
            wall: A row per `wall` line, the total energies and their changes (Ha).
            wall_eig: A row per `wall_eig` line, the eigenvalues (Ha).
            wall_failed: A row per `wall_failed` line, `atom` "ae" or "ps".
            The sphere tables carry `rs` (bohr) from the block's `ws_r` line. The summary's
            reals may be inf (a cutoff not reached) or NaN, and its counts -1, for a part left
            out.
    """
    rows: dict[str, list[tuple]] = {name: [] for name in CONFINE_COLUMNS}
    wall_r0: float | None = None
    summary: dict[str, float | int] | None = None
    block: tuple[int, int, float] | None = None
    for line in lines:
        tag, *fields = line.split() or [""]
        if tag == "ws_r":
            block = (int(fields[0]), int(fields[1]), fort_float(fields[2]))
        elif tag in _SPHERE_LINES:
            table, nint, extra = _SPHERE_LINES[tag]
            ell, ikap = int(fields[0]), int(fields[1])
            if block is None or block[:2] != (ell, ikap):
                raise ValueError(f"`{tag}` line outside its `ws_r` block: {line!r}")
            rows[table].append(
                (
                    *block,
                    *(int(value) for value in fields[2 : 2 + nint]),
                    *(fort_float(value) for value in fields[2 + nint :]),
                    *extra,
                )
            )
        elif tag == "confine_summary":
            summary = {
                key: int(value) if key.startswith("n_") else fort_float(value)
                for key, value in zip(SUMMARY_KEYS, fields)
            }
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
    tables["summary"] = summary
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
            empty, and a summary of NaN and -1.
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
