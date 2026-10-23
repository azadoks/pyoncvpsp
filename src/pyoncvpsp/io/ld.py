"""Parser for the stdout of `oncvpsp_ld.x` (azadoks/oncvpsp), the log derivatives of a
pseudopotential and its all-electron atom at any radii and energies.

The output is a block per l, j and radius: a line `ld_r l ikap r` (bohr), then a line
`ld l ikap e ae ps` per energy (Ha), the all-electron and pseudopotential phases
atan(r u'/u) as continuous phases. `ikap` is 1 for j = l + 1/2 (and every l of a non- or
scalar-relativistic run), 2 for j = l - 1/2. The `ld_r` and `ld` lines of the DIAGNOSTICS block
of an ONCVPSP output have the same form, so `ld_table` reads either (see
`OncvpspTextParser.diagnostics_ld_table`). Errors are printed on stdout as `<routine>: ERROR ...`,
with a nonzero exit status.
"""

from collections.abc import Iterable, Sequence
from os import PathLike
import re

import numpy as np

from pyoncvpsp.io._utils import fort_float

__all__ = ("LD_COLUMNS", "OncvpspLdError", "ld_command", "ld_table", "parse_oncvpsp_ld")

# Columns of the table of `ld_table`
LD_COLUMNS: tuple[str, ...] = ("l", "ikap", "r", "e_ha", "phase_ae", "phase_ps")

RE_ERROR: re.Pattern[str] = re.compile(r"^\s*(?P<routine>\S+): ERROR (?P<message>.*)$")


class OncvpspLdError(Exception):
    """Error reported by `oncvpsp_ld.x`, or output it did not finish."""


def ld_table(lines: Iterable[str]) -> dict[str, np.ndarray]:
    """Tidy table of the `ld_r` and `ld` lines, one row per `ld` line; other lines are ignored.

    Args:
        lines: Lines of the stdout of `oncvpsp_ld.x`, or of the DIAGNOSTICS block of an output.

    Raises:
        ValueError: If an `ld` line comes before any `ld_r` line, or is for another channel.

    Returns:
        dict[str, np.ndarray]: The columns `LD_COLUMNS`, of equal length:
            l (int): Orbital angular momentum.
            ikap (int): 1 for j = l + 1/2 (or an l-channel), 2 for j = l - 1/2.
            r (float): Radius of the log derivative (bohr), from the block's `ld_r` line.
            e_ha (float): Energy (Ha).
            phase_ae (float): All-electron phase atan(r u'/u).
            phase_ps (float): Pseudopotential phase atan(r u'/u).
    """
    rows: list[tuple[int, int, float, float, float, float]] = []
    block: tuple[int, int, float] | None = None
    for line in lines:
        tag, *fields = line.split() or [""]
        if tag == "ld_r":
            block = (int(fields[0]), int(fields[1]), fort_float(fields[2]))
        elif tag == "ld":
            ell, ikap = int(fields[0]), int(fields[1])
            if block is None or block[:2] != (ell, ikap):
                raise ValueError(f"`ld` line outside its `ld_r` block: {line!r}")
            rows.append(
                (
                    *block,
                    fort_float(fields[2]),
                    fort_float(fields[3]),
                    fort_float(fields[4]),
                )
            )
    columns = list(zip(*rows)) if rows else [()] * len(LD_COLUMNS)
    return {
        name: np.array(values, dtype=int if name in ("l", "ikap") else float)
        for name, values in zip(LD_COLUMNS, columns)
    }


def parse_oncvpsp_ld(text: str, returncode: int | None = None) -> dict[str, np.ndarray]:
    """Parse the stdout of `oncvpsp_ld.x`.

    Args:
        text: Its stdout.
        returncode: Its exit status, if known; nonzero is an error even without an error line.

    Raises:
        OncvpspLdError: If the output has an error line, the exit status is nonzero, or there
            are no log derivatives.

    Returns:
        dict[str, np.ndarray]: The table of `ld_table`.
    """
    lines = text.splitlines()
    errors = [match for line in lines if (match := RE_ERROR.match(line))]
    if errors:
        raise OncvpspLdError(
            "; ".join(f"{m['routine']}: {m['message'].strip()}" for m in errors)
        )
    if returncode:
        raise OncvpspLdError(f"oncvpsp_ld.x exited with status {returncode}")
    table = ld_table(lines)
    if not len(table["l"]):
        raise OncvpspLdError("no log derivatives in the output")
    return table


def ld_command(
    ae_file: str | PathLike,
    ps_file: str | PathLike,
    radii: Sequence[float] | None = None,
    channels: Sequence[int] | None = None,
    energies: tuple[float, float, float] | None = None,
    executable: str | PathLike = "oncvpsp_ld.x",
) -> list[str]:
    """Command line of `oncvpsp_ld.x`, for `subprocess.run`.

    Args:
        ae_file: All-electron data file of an `_ae.x` (`-a`).
        ps_file: Pseudopotential data file of a run with `--ps-file` (`-p`).
        radii: Radii (bohr), each for every l (`-r`); default the input's rxpsh.
        channels: Angular momenta from 0 to 3 (`-l`); default all four.
        energies: `(emin, emax, de)` (Ha), from emax down by de to above emin (`-e`); default the
            input's epsh1, epsh2, depsh.
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
    return command
