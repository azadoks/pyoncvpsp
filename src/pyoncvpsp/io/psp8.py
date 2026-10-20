"""Parser for the Abinit psp8 pseudopotential file written by ONCVPSP.

The file (``pspcod = 8``) is a header of six or seven lines, then one block per channel: a line giving
`l` (and, for a block of projectors, their energies `ekb`), followed by `mmax` lines of
`i, r, value(s)`. The blocks are the projectors of each `l` (scalar-relativistic, in a
fully-relativistic file), the local potential (at `l = lloc`, or after the projectors if
`lloc > lmax`), then the spin-orbit projectors of each `l >= 1` (`extension_switch` 2 or 3). If
`fchrg > 0` the model core charge and its first four derivatives follow, and if `extension_switch` is
1 or 3, the densities: the pseudo valence density (and, in builds of azadoks/oncvpsp, the all-electron
valence and core densities). ONCVPSP ends the file with the echo of its input between `<INPUT>` and
`</INPUT>`.

The projectors and their energies are not unique: any change of basis among the projectors of one `l`
that keeps the nonlocal kernel `sum_ij |beta_i> D_ij <beta_j|` gives the same pseudopotential.
"""

from typing import Any

import numpy as np

from pyoncvpsp.io._utils import fort_float

__all__ = ("parse_psp8",)


def _ints(line: str) -> list[int]:
    """Leading integers of a header line, before its comment."""
    values = []
    for word in line.split():
        try:
            values.append(int(word))
        except ValueError:
            break
    return values


def parse_psp8(text: str) -> dict[str, Any]:
    """Parse the text of a psp8 file (with or without ONCVPSP's `<INPUT>` echo).

    Args:
        text: Text of the file, from its title line on.

    Raises:
        ValueError: If the file is not a psp8 file, or a block is not where the header puts it.

    Returns:
        dict:
            title (str): First line.
            zatom (float), zion (float), pspd (str): Atomic number, ionic charge, date.
            pspcod (int), pspxc (int), lmax (int), lloc (int), mmax (int), r2well (float).
            rchrg (float), fchrg (float), qchrg (float): Model core radius and flag (`fchrg > 0`).
            nproj (list[int]): Number of (scalar-relativistic) projectors of each l = 0..lmax.
            extension_switch (list[int]): 1 (valence density), 2 (spin-orbit) or 3 (both), and the
                values that follow it.
            nprojso (list[int] | None): Number of spin-orbit projectors of each l = 1..lmax, if any.
            r (np.ndarray): Radial grid (bohr).
            local_potential (np.ndarray): Local potential (Ha).
            projectors (list[dict]): One per block of projectors, in file order:
                l (int): Orbital angular momentum.
                kind (str): "scalar" or "spin_orbit".
                ekb (np.ndarray): Projector energies, the diagonal of D (Ha).
                beta (np.ndarray): Projectors, shape (nproj, mmax).
            model_core (np.ndarray | None): Model core charge and its first four derivatives, shape
                (5, mmax), if `fchrg > 0`.
            densities (np.ndarray | None): Densities, shape (ncolumns, mmax), if `extension_switch`
                is 1 or 3.
            input (str | None): Echo of the input between `<INPUT>` and `</INPUT>`.
    """
    lines = text.splitlines()
    if len(lines) < 7:
        raise ValueError("psp8: too short for a header")
    head = lines[:6]
    words = head[1].split()
    zatom, zion, pspd = fort_float(words[0]), fort_float(words[1]), words[2]
    pspcod, pspxc, lmax, lloc, mmax, *rest = _ints(head[2])
    if pspcod != 8:
        raise ValueError(f"psp8: pspcod={pspcod}, not 8")
    r2well = float(rest[0]) if rest else 0.0
    rchrg, fchrg, qchrg = (fort_float(x) for x in head[3].split()[:3])
    nproj = _ints(head[4])[: lmax + 1]
    nproj += [0] * (lmax + 1 - len(nproj))
    extension_switch = _ints(head[5])
    pos = 6
    nprojso = None
    if extension_switch and extension_switch[0] in (2, 3):
        nprojso = _ints(lines[pos])[:lmax]
        nprojso += [0] * (lmax - len(nprojso))
        pos += 1

    def rows(count: int) -> np.ndarray:
        """`count` lines of `i, r, values` from `pos`, as an array of shape (ncolumns, count)."""
        nonlocal pos
        block = lines[pos : pos + count]
        if len(block) < count:
            raise ValueError(f"psp8: file ends in a block at line {pos + 1}")
        table = np.array([[fort_float(x) for x in line.split()] for line in block])
        pos += count
        return table.T

    def header(ell: int) -> list[str]:
        nonlocal pos
        words = lines[pos].split()
        if not words or int(words[0]) != ell:
            raise ValueError(f"psp8: no block for l={ell} at line {pos + 1}")
        pos += 1
        return words

    r = None
    local_potential = None
    projectors: list[dict[str, Any]] = []

    def local_block(ell: int) -> None:
        nonlocal r, local_potential
        header(ell)
        table = rows(mmax)
        r, local_potential = table[1], table[2]

    def projector_block(ell: int, count: int, kind: str) -> None:
        nonlocal r
        words = header(ell)
        ekb = np.array([fort_float(x) for x in words[1 : 1 + count]])
        table = rows(mmax)
        r = table[1]
        projectors.append(
            {"l": ell, "kind": kind, "ekb": ekb, "beta": table[2 : 2 + count]}
        )

    for ell in range(lmax + 1):
        if ell == lloc:
            local_block(ell)
        elif nproj[ell] > 0:
            projector_block(ell, nproj[ell], "scalar")
    if lloc > lmax:
        local_block(lloc)
    if nprojso is not None:
        for ell in range(1, lmax + 1):
            if nprojso[ell - 1] > 0:
                projector_block(ell, nprojso[ell - 1], "spin_orbit")
    model_core = rows(mmax)[2:7] if fchrg > 0.0 else None
    densities = None
    if extension_switch and extension_switch[0] in (1, 3):
        densities = rows(mmax)[2:]
    echo = None
    if pos < len(lines) and lines[pos].strip() == "<INPUT>":
        end = next(
            (k for k in range(pos + 1, len(lines)) if lines[k].strip() == "</INPUT>"),
            len(lines),
        )
        echo = "\n".join(lines[pos + 1 : end])
    if r is None or local_potential is None:
        raise ValueError("psp8: no local potential")
    return {
        "title": lines[0].strip(),
        "zatom": zatom,
        "zion": zion,
        "pspd": pspd,
        "pspcod": pspcod,
        "pspxc": pspxc,
        "lmax": lmax,
        "lloc": lloc,
        "mmax": mmax,
        "r2well": r2well,
        "rchrg": rchrg,
        "fchrg": fchrg,
        "qchrg": qchrg,
        "nproj": nproj,
        "extension_switch": extension_switch,
        "nprojso": nprojso,
        "r": r,
        "local_potential": local_potential,
        "projectors": projectors,
        "model_core": model_core,
        "densities": densities,
        "input": echo,
    }
