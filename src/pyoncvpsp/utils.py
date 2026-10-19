"""Utilities."""

import numpy as np

__all__ = ["build_mesh"]


def build_mesh(
    zz: float,
    amesh: float | None = None,
    rr1: float | None = None,
    rmax: float | None = None,
    relativistic: bool = False,
) -> np.ndarray:
    """Construct the radial mesh for a given atomic number zz as in ONCVPSP.

    Args:
        zz (float): Atomic number.
        amesh (float | None): Ratio of successive mesh points. Defaults to ONCVPSP's 1.006.
        rr1 (float | None): [Bohr] First mesh point. Defaults to ONCVPSP's `0.0005 / zz`, capped
            at `0.00005` except in the fully-relativistic code.
        rmax (float | None): [Bohr] Outer radius of the mesh. Defaults to ONCVPSP's 45.
        relativistic (bool): Whether the default `rr1` is that of the fully-relativistic code.

    Returns:
        np.ndarray: Logarithmic radial mesh points.
    """
    if amesh is None:
        amesh = 1.006
    al: float = np.log(amesh)
    if rr1 is None:
        rr1 = 0.0005 / zz if relativistic else min(0.0005 / zz, 0.0005 / 10)
    if rmax is None:
        rmax = 45.0
    mmax: int = int(np.log(rmax / rr1) / al)
    return rr1 * np.exp(al * np.arange(mmax))
