"""ONVPSP input file handling."""

import logging
from copy import deepcopy
from math import floor
import re
from os import PathLike
from typing import Annotated, Any, Iterator, TextIO
import warnings

import numpy as np
import toml
from pydantic import BaseModel, Field, field_validator, model_validator

from ...utils import build_mesh
from .._utils import flatten, recursive_merge

LOGGER: logging.Logger = logging.getLogger(__name__)


class OncvpspAtomInput(BaseModel):  # pylint: disable=too-few-public-methods
    r"""Input parameters defining the atom being pseudized.

    Attributes:
        atsym (str): Atomic symbol of the element (e.g. 'H', 'He', ...).
        z (float): Nuclear charge.
        nc (int): Number of core states.
        nv (int): Number of valence states.
        iexc (int): Exchange-correlation functional. Negative values indicate Libxc; 1-4 are built-in functionals;
            5 is r2SCAN (METAPSP only); 6 is Slater exchange with VWN5 correlation (Libxc -001007) and 7 the same with
            the relativistic exchange of MacDonald and Vosko at `c_light` (Libxc -532007), azadoks/oncvpsp only.
        psfile (str): Pseudopotential output type.
        c_light (None | float): Speed of light of the scalar- and fully-relativistic equations
            and of the relativistic exchange: the optional seventh value of the first line, or
            `c_light` in `[oncvpsp]` of the TOML input (azadoks/oncvpsp). If `None`, ONCVPSP's
            137.036; ONCVPSP takes a value of 0 or less to mean the same, which is read here as
            `None`.
    """

    atsym: Annotated[
        str, Field(description="Atomic symbol of the element (e.g. 'H', 'He', ...)")
    ]
    z: Annotated[
        float, Field(gt=0.0, le=120.0, description="Atomic charge (number of protons)")
    ]
    nc: Annotated[int, Field(ge=0, description="Number of core states")]
    nv: Annotated[int, Field(ge=1, description="Number of valence states")]
    iexc: Annotated[
        int,
        Field(
            le=7,
            description=(
                "Exchange-correlation functional. Negative values indicate Libxc. 1-4 are built-in functionals."
                " 5 is r2SCAN (METAPSP only). 6 is Slater + VWN5 and 7 is VWN5 with the MacDonald-Vosko"
                " relativistic exchange at c_light (azadoks/oncvpsp only)."
            ),
        ),
    ]
    psfile: Annotated[str, Field(description="Pseudopotential output type.")]
    c_light: None | Annotated[float, Field(ge=1.0)] = None

    @field_validator("c_light", mode="before")
    @classmethod
    def _nonpositive_is_default(cls, value: Any) -> Any:
        if isinstance(value, (int, float)) and value <= 0:
            return None
        return value

    @model_validator(mode="after")
    def _check_atomic_symbol(self) -> "OncvpspAtomInput":
        if len(self.atsym) == 0:
            raise ValueError("Atomic symbol cannot be empty")
        if len(self.atsym) > 2:
            raise ValueError("Atomic symbol must be at most 2 characters")
        return self

    @model_validator(mode="after")
    def _check_exchange_correlation_type(self) -> "OncvpspAtomInput":
        if self.iexc == 0:
            raise ValueError(
                "exchange_correlation_type can be negative (LibXC), 1-4 (built-in), 5 (METAPSP r2SCAN),"
                " 6-7 (azadoks/oncvpsp VWN), but not 0"
            )
        return self

    @model_validator(mode="after")
    def _check_psp_output_format(self) -> "OncvpspAtomInput":
        valid_formats = {"psp8", "upf", "psml", "both", "all", "none"}
        if self.psfile not in valid_formats:
            raise ValueError(
                f"Invalid pseudopotential output format: {self.psfile}. "
                f"Valid formats are: {', '.join(valid_formats)}"
            )
        return self

    def as_dat_string(self, with_comment: bool = False) -> str:
        """String representation of the pseudopotential input in .dat file format."""
        s: str = (
            f"{self.atsym:>10s} {self.z:>10.5f} {self.nc:>10d} "
            f"{self.nv:>10d} {self.iexc:>10d} {self.psfile:>10s}"
        )
        if self.c_light is not None:
            s += f" {self.c_light!r:>10s}"
        if with_comment:
            c_light = f" {'c_light':>10s}" if self.c_light is not None else ""
            s = (
                f"# {'atsym':>8s} {'z':>10s} {'nc':>10s} "
                f"{'nv':>10s} {'iexc':>10s} {'psfile':>10s}{c_light}\n" + s
            )
        return s


class AtomicStateInput(BaseModel):  # pylint: disable=too-few-public-methods
    r"""Input parameters describing an atomic state.

    Attributes:
        n (int): Principal quantum number (1 <= n).
        l (int): Angular momentum quantum number (0 <= l <= n - 1).
        f (float): Occupation number (0.0 <= f <= 2(2l + 1)).
    """

    n: Annotated[int, Field(ge=1)]
    l: Annotated[int, Field(ge=0)]  # noqa: E741
    f: Annotated[float, Field(ge=0.0)]

    @model_validator(mode="after")
    def _check(self) -> "AtomicStateInput":
        if self.l > self.n - 1:
            raise ValueError(f"Invalid angular momentum: (n={self.n}), l={self.l}")
        if self.f > 2 * (2 * self.l + 1):
            raise ValueError(
                f"Invalid occupation number: (n={self.n}, l={self.l}), f={self.f}"
            )
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AtomicStateInput):
            return NotImplemented
        if self.n != other.n:
            LOGGER.debug("n differs: %d != %d", self.n, other.n)
            return False
        if self.l != other.l:
            LOGGER.debug("l differs: %d != %d", self.l, other.l)
            return False
        if self.f != other.f:
            LOGGER.debug("f differs: %f != %f", self.f, other.f)
            return False
        return True

    def as_dat_string(self, with_comment: bool = False) -> str:
        """String representation of the atomic state in .dat file format."""
        s: str = f"{self.n:>10d} {self.l:>10d} {self.f:>10.5f}"
        if with_comment:
            s = f"# {'n':>8s} {'l':>10s} {'f':>10s}\n" + s
        return s


class AtomicConfigurationInput(BaseModel):  # pylint: disable=too-few-public-methods
    r"""Input parameters describing a full atomic configuration (composed of multiple atomic states).

    Attributes:
        n (list[int]): List of principal quantum numbers for each state.
        l (list[int]): List of angular momentum quantum numbers for each state.
        f (list[float]): List of occupation numbers for each state.
    """

    n: Annotated[list[int], Field(min_length=1)]
    l: Annotated[list[int], Field(min_length=1)]  # noqa: E741
    f: Annotated[list[float], Field(min_length=1)]

    def __len__(self) -> int:
        return len(self.n)

    def __getitem__(self, index: int) -> AtomicStateInput:
        return AtomicStateInput(
            n=self.n[index],
            l=self.l[index],
            f=self.f[index],
        )

    def __iter__(self):  # type: ignore
        for i in range(len(self)):
            yield self[i]

    @property
    def charge(self) -> float:
        """Total occupation (charge) of the atomic configuration."""
        return sum(self.f)

    @model_validator(mode="after")
    def _check(self) -> "AtomicConfigurationInput":
        if not len(self.n) == len(self.l) == len(self.f):
            raise ValueError(
                "n, l, and f must have the same length: "
                f"len(n)={len(self.n)}, len(l)={len(self.l)}, len(f)={len(self.f)}"
            )
        return self

    def as_dat_string(self, with_comment: bool = False) -> str:
        """String representation of the atomic state in .dat file format."""
        # Order is significant: ONCVPSP treats the first `nc` states as core states
        states = list(self)
        lines = [states[0].as_dat_string(with_comment)]
        if len(states) > 1:
            for state in states[1:]:
                lines.append(state.as_dat_string())
        return "\n".join(lines)


class PseudopotentialInput(BaseModel):  # pylint: disable=too-few-public-methods
    r"""Input parameters for a single pseudopotential channel.

    Attributes:
        l (int): Angular momentum quantum number.
        rc (float): [Bohr] Core radius beyond which the pseudopotential matches the all-electron potential.
            Correspondingly, the radius beyond which the pseudo-wavefunctions for this angular momentum
            match the all-electron wavefunctions.
        ep (float): [Hartree] Energy at which the pseudopotential is constructed.
        ncon (int): Number of constraints used in matching the pseudo-wavefunction to the all-electron
            wavefunction at rc. :math:`d^{i}\phi/dr^{i} = d^{i}\psi/dr^{i}` for :math:`i=0,...,n_{\mathrm{con}}-1`.
        nbas (int): Number of basis functions used in optimizing the pseudo-wavefunction (ncon + 2 <= nbas <= ncon + 5).
        qcut (float): [1/Bohr] Cutoff wavevector used for defining the residual kinetic energy.
    """

    l: Annotated[int, Field(ge=0)]  # noqa: E741
    rc: Annotated[float, Field(gt=0.0)]
    ep: float
    ncon: Annotated[int, Field(ge=3, le=5)]
    nbas: Annotated[int, Field(ge=5, le=10)]
    qcut: Annotated[float, Field(gt=0.0)]

    @model_validator(mode="after")
    def _check(self) -> "PseudopotentialInput":
        if not self.ncon + 2 <= self.nbas <= self.ncon + 5:
            raise ValueError(
                f"n_basis must be between n_constraints + 2 and n_constraints + 5: "
                f"n_basis={self.nbas}, n_constraints={self.ncon}"
            )
        return self

    def as_dat_string(self, with_comment: bool = False) -> str:
        """String representation of the pseudopotential input in .dat file format."""
        s: str = f"{self.l:>10d} {self.rc:>10.5f} {self.ep:10.5f} {self.ncon:>10d} {self.nbas:>10d} {self.qcut:>10.5f}"
        if with_comment:
            s = (
                f"# {'l':>8s} {'rc':>10s} {'ep':>10s} {'ncon':>10s} {'nbas':>10s} {'qcut':>10s}\n"
                + s
            )
        return s


class PseudoPotentialsInput(BaseModel):  # pylint: disable=too-few-public-methods
    r"""Input parameters for all pseudopotential channels.

    Attributes:
        lmax (int): Maximum angular momentum channel of the pseudopotential.
        l (list[int]): List of angular momentum quantum numbers for each channel (`all(li <= lmax for li in l)`).
        rc (list[float]): List of core radii for each channel.
        ep (list[float]): List of energies at which each pseudopotential channel is constructed.
        ncon (list[int]): List of number of constraints for each channel.
        nbas (list[int]): List of number of basis functions for each channel.
        qcut (list[float]): List of cutoff wavevectors for each channel.
    """

    lmax: Annotated[
        int,
        Field(
            ge=0, description="Maximum angular momentum channel of the pseudopotential."
        ),
    ]
    l: Annotated[list[int], Field(min_length=1)]  # noqa: E741
    rc: Annotated[list[float], Field(min_length=1)]
    ep: Annotated[list[float], Field(min_length=1)]
    ncon: Annotated[list[int], Field(min_length=1)]
    nbas: Annotated[list[int], Field(min_length=1)]
    qcut: Annotated[list[float], Field(min_length=1)]

    def __len__(self) -> int:
        return self.lmax + 1

    def __getitem__(self, index: int) -> PseudopotentialInput:
        return PseudopotentialInput(
            l=self.l[index],
            rc=self.rc[index],
            ep=self.ep[index],
            ncon=self.ncon[index],
            nbas=self.nbas[index],
            qcut=self.qcut[index],
        )

    def __iter__(self):  # type: ignore
        for i in range(len(self)):
            yield self[i]

    @model_validator(mode="after")
    def _check(self) -> "PseudoPotentialsInput":
        if sorted(self.l) != list(range(self.lmax + 1)):
            raise ValueError(
                f"Pseudo potentials must cover all l from 0 to {self.lmax}"
            )
        if (
            not len(self.l)
            == len(self.rc)
            == len(self.ep)
            == len(self.ncon)
            == len(self.nbas)
            == len(self.qcut)
        ):
            raise ValueError(
                "All pseudopotential parameter lists must have the same length"
            )
        for ell in self.l:
            if ell > self.lmax:
                raise ValueError(
                    f"Pseudopotential with l={ell} exceeds l_max={self.lmax}"
                )
        return self

    def as_dat_string(self, with_comment: bool = False) -> str:
        """String representation of the pseudopotential input in .dat file format."""
        psps: list[PseudopotentialInput] = [self[ell] for ell in range(self.lmax + 1)]
        lines: list[str] = []
        if with_comment:
            lines.append(f"# {'lmax':>8s}\n{self.lmax:>10d}\n#")
        else:
            lines.append(f"{self.lmax:>10d}")
        lines.append(psps[0].as_dat_string(with_comment))
        if len(psps) > 1:
            for psp in psps[1:]:
                lines.append(psp.as_dat_string())
        return "\n".join(lines)


class LocalPotentialInput(BaseModel):  # pylint: disable=too-few-public-methods
    """Input parameters for the local part of the KB separable form of the pseudopotential

    Attributes:
        lloc (int): Angular momentum channel used to construct the local potential. If `lloc=4`,
            then a polynomial continuation of the all-electron potential from `r=rcloc` to `r=0` is used.
        lpopt (None | int): If `lloc=4`, specifies the type of polynomial used in the continuation (1 <= lpopt <= 5).
            1) Match the 0th, 2nd, and 4th radial derivatives at `rcloc`.
            2) Match the 0th, 4th, and 6th radial derivatives at `rcloc`.
            3) Match the 0th, 4th, 5th, and 6th radial derivatives at `rcloc`.
            4) Match the 0th, 4th, 6th, and 8th radial derivatives at `rcloc`.
            5) Match the 0th, 2nd, 4th, and 6th radial derivatives at `rcloc`.
        rcloc (None | float): [Bohr] The radius beyond which the local potential matches the all-electron potential.
            If `lloc != 4`, `all(rcloc < rc[l] for l in range(lmax + 1))` must hold.
            If `lloc == 4`, `all(rcloc <= rc[l] for l in range(lmax + 1))` must hold.
        dvloc0 (None | float): [Hartree] If `lloc=4`, shifts the local potential at `r = 0` depending on `lpopt`:
            1) Shift by `dvloc0 * (1-(r/rcloc)**2)**3`
            2) Shift by `dvloc0 * (1-(r/rcloc)**4)**3`
            3) Shift by `dvloc0 * (1-(r/rcloc)**4)**4`
            4) Shift by `dvloc0 * (1-(r/rcloc)**4)**4`
            5) Shift by `dvloc0 * (1-(r/rcloc)**4)**4`
    """

    lloc: Annotated[int, Field(ge=0)]
    # lpopt, rcloc, and dvloc0 are ignored by ONCVPSP unless lloc=4, so they are only validated then
    lpopt: None | int = None
    rcloc: None | float = None
    dvloc0: None | float = None

    @model_validator(mode="after")
    def _check(self) -> "LocalPotentialInput":
        if self.lloc == 4:
            exceptions = []
            if self.lpopt is None:
                exceptions.append(ValueError("lpopt must be set when lloc=4"))
            elif not 1 <= self.lpopt <= 5:
                exceptions.append(
                    ValueError(
                        f"lpopt must be between 1 and 5 when lloc=4: {self.lpopt}"
                    )
                )
            if self.rcloc is None:
                exceptions.append(ValueError("rcloc must be set when lloc=4"))
            elif self.rcloc <= 0.0:
                exceptions.append(
                    ValueError(
                        f"rcloc must be greater than 0 when lloc=4: {self.rcloc}"
                    )
                )
            if self.dvloc0 is None:
                exceptions.append(ValueError("dvloc0 must be set when lloc=4"))
            if exceptions:
                raise ExceptionGroup(
                    "Invalid LocalPotentialInput for lloc=4", exceptions
                )
        return self

    def as_dat_string(self, with_comment: bool = False) -> str:
        """String representation of the pseudopotential input in .dat file format."""
        lpopt: int = self.lpopt if self.lpopt else 0
        rc: float = self.rcloc if self.rcloc else 0.0
        dvloc0: float = self.dvloc0 if self.dvloc0 else 0.0
        s: str = f"{self.lloc:>10d} {lpopt:>10d} {rc:>10.5f} {dvloc0:>10.5f}"
        if with_comment:
            s = f"# {'lloc':>8s} {'lpopt':>10s} {'rc(5)':>10s} {'dvloc0':>10s}\n" + s
        return s


class VKBProjectorInput(BaseModel):  # pylint: disable=too-few-public-methods
    """Input parameters for the Vanderbilt / Kleinman-Bylander projector(s) of one angular momentum channel.

    Attributes:
        l (int): Angular momentum quantum number.
        nproj (int): Number of projectors for this angular momentum channel.
        debl (float): [Hartree] Energy added to that of the previous pseudo-wavefunction for constructing
            the 2nd (and subsequent) projector(s) if no corresponding occupied bound state exists.
    """

    l: Annotated[int, Field(ge=0)]  # noqa: E741
    nproj: Annotated[int, Field(ge=0, le=5)]
    debl: Annotated[float, Field(ge=0.0)] = 0.0

    def as_dat_string(self, with_comment: bool = False) -> str:
        """String representation of the pseudopotential input in .dat file format."""
        s: str = f"{self.l:>10d} {self.nproj:>10d} {self.debl:>10.5f}"
        if with_comment:
            s = f"# {'l':>8s} {'nproj':>10s} {'debl':>10s}\n" + s
        return s


class VKBProjectorsInput(BaseModel):  # pylint: disable=too-few-public-methods
    """Input parameters for all Vanderbilt/ Kleinman-Bylander projectors.

    Attributes:
        l (list[int]): List of angular momentum quantum numbers for each channel (`all(li <= lmax for li in l)`).
        nproj (list[int]): List of number of projectors for each channel.
        debl (list[float]): List of energy increments for constructing additional projectors for each channel.
    """

    l: Annotated[list[int], Field(min_length=1)]  # noqa: E741
    nproj: Annotated[list[int], Field(min_length=1)]
    debl: Annotated[list[float], Field(min_length=1)]

    def __len__(self) -> int:
        return len(self.l)

    def __getitem__(self, index: int) -> VKBProjectorInput:
        return VKBProjectorInput(
            l=self.l[index],
            nproj=self.nproj[index],
            debl=self.debl[index],
        )

    def __iter__(self):  # type: ignore
        for i in range(len(self)):
            yield self[i]

    @model_validator(mode="after")
    def _check(self) -> "VKBProjectorsInput":
        if sorted(self.l) != list(range(max(self.l) + 1)):
            raise ValueError(f"VKB projectors must cover all l from 0 to {max(self.l)}")
        return self

    def as_dat_string(self, with_comment: bool = False) -> str:
        """String representation of the pseudopotential input in .dat file format."""
        projectors: list[VKBProjectorInput] = [
            self[ell] for ell in range(max(self.l) + 1)
        ]
        lines: list[str] = [projectors[0].as_dat_string(with_comment)]
        if len(projectors) > 1:
            for p in projectors[1:]:
                lines.append(p.as_dat_string(with_comment=False))
        return "\n".join(lines)


class ModelCoreChargeInput(BaseModel):  # pylint: disable=too-few-public-methods
    r"""Input parameters controlling the construction of a model core charge density.

    Attributes:
        icmod (int): Model core charge type:
            0) No model core charge.
            1) Polynomial continuation of the all-electron core charge density. Matches the
                all-electron core charge density and its first 4 radial derivatives at
                the radius where :math:`\rho_{c}(r) = f_{c}^{\mathrm{fact}} \rho_v(r)`.
            2) Teter model core charge where the Teter parameters are adjusted such that
                the model core charge matches the all-electron core charge and its first
                derivative at the radius where :math:`\rho_{c}(r) = f_{c}^{\mathrm{fact}} \rho_v(r)`.
            3) Teter model core charge with fixed amplitude and scale factor (if `teter_relative=True`)
                or fixed amplitude and scale (if `teter_relative=False`).
            4) Optimized Teter model core charge where the Teter parameters are selected to minimize
                the RMSE in the 2nd derivative of the exchange-correlation energy w.r.t. the all-electron
                atom.
            5) (METAPSP only) Polynomial model core charge and kinetic energy densities, with amplitudes
                `fcfact` and `rcfact` times the maximum pseudo density at the origin and at the inner
                matching region, respectively.
            6) (azadoks/oncvpsp only) Budgeted Martins model core charge: the all-electron core
                charge beyond the smallest radius r_s at which a Martins function
                :math:`\exp(a + b r^2 + c r^4)` matched to it needs no more than the plane-wave
                cutoff `fcfact` (Ha) for an exchange-correlation energy error of `rcfact` (Ha).
                With `fcfact <= 0` the cutoff is instead that of the pseudo-waves for the same
                error, times `-fcfact` (times 1 for `fcfact == 0`). Newer builds fill inside r_s
                with the minimum-tail polynomial of PseudoPotentialGenerators.jl (`BudgetedCore`),
                and fall back to the Martins function where that polynomial is not monotone.
            7) (azadoks/oncvpsp only) Hardness-matched model core charge (`HardnessMatchedCore` of
                PseudoPotentialGenerators.jl): of the cores of icmod 6 at the radii from the
                smallest rc down to a little below the icmod 6 r_s, the one that fits the cutoff
                and whose exchange-correlation hardness of the pseudo valence states is closest to
                that of the all-electron core. Same `fcfact`, `rcfact` and `slack` as icmod 6.
        fcfact (float): Used to determine the matching radius for `icmod in (1, 2)`. Unused for `icmod in (0, 3, 4)`.
            Plane-wave cutoff (Ha) for `icmod in (6, 7)` if positive; else 0 for the cutoff of the
            pseudo-waves, or minus its multiple. Negative only for `icmod in (6, 7)`.
        rcfact (float): Exchange-correlation energy error (Ha) for `icmod in (6, 7)`, where 0 means the
            ONCVPSP default (1e-4 Ha up to f098dc9, 1e-5 Ha in the builds that read `slack`).
        slack (float): For `icmod in (6, 7)`, the fraction of the cutoff the model core must fit;
            the core is still made for the density cutoff of the whole cutoff. In (0, 1], default
            1. Older builds ignore it.
        teter_relative (bool): If True, `teter_amp*` and `teter_scale*` specify relative
            ranges for optimizing the Teter parameters. If False, they specify absolute ranges.
        teter_amp_min (None | float): Minimum Teter amplitude for grid search (relative or absolute).
        teter_amp_max (None | float): Maximum Teter amplitude for grid search (relative or absolute).
        teter_amp_step (None | float): Teter amplitude step size for grid search (relative or absolute).
        teter_scale_min (None | float): Minimum Teter scale factor for grid search (relative or absolute).
        teter_scale_max (None | float): Maximum Teter scale factor for grid search (relative or absolute).
        teter_scale_step (None | float): Teter scale factor step size for grid search (relative or absolute).
    """

    icmod: Annotated[int, Field(ge=0, le=7)]
    fcfact: Annotated[float, Field()] = 0.0
    rcfact: Annotated[float, Field(ge=0.0)] = 0.0
    slack: Annotated[float, Field(gt=0.0, le=1.0)] = 1.0
    teter_relative: Annotated[bool, Field()] = True
    teter_amp_min: None | Annotated[float, Field()] = None
    teter_amp_max: None | Annotated[float, Field()] = None
    teter_amp_step: None | Annotated[float, Field()] = None
    teter_scale_min: None | Annotated[float, Field()] = None
    teter_scale_max: None | Annotated[float, Field()] = None
    teter_scale_step: None | Annotated[float, Field()] = None

    @model_validator(mode="after")
    def _check(self) -> "ModelCoreChargeInput":
        if self.icmod in (1, 3):
            if self.fcfact == 0.0:
                raise ValueError("fcfact must be greater than 0 for icmod 1 and 3")
        if self.icmod == 3:
            if self.rcfact == 0.0:
                raise ValueError("rcfact must be greater than 0 for icmod 3")
        if self.icmod not in (6, 7) and self.fcfact < 0.0:
            raise ValueError("fcfact must not be negative except for icmod 6 and 7")
        if self.icmod not in (6, 7) and self.slack != 1.0:
            raise ValueError("slack is only read for icmod 6 and 7")
        return self

    def as_dat_string(self, with_comment: bool = False) -> str:
        """String representation of the pseudopotential input in .dat file format."""
        if not self.teter_relative:
            raise ValueError("Absolute Teter parameters not supported by text input")
        if self.icmod in (0, 1, 2):
            s = f"{self.icmod:>10d} {self.fcfact:>10.5f}"
            if with_comment:
                s = f"# {'icmod':>8s} {'fcfact':>10s}\n" + s
        elif self.icmod in (3, 5):
            s = f" {self.icmod:>10d} {self.fcfact:>10.5f} {self.rcfact:>10.5f}"
            if with_comment:
                s = f"# {'icmod':>8s} {'fcfact':>10s} {'rcfact':>10s}\n" + s
        elif self.icmod in (6, 7):
            # rcfact is an energy error (Ha), often below the resolution of %10.5f
            if self.slack != 1.0:
                # slack follows rcfact, so rcfact must be written; ONCVPSP rejects 0 there, so
                # the default of the builds that read slack
                rcfact = self.rcfact if self.rcfact > 0.0 else 1.0e-5
                s = f" {self.icmod:>10d} {self.fcfact:>10.5f} {rcfact:>12.5e} {self.slack:>10.5f}"
                if with_comment:
                    s = (
                        f"# {'icmod':>8s} {'fcfact':>10s} {'rcfact':>12s} {'slack':>10s}\n"
                        + s
                    )
            elif self.rcfact == 0.0:
                s = f"{self.icmod:>10d} {self.fcfact:>10.5f}"
                if with_comment:
                    s = f"# {'icmod':>8s} {'fcfact':>10s}\n" + s
            else:
                s = f" {self.icmod:>10d} {self.fcfact:>10.5f} {self.rcfact:>12.5e}"
                if with_comment:
                    s = f"# {'icmod':>8s} {'fcfact':>10s} {'rcfact':>12s}\n" + s
        else:
            teter_grid_params = (
                self.teter_amp_min,
                self.teter_amp_max,
                self.teter_amp_step,
                self.teter_scale_min,
                self.teter_scale_max,
                self.teter_scale_step,
            )
            if all(v is None for v in teter_grid_params):
                s = f"{self.icmod:>10d} {self.fcfact:>10.5f}"
                if with_comment:
                    s = f"# {'icmod':>8s} {'fcfact':>10s}\n" + s
            else:
                s = (
                    f" {self.icmod:>10d} {self.fcfact:>10.5f} {self.rcfact:>10.5f} "
                    f"{self.teter_amp_min:>10.5f} {self.teter_amp_max:>10.5f} {self.teter_amp_step:>10.5f} "
                    f"{self.teter_scale_min:>10.5f} {self.teter_scale_max:>10.5f} {self.teter_scale_step:>10.5f}"
                )
                if with_comment:
                    s = (
                        f"# {'icmod':>8s} {'fcfact':>10s} {'rcfact':>10s} "
                        f"{'fcfact_min':>10s} {'fcfact_max':>10s} {'fcfact_step':>10s} "
                        f"{'rcfact_min':>10s} {'rcfact_max':>10s} {'rcfact_step':>10s}\n"
                        + s
                    )
        return s


class LogDerivativeInput(BaseModel):  # pylint: disable=too-few-public-methods
    """Input parameters controlling the calculation of logarithmic derivatives.

    Attributes:
        epsh1 (float): [Hartree] Minimum energy for logarithmic derivative calculation.
        epsh2 (float): [Hartree] Maximum energy for logarithmic derivative calculation.
        depsh (float): [Hartree] Energy step size for logarithmic derivative calculation.
        rxpsh (None | float): [Bohr] Radius at which logarithmic derivatives are calculated.
            If `None`, the radius is chosen automatically at a few grid points beyond the largest
            core radius used in the pseudopotential construction.
    """

    epsh1: float
    epsh2: float
    depsh: Annotated[float, Field(gt=0.0)]
    rxpsh: None | Annotated[float, Field(gt=0.0)] = None

    @model_validator(mode="after")
    def _check(self) -> "LogDerivativeInput":
        if self.epsh2 <= self.epsh1:
            raise ValueError("e_max must be greater than e_min")
        return self

    def as_dat_string(self, with_comment: bool = False) -> str:
        """String representation of the pseudopotential input in .dat file format."""
        if self.rxpsh is None:
            s = f"{self.epsh1:>10.5f} {self.epsh2:>10.5f} {self.depsh:>10.5f}"
            if with_comment:
                s = f"# {'epsh1':>8s} {'epsh2':>10s} {'depsh':>10}\n" + s
        else:
            warnings.warn(
                "`rxpsh` is not supported by all versions of ONCVPSP! Check that expected behavior is correct."
            )
            s = f"{self.epsh1:>10.5f} {self.epsh2:>10.5f} {self.depsh:>10.5f} {self.rxpsh:>10.5f}"
            if with_comment:
                s = (
                    f"# {'epsh1':>8s} {'epsh2':>10s} {'depsh':>10s} {'rxpsh':>10s}\n"
                    + s
                )
        return s


class OutputGridInput(BaseModel):  # pylint: disable=too-few-public-methods
    """Input parameters controlling the linear mesh (starting from `r=0`)
    used for writing the output pseudopotential file(s).

    Attributes:
        rlmax (float): [Bohr] Maximum radius of the output grid (exclusive).
        drl (float): [Bohr] Spacing of the output grid.
        upf_mesh (int): Most points of the UPF file on the log mesh: if positive, the UPF file
            is written on every `istr`-th point of the log mesh out to `rlmax`, `istr` the
            smallest stride leaving at most `upf_mesh` points; 0 for the linear output grid.
            The optional third value of the line, or `upf_mesh` in `[pp_output]` of the TOML
            input (azadoks/oncvpsp). QE reads at most 3500 points.
    """

    rlmax: Annotated[float, Field(gt=0.0)]
    drl: Annotated[float, Field(gt=0.0)]
    upf_mesh: Annotated[int, Field(ge=0)] = 0

    def as_dat_string(self, with_comment: bool = False) -> str:
        """String representation of the pseudopotential input in .dat file format, with
        `upf_mesh` only when it is not 0."""
        s = f"{self.rlmax:>10.5f} {self.drl:>10.5f}"
        if self.upf_mesh:
            s += f" {self.upf_mesh:>10d}"
        if with_comment:
            upf_mesh = f" {'upf_mesh':>10s}" if self.upf_mesh else ""
            s = f"# {'rlmax':>8s} {'drl':>10s}{upf_mesh}\n" + s
        return s


class LogMeshInput(BaseModel):  # pylint: disable=too-few-public-methods
    """Input parameters of the log radial mesh `rr(i) = rr1 * amesh**(i - 1)`, out to `rmax`, on
    which ONCVPSP solves the atom: the optional last line of the .dat input (after the test
    configurations) or the `[log_mesh]` table of the TOML input, which azadoks/oncvpsp reads.
    ONCVPSP uses its built-in value for each one that is not given, and takes a value of 0 or
    less to mean the same, which is read here as `None`.

    Attributes:
        amesh (None | float): Ratio of successive mesh points. If `None`, 1.006.
        rr1 (None | float): [Bohr] First mesh point. If `None`, `0.0005 / z`, capped at
            `0.00005` except in the fully-relativistic code.
        rmax (None | float): [Bohr] Outer radius of the mesh. If `None`, 45.
    """

    amesh: None | Annotated[float, Field(gt=1.0)] = None
    rr1: None | Annotated[float, Field(gt=0.0)] = None
    rmax: None | Annotated[float, Field(gt=0.0)] = None

    @field_validator("amesh", "rr1", "rmax", mode="before")
    @classmethod
    def _nonpositive_is_default(cls, value: Any) -> Any:
        if isinstance(value, (int, float)) and value <= 0:
            return None
        return value

    @property
    def is_set(self) -> bool:
        """Whether any of the mesh parameters is given, so that the input sets the mesh."""
        return any(v is not None for v in (self.amesh, self.rr1, self.rmax))

    def as_dat_string(self, with_comment: bool = False) -> str:
        """String representation of the log mesh input in .dat file format, with -1 for the
        parameters that are not given."""
        values = (-1.0 if v is None else v for v in (self.amesh, self.rr1, self.rmax))
        s = " ".join(f"{v!r:>10s}" for v in values)
        if with_comment:
            s = f"# {'amesh':>8s} {'rr1':>10s} {'rmax':>10s}\n" + s
        return s


# Keys of the Teter grid of icmod 4 in the TOML input of ONCVPSP, by field name
_TOML_TETER_GRID_KEYS = {
    "teter_amp_min": "fcfact_min",
    "teter_amp_max": "fcfact_max",
    "teter_amp_step": "fcfact_step",
    "teter_scale_min": "rcfact_min",
    "teter_scale_max": "rcfact_max",
    "teter_scale_step": "rcfact_step",
}


def _to_oncvpsp_toml(data: dict[str, Any]) -> dict[str, Any]:
    """Arrange the fields of `OncvpspInput.model_dump()` as the TOML input of ONCVPSP
    (`oncvpsp.x -t`, src/input_toml_m.f90): `psfile` in `[pp_output]`, the output grid as
    `[linear_mesh]` `a` (drl) and `rmax` (rlmax) with `upf_mesh` in `[pp_output]`, and the icmod
    4 Teter grid as `fcfact_*` and `rcfact_*`. ONCVPSP ignores the keys it does not read, such as
    `teter_relative`. The log mesh parameters that are not given are left out, and `[log_mesh]`
    too when none is; so are `c_light` when it is not given and `upf_mesh` when it is 0."""
    data = deepcopy(data)
    data["pp_output"] = {"psfile": data["oncvpsp"].pop("psfile")}
    if data["oncvpsp"].get("c_light", 0) is None:
        del data["oncvpsp"]["c_light"]
    log_mesh = {k: v for k, v in data.pop("log_mesh", {}).items() if v is not None}
    if log_mesh:
        data["log_mesh"] = log_mesh
    output_grid = data.pop("output_grid")
    data["linear_mesh"] = {"a": output_grid["drl"], "rmax": output_grid["rlmax"]}
    if output_grid.get("upf_mesh"):
        data["pp_output"]["upf_mesh"] = output_grid["upf_mesh"]
    core = data["model_core_charge"]
    for field, key in _TOML_TETER_GRID_KEYS.items():
        value = core.pop(field, None)
        if value is not None:
            core[key] = value
    return data


def _from_oncvpsp_toml(data: dict[str, Any]) -> dict[str, Any]:
    """Arrange the TOML input of ONCVPSP as the fields of `OncvpspInput`; the inverse of
    `_to_oncvpsp_toml`. Data already arranged as the fields (as pyoncvpsp wrote TOML before)
    are returned as they are."""
    data = deepcopy(data)
    upf_mesh = None
    if "pp_output" in data:
        pp_output = data.pop("pp_output")
        data.setdefault("oncvpsp", {})["psfile"] = pp_output["psfile"]
        upf_mesh = pp_output.get("upf_mesh")
    if "linear_mesh" in data:
        linear_mesh = data.pop("linear_mesh")
        data["output_grid"] = {"rlmax": linear_mesh["rmax"], "drl": linear_mesh["a"]}
    if upf_mesh is not None:
        data.setdefault("output_grid", {})["upf_mesh"] = upf_mesh
    core = data.get("model_core_charge", {})
    for field, key in _TOML_TETER_GRID_KEYS.items():
        if key in core:
            core[field] = core.pop(key)
    return data


class OncvpspInput(BaseModel):
    """ONCVPSP Input parameters"""

    oncvpsp: OncvpspAtomInput
    reference_configuration: AtomicConfigurationInput
    pseudopotentials: PseudoPotentialsInput
    local_potential: LocalPotentialInput
    vkb_projectors: VKBProjectorsInput
    model_core_charge: ModelCoreChargeInput
    log_derivative_analysis: LogDerivativeInput
    test_configurations: list[AtomicConfigurationInput] = []
    output_grid: OutputGridInput
    log_mesh: LogMeshInput = Field(default_factory=LogMeshInput)

    @property
    def radial_mesh(self) -> np.ndarray:
        """Radial grid used for pseudopotential construction and output: the log mesh of the
        input, with the defaults of the non- and scalar-relativistic codes.

        Returns:
            np.ndarray: Radial grid points in Bohr.
        """
        return build_mesh(
            self.oncvpsp.z,
            amesh=self.log_mesh.amesh,
            rr1=self.log_mesh.rr1,
            rmax=self.log_mesh.rmax,
        )

    @property
    def rxpsh(self) -> list[float]:
        """Radii at which logarithmic derivatives are calculated for each angular momentum channel.

        Returns:
            list[float]: List of radii in Bohr.
        """
        if self.log_derivative_analysis.rxpsh is not None:
            return [self.log_derivative_analysis.rxpsh] * 4
        else:
            # The main program rounds all rc values to 5 decimal places before using them
            # `run_phsft(_r)` uses `irc(l+1) + 2` for l <= lmax and `irc(lloc+1)` otherwise
            radial_mesh = self.radial_mesh
            rcloc = (
                self.local_potential.rcloc
                if self.local_potential.lloc == 4
                else self.pseudopotentials.rc[self.local_potential.lloc]
            )
            assert rcloc is not None  # validated to be set when lloc == 4
            rcloc_rounded = floor(1e5 * rcloc) / 1e5
            rxpsh_default = radial_mesh[radial_mesh >= rcloc_rounded][0]
            rxpsh = [rxpsh_default] * 4
            for l in range(self.pseudopotentials.lmax + 1):  # noqa: E741
                rc_rounded = floor(1e5 * self.pseudopotentials.rc[l]) / 1e5
                rxpsh[l] = radial_mesh[radial_mesh >= rc_rounded][2]
            return rxpsh

    def merge(self, other: dict[str, Any] | None = None, **kwargs) -> "OncvpspInput":
        """Merge this set of input parameters with another set of parameters or keyword arguments.
        Keys in `other` and `kwargs` will override those in `self`.
        Keys in `kwargs` take precedence over both `other` and `self`.

        Returns:
            OncvpspInput: Merged ONCVPSP input parameters.
        """
        data = self.model_dump()
        merged = recursive_merge(data, other or {})
        merged = recursive_merge(merged, kwargs)
        return OncvpspInput(**merged)

    def as_dat_string(self, with_comment=True) -> str:
        """Write the input to a string in the format expected by ONCVPSP for stdin input.

        Args:
            with_comment (bool, optional): Add explanatory comments to the input string. Defaults to True.

        Returns:
            str: ONCVPSP stdin string
        """
        log_mesh: list[str] = []
        if self.log_mesh.is_set:
            log_mesh = [self.log_mesh.as_dat_string(with_comment=with_comment)]
            if with_comment:
                log_mesh = ["#", "# LOG MESH", *log_mesh]
        if with_comment:
            return "\n".join(
                [
                    "# ATOM AND REFERENCE CONFIGURATION",
                    self.oncvpsp.as_dat_string(with_comment=with_comment),
                    "#",
                    self.reference_configuration.as_dat_string(
                        with_comment=with_comment
                    ),
                    "#",
                    "# PSEUDOPOTENTIAL AND OPTIMIZATION",
                    self.pseudopotentials.as_dat_string(with_comment=with_comment),
                    "#",
                    "# LOCAL POTENTIAL",
                    self.local_potential.as_dat_string(with_comment=with_comment),
                    "#",
                    "# VANDERBILT-KLEINMAN-BYLANDER PROJECTORS",
                    self.vkb_projectors.as_dat_string(with_comment=with_comment),
                    "#",
                    "# MODEL CORE CHARGE",
                    self.model_core_charge.as_dat_string(with_comment=with_comment),
                    "#",
                    "# LOG DERIVATIVE ANALYSIS",
                    self.log_derivative_analysis.as_dat_string(
                        with_comment=with_comment
                    ),
                    "#",
                    "# OUTPUT GRID",
                    self.output_grid.as_dat_string(with_comment=with_comment),
                    "#",
                    "# TEST CONFIGURATIONS",
                    f"# {'ncnf':>8s}",
                    f"{len(self.test_configurations):>10d}",
                ]
                + [
                    f"#\n# {'nvcnf':>8s}\n{len(config):>10d}\n"
                    + config.as_dat_string(with_comment=with_comment)
                    for config in self.test_configurations
                ]
                + log_mesh
            )
        return "\n".join(
            [
                self.oncvpsp.as_dat_string(with_comment=with_comment),
                self.reference_configuration.as_dat_string(with_comment=with_comment),
                self.pseudopotentials.as_dat_string(with_comment=with_comment),
                self.local_potential.as_dat_string(with_comment=with_comment),
                self.vkb_projectors.as_dat_string(with_comment=with_comment),
                self.model_core_charge.as_dat_string(with_comment=with_comment),
                self.log_derivative_analysis.as_dat_string(with_comment=with_comment),
                self.output_grid.as_dat_string(with_comment=with_comment),
                f"{len(self.test_configurations):>10d}",
            ]
            + [
                f"{len(config):>10d}\n"
                + config.as_dat_string(with_comment=with_comment)
                for config in self.test_configurations
            ]
            + log_mesh
        )

    @classmethod
    def from_dat_file(cls, file: str | PathLike | TextIO) -> "OncvpspInput":
        """Load the input from an ONCVPSP dat file.

        Args:
            file (PathLike | TextIO): Path to ONCVPSP dat file or file-like object.
        Returns:
            OncvpspInput: ONCVPSP input parameters.
        """
        if isinstance(file, (str, PathLike)):
            with open(file, "r", encoding="utf-8") as f:
                return cls.from_dat_string(f.read())
        else:
            return cls.from_dat_string(file.read())

    def to_dat_file(
        self, file: str | PathLike | TextIO, with_comment: bool = True
    ) -> None:
        """Write the input to an ONCVPSP dat file.

        Args:
            file (PathLike | TextIO): Path to ONCVPSP dat file or file-like object.
            with_comment (bool, optional): Add explanatory comments to the input string. Defaults to True.
        """
        s = self.as_dat_string(with_comment=with_comment)
        if isinstance(file, (str, PathLike)):
            with open(file, "w", encoding="utf-8") as f:
                f.write(s)
        else:
            file.write(s)

    @classmethod
    def from_dat_string(cls, s: str) -> "OncvpspInput":  # pylint: disable=too-many-locals,too-many-statements
        """Load the input from an ONCVPSP stdin string.

        Args:
            s (str): Input in ONCVPSP stdin format

        Returns:
            OncvpspInput: ONCVPSP input parameters.
        """
        lines: list[str] = s.strip().splitlines()
        # Strip whitespace
        lines = [line.strip() for line in lines]
        # Remove empty lines
        lines = [line for line in lines if line]
        # Remove empty and comment lines
        lines = [line for line in lines if not line.startswith("#")]
        # Remove inline comments
        lines = [line.split("#", 1)[0].strip() for line in lines]
        iter_: Iterator[str] = iter(lines)
        # ATOM AND REFERENCE CONFIGURATION
        # atsym, z, nc, nv, iexc, psfile, (c_light)
        words: list[str] = next(iter_).split()
        # ONCVPSP echoes the input with format (a2,f6.2), which runs together e.g. "Og118.00"
        if symbol_match := re.fullmatch(r"([A-Za-z]{1,2})([-+.\d].*)", words[0]):
            words = [symbol_match.group(1), symbol_match.group(2), *words[1:]]
        atsym = words[0]
        z = float(words[1])
        nc = int(words[2])
        nv = int(words[3])
        iexc = int(words[4])
        psfile = words[5]
        c_light = float(words[6].lower().replace("d", "e")) if len(words) >= 7 else None
        # n, l, f (nc + nv lines)
        na: list[int] = [0] * (nc + nv)
        la: list[int] = [0] * (nc + nv)
        fa: list[float] = [0.0] * (nc + nv)
        for i in range(nc + nv):
            words = next(iter_).split()
            na[i] = int(words[0])
            la[i] = int(words[1])
            fa[i] = float(words[2])
        # PSEUDOPOTENTIAL AND OPTIMIZATION
        # lmax
        lmax = int(next(iter_).split()[0])
        # l, rc, ep, ncon, nbas, qcut (lmax + 1 lines, l's must be in order)
        rc: list[float] = [0.0] * (lmax + 1)
        ep: list[float] = [0.0] * (lmax + 1)
        ncon: list[int] = [0] * (lmax + 1)
        nbas: list[int] = [0] * (lmax + 1)
        qcut: list[float] = [0.0] * (lmax + 1)
        for ell in range(lmax + 1):
            words = next(iter_).split()
            assert ell == int(words[0])
            rc[ell] = float(words[1])
            ep[ell] = float(words[2])
            ncon[ell] = int(words[3])
            nbas[ell] = int(words[4])
            qcut[ell] = float(words[5])
        # LOCAL POTENTIAL
        # lloc, lpopt, rc(5), dvloc0
        words = next(iter_).split()
        lloc = int(words[0])
        lpopt = int(words[1])
        rc5 = float(words[2])
        dvloc0 = float(words[3])
        # VANDERBILT-KLEINMAN-BYLANDER PROJECTORs
        # nproj, debl (lmax + 1 lines, l's must be in order)
        nproj: list[int] = [0] * (lmax + 1)
        debl: list[float] = [0.0] * (lmax + 1)
        for ell in range(lmax + 1):
            words = next(iter_).split()
            assert ell == int(words[0])
            nproj[ell] = int(words[1])
            debl[ell] = float(words[2])
        # MODEL CORE CHARGE
        # icmod, fcfact, (rcfact), (fcfact_min, fcfact_max, fcfact_step, rcfact_min, rcfact_max, rcfact_step)
        words = next(iter_).split()
        icmod = int(words[0])
        fcfact = float(words[1])
        rcfact: float = 0.0 if len(words) < 3 else float(words[2])
        slack: float = float(words[3]) if icmod in (6, 7) and len(words) >= 4 else 1.0
        teter_grid: dict[str, float] = {}
        if icmod == 4 and len(words) >= 9:
            teter_grid = dict(
                zip(
                    (
                        "teter_amp_min",
                        "teter_amp_max",
                        "teter_amp_step",
                        "teter_scale_min",
                        "teter_scale_max",
                        "teter_scale_step",
                    ),
                    (float(w) for w in words[3:9]),
                )
            )
        # LOG DERIVATIVE ANALYSIS
        # epsh1, epsh2, depsh, (rxpsh)
        words = next(iter_).split()
        epsh1 = float(words[0])
        epsh2 = float(words[1])
        depsh = float(words[2])
        rxpsh = None if len(words) < 4 else float(words[3])
        # OUTPUT GRID
        # rlmax, drl, (upf_mesh)
        words = next(iter_).split()
        rlmax = float(words[0])
        drl = float(words[1])
        upf_mesh = int(words[2]) if len(words) >= 3 else 0
        # TEST CONFIGURATIONS
        # ncnf
        ncnf = int(next(iter_).strip())
        # nvcnf (repeated ncnf times)
        # n, l, f (nvcnf lines, repeated following ncnf's ncnf times)
        test_configurations: list[dict[str, list[int | float]]] = []
        nvcnf: list[int] = [0] * ncnf
        for i in range(ncnf):
            nvcnf[i] = int(next(iter_).strip())
            test_configurations.append(
                {"n": [0] * nvcnf[i], "l": [0] * nvcnf[i], "f": [0.0] * nvcnf[i]}
            )
            for j in range(nvcnf[i]):
                words = next(iter_).split()
                test_configurations[i]["n"][j] = int(words[0])
                test_configurations[i]["l"][j] = int(words[1])
                test_configurations[i]["f"][j] = float(words[2])
        # LOG MESH (optional; azadoks/oncvpsp)
        # amesh, rr1, rmax
        # In an output, the echo of the input is followed by the results, which are not a mesh
        log_mesh: dict[str, float] = {}
        if (line := next(iter_, None)) is not None:
            words = line.split()
            try:
                values = [float(w.lower().replace("d", "e")) for w in words]
            except ValueError:
                values = []
            if len(values) == 3:
                log_mesh = dict(zip(("amesh", "rr1", "rmax"), values))
        data = {
            "oncvpsp": {
                "atsym": atsym,
                "z": z,
                "nc": nc,
                "nv": nv,
                "iexc": iexc,
                "psfile": psfile,
                "c_light": c_light,
            },
            "reference_configuration": {
                "n": na,
                "l": la,
                "f": fa,
            },
            "pseudopotentials": {
                "lmax": lmax,
                "l": list(range(lmax + 1)),
                "rc": rc,
                "ep": ep,
                "ncon": ncon,
                "nbas": nbas,
                "qcut": qcut,
            },
            "local_potential": {
                "lloc": lloc,
                "lpopt": lpopt,
                "rcloc": rc5,
                "dvloc0": dvloc0,
            },
            "vkb_projectors": {
                "l": list(range(lmax + 1)),
                "nproj": nproj,
                "debl": debl,
            },
            "model_core_charge": {
                "icmod": icmod,
                "fcfact": fcfact,
                "rcfact": rcfact,
                "slack": slack,
                **teter_grid,
            },
            "log_derivative_analysis": {
                "epsh1": epsh1,
                "epsh2": epsh2,
                "depsh": depsh,
                "rxpsh": rxpsh,
            },
            "output_grid": {
                "rlmax": rlmax,
                "drl": drl,
                "upf_mesh": upf_mesh,
            },
            "test_configurations": test_configurations,
            "log_mesh": log_mesh,
        }
        return cls(**data)  # type: ignore

    def as_dict(self, flat=False) -> dict[str, Any]:
        """Return the ONCVPSP input parameters as a nested dictionary.

        Returns:
            dict[str, Any]: ONCVPSP input parameters.
        """
        dict_ = self.model_dump()
        if flat:
            dict_ = flatten(dict_)
        return dict_

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OncvpspInput":
        """Create an OncvpspInput instance from a nested dictionary."""
        return cls(**data)

    def as_toml_string(self) -> str:
        """Return the ONCVPSP input parameters as a TOML string, in the layout that ONCVPSP
        reads with `-t`.

        Returns:
            str: ONCVPSP input parameters in TOML format.
        """
        return toml.dumps(_to_oncvpsp_toml(self.model_dump()))

    @classmethod
    def from_toml_string(cls, s: str) -> "OncvpspInput":
        """Create an OncvpspInput instance from a TOML string.

        Args:
            s (str): ONCVPSP input parameters in TOML format.
        Returns:
            OncvpspInput: ONCVPSP input parameters.
        """
        data = toml.loads(s)
        return cls(**_from_oncvpsp_toml(data))

    def to_toml_file(self, file: str | PathLike | TextIO) -> None:
        """Write the ONCVPSP input parameters to a TOML file, in the layout that ONCVPSP reads
        with `-t`.

        Args:
            file (PathLike | TextIO): Path to TOML file or file-like object.
        """
        data = _to_oncvpsp_toml(self.model_dump())
        if isinstance(file, (str, PathLike)):
            with open(file, "w", encoding="utf-8") as fp:
                toml.dump(data, fp)
        else:
            toml.dump(data, file)

    @classmethod
    def from_toml_file(cls, file: str | PathLike | TextIO) -> "OncvpspInput":
        """Load the ONCVPSP input parameters from a TOML file.

        Args:
            file (PathLike | TextIO): Path to TOML file or file-like object.
        Returns:
            OncvpspInput: ONCVPSP input parameters.
        """
        if isinstance(file, (str, PathLike)):
            with open(file, "r", encoding="utf-8") as fp:
                data = toml.load(fp)
        else:
            data = toml.load(file)
        return cls(**_from_oncvpsp_toml(data))

    @model_validator(mode="after")
    def _check_number_of_states(self) -> "OncvpspInput":
        if len(self.reference_configuration) != self.oncvpsp.nc + self.oncvpsp.nv:
            raise ValueError(
                f"Number of states in reference configuration ({len(self.reference_configuration)}) "
                f"does not match n_core + n_valence ({self.oncvpsp.nc} + {self.oncvpsp.nv} = "
                f"{self.oncvpsp.nc + self.oncvpsp.nv})"
            )
        if not 1 <= (self.oncvpsp.nc + self.oncvpsp.nv) <= 30:
            raise ValueError(
                f"Total number of states (n_core + n_valence = {self.oncvpsp.nc + self.oncvpsp.nv}) "
                f"must be between 1 and 30"
            )
        return self

    @model_validator(mode="after")
    def _check_reference_configuration(self) -> "OncvpspInput":
        total_occupation = sum(state.f for state in self.reference_configuration)
        if total_occupation > self.oncvpsp.z:
            raise ValueError(
                "The reference configuration is a negative ion: "
                f"total occupation {total_occupation} > atomic charge {self.oncvpsp.z}"
            )
        return self

    @model_validator(mode="after")
    def _check_pseudo_potentials(self) -> "OncvpspInput":
        if self.local_potential.lloc == 4:
            for psp in self.pseudopotentials:
                if psp.rc < self.local_potential.rcloc:  # type: ignore
                    raise ValueError(
                        f"Pseudopotential with l={psp.l} has rc={psp.rc} "
                        f"smaller than local potential rc={self.local_potential.rcloc}"
                    )
        return self

    @model_validator(mode="after")
    def _check_output_grid(self) -> "OncvpspInput":
        rc_max: float = max(self.pseudopotentials.rc)
        if self.local_potential.lloc == 4:
            rc_max = max(rc_max, self.local_potential.rcloc)  # type: ignore
        if self.output_grid.rlmax < rc_max:
            raise ValueError(
                f"Output grid rlmax={self.output_grid.rlmax} is smaller than "
                f"maximum rc={rc_max} of pseudopotentials"
            )
        return self

    @model_validator(mode="after")
    def _check_log_mesh(self) -> "OncvpspInput":
        # the checks of check_data in ONCVPSP
        rc_max: float = max(self.pseudopotentials.rc)
        if self.local_potential.lloc == 4:
            rc_max = max(rc_max, self.local_potential.rcloc)  # type: ignore
        if self.log_mesh.rr1 is not None and self.log_mesh.rr1 >= rc_max:
            raise ValueError(
                f"Log mesh rr1={self.log_mesh.rr1} is not smaller than "
                f"maximum rc={rc_max} of pseudopotentials"
            )
        if (
            self.log_mesh.rmax is not None
            and self.log_mesh.rmax <= self.output_grid.rlmax
        ):
            raise ValueError(
                f"Log mesh rmax={self.log_mesh.rmax} is not larger than "
                f"output grid rlmax={self.output_grid.rlmax}"
            )
        return self
