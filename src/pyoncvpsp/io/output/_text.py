"""Parser for the ONCVPSP/METAPSP stdout text output."""
# pylint: disable=too-many-lines

from collections import defaultdict
from functools import cached_property
from os import PathLike
from typing import TextIO, Any
import pathlib as pl
import re

import numpy as np

from pyoncvpsp.channels import ChannelKey, ProjectorKey, j_channels
from pyoncvpsp.io._utils import fort_float
from pyoncvpsp.projectors import (
    ProjectorLayout,
    ProjectorSlot,
    Solver,
    predict_projector_layout,
)
from pyoncvpsp.io.input._models import OncvpspInput
from ._data import WARNINGS, ERRORS
from ._messages import OncvpspFailure, OncvpspMessage, Stage
from . import _attribution, _sections
from ._sections import Block, BlockKind, LineContext

RE_FLOAT: str = r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][+-]?\d+)?"


class OncvpspOutputError(Exception):
    """Error message from ONCVPSP."""

    line_numbers: list[int]
    context_lines: list[str]
    context_line_numbers: list[int]
    name: str
    subroutine: str
    description: str

    def __init__(
        self,
        line_numbers: list[int],
        context_lines: list[str],
        context_line_numbers: list[int],
        name: str,
        subroutine: str,
        description: str,
    ) -> None:
        self.line_numbers = line_numbers
        self.context_lines = context_lines
        self.context_line_numbers = context_line_numbers
        self.name = name
        self.subroutine = subroutine
        self.description = description
        message = (
            f"ONCVPSP error encountered at line {line_numbers[0]}:\n"
            f"  Subroutine: {subroutine}\n"
            f"  Description: {description}\n"
            "  Context:\n"
            + "\n".join(
                f"\t{ln:>6d} {'>' if ln in line_numbers else ' '}|{line}"
                for ln, line in zip(context_line_numbers, context_lines)
            )
        )
        super().__init__(message)

    @property
    def error_lines(self) -> list[str]:
        """Lines which contain the error.

        Returns:
            list[str]: Lines containing the error.
        """
        return [
            line
            for ln, line in zip(self.context_line_numbers, self.context_lines)
            if ln in self.line_numbers
        ]


class OncvpspTextParser:
    """Parser for ONCVPSP/METAPSP stdout text output."""

    _lines: list[str]

    def __init__(
        self,
        path: str | PathLike | None = None,
        io: TextIO | None = None,
        text: str | None = None,
        lines: list[str] | None = None,
        check: bool = True,
        werror: bool = False,
    ) -> None:
        sources = iter(x is not None for x in [path, io, text, lines])
        # First `any`: consume the iteratble until the first True or it is exhausted
        # Second `any`: check if there are any remaining True values in the iterator
        assert any(sources) and not any(sources), (
            "Exactly one input source must be provided."
        )
        if text is not None:
            lines = text.splitlines(keepends=False)
        elif path is not None:
            path = pl.Path(path)
            with path.open("r", encoding="latin-1") as f:
                lines = f.readlines()
        elif io is not None:
            lines = io.readlines()
        elif lines is None:
            raise RuntimeError("unreachable")
        self._lines = [line.rstrip() for line in lines]
        if check:
            self.check(werror=werror)

    def _parse_errors(
        self, definitions: list[dict], context_window: int | tuple[int, int] = 3
    ) -> list[dict]:
        _context_window = (
            (context_window, context_window)
            if isinstance(context_window, int)
            else context_window
        )
        patterns = [re.compile(definition["pattern"]) for definition in definitions]
        errors = []
        for i, line in enumerate(self._lines):
            line_number = i + 1
            for definition, pattern in zip(definitions, patterns):
                # The substring check is a fast pre-filter; the pattern distinguishes similar messages
                if definition["match_string"] in line and pattern.search(line):
                    line_numbers = range(
                        line_number, line_number + definition.get("line_count", 1)
                    )
                    context_start = max(0, i - _context_window[0])
                    context_end = min(len(self._lines), i + _context_window[1] + 1)
                    context_lines = self._lines[context_start:context_end]
                    context_line_numbers = list(
                        range(context_start + 1, context_end + 1)
                    )
                    warning = {
                        "line_numbers": list(line_numbers),
                        "context_lines": context_lines,
                        "context_line_numbers": context_line_numbers,
                        "name": definition.get("name", "unknown_error"),
                        "subroutine": definition.get("subroutine", "Unknown"),
                        "description": definition.get(
                            "description", "No description available."
                        ),
                    }
                    errors.append(warning)
        return errors

    def _get_lines_between(
        self,
        start_pattern: str | re.Pattern[str],
        stop_pattern: str | re.Pattern[str],
        include_start: bool = False,
        include_stop: bool = False,
    ) -> list[str]:
        start_match = self.search(start_pattern)
        if not start_match:
            return []
        start_idx = start_match[0] + (0 if include_start else 1)
        stop_match = self.search(stop_pattern, start=start_idx)
        if not stop_match:
            return []
        end_idx = stop_match[0] + (1 if include_stop else 0)
        return self.clean_lines[start_idx:end_idx]

    def search(
        self, pattern: str | re.Pattern[str], start: int = 0
    ) -> tuple[int, re.Match[str]] | None:
        """Find the line in `self.lines` which matches the regex `pattern` starting from `self.lines[start]`.

        Args:
            pattern (str | re.Pattern[str]): Pattern to search for.
            start (int, optional): Index of line to start searching from. Defaults to 0.

        Returns:
            tuple[int, re.Match[str]] | None: Tuple of (line index, regex match object) if found, else None.
        """
        regex = re.compile(pattern) if isinstance(pattern, str) else pattern
        for i in range(start, len(self.clean_lines)):
            if match := regex.search(self.clean_lines[i]):
                return i, match
        return None

    def findall(
        self, pattern: str | re.Pattern[str], start: int = 0, stop: int = -1
    ) -> list[tuple[int, re.Match[str]]]:
        """Find all lines in `self.lines[start:stop]` which match the regex `pattern`.

        Args:
            pattern (str | re.Pattern[str]): Pattern to serch for.
            start (int, optional): Lower bound of search window (inclusive). Defaults to 0.
            stop (int, optional): Upper bound of search window (exclusive). Defaults to -1.

        Returns:
            list[tuple[int, re.Match[str]]]: List of tuples of (line index, regex match object).
        """
        regex = re.compile(pattern) if isinstance(pattern, str) else pattern
        stop = min(stop, len(self.clean_lines)) if stop >= 0 else len(self.clean_lines)
        matches = []
        for i in range(start, stop):
            if match := regex.search(self.clean_lines[i]):
                matches.append((i, match))
        return matches

    @cached_property
    def warnings(self) -> list[dict]:
        """List of warning summaries.

        Returns:
            list[dict]: List of warnings encountered in the output.
        """
        return self._parse_errors(WARNINGS)

    @cached_property
    def errors(self) -> list[dict]:
        """List of error summaries.

        Returns:
            list[dict]: List of errors encountered in the output.
        """
        return self._parse_errors(ERRORS)

    def check(
        self, doraise: bool = True, werror: bool = False
    ) -> list[OncvpspOutputError]:
        """Check the output for errors and warnings (if `werror`).
        Raise an `ExceptionGroup` containing `OncvpspOutputError`s, if found.

        Args:
            doraise (bool, optional): Raise an `ExceptionGroup` if any errors are found. Defaults to True.
            werror (bool, optional): Raise errors for any warnings encountered. Defaults to False.

        Returns:
            list[OncvpspOutputError]: Errors (and warnings, if `werror`) found in the output.

        Raises:
            ExceptionGroup: Group of `OncvpspOutputError`s describing errors and warnings encountered.
        """
        exceptions = []
        # Only give 'incomplete output' error when the program did not report any other errors
        # but is nonetheless unfinished (e.g. still running, killed, etc.)
        if not self.errors and not self.completed:
            exceptions.append(
                OncvpspOutputError(
                    line_numbers=[len(self._lines)],
                    context_lines=self._lines[-5:],
                    context_line_numbers=list(
                        range(len(self._lines) - 4, len(self._lines) + 1)
                    ),
                    name="incomplete_output",
                    subroutine="main",
                    description="Missing or truncated plotting data: "
                    + ", ".join(self.missing_plot_data),
                )
            )
        exceptions.extend([OncvpspOutputError(**error) for error in self.errors])
        if werror:
            exceptions.extend(
                [OncvpspOutputError(**warning) for warning in self.warnings]
            )
        if exceptions and doraise:
            raise ExceptionGroup(
                f"Found {len(exceptions)} error(s) in ONCVPSP output.",
                exceptions,
            )
        return exceptions

    @cached_property
    def completed(self) -> bool:
        """Whether ONCVPSP/METAPSP successfully pseudized the atom and wrote all plotting data."""
        return not self.missing_plot_data

    @cached_property
    def missing_plot_data(self) -> list[str]:
        """Plotting data sections which are missing or truncated, e.g. because ONCVPSP/METAPSP
        stopped or was killed while writing them. The pseudopotential file is not considered,
        as its output may be disabled.

        Returns:
            list[str]: Names of the missing or truncated sections; empty if all were written.
        """
        start_match = self.search("DATA FOR PLOTTING")
        if not start_match:
            return ["DATA FOR PLOTTING"]
        start = start_match[0]
        sections = {
            "unscreened_semilocal_potentials": r"^!p\s",
            "charge_densities": r"^!r\s",
            "wavefunctions": r"^&\s",
            # VKB projectors are not checked: e.g. ONCVPSP 3.2.2 omits them when all nproj = 1
            "convergence_profiles": r"^!C\s",
        }
        missing = [
            name
            for name, pattern in sections.items()
            if not self.search(pattern, start=start)
        ]
        # The number of log derivative blocks depends on the program version, but all blocks
        # share the same energy grid, so a truncated block is shorter than the others
        log_derivatives = self.log_derivatives
        if not log_derivatives or len({len(ld["e"]) for ld in log_derivatives}) != 1:
            missing.append("log_derivatives")
        # The gnuplot script is always written after the last of the plotting data
        if not self.search(r"^GNUSCRIPT", start=start):
            missing.append("gnuplot_script")
        return missing

    @cached_property
    def raw_lines(self) -> list[str]:
        """Raw lines of the output file."""
        return self._lines

    @cached_property
    def _message_line_indices(self) -> set[int]:
        """Indices of the raw lines corresponding to warnings and errors."""
        return {
            i - 1
            for message in self.warnings + self.errors
            for i in message["line_numbers"]
        }

    @cached_property
    def clean_lines(self) -> list[str]:
        """Lines of the output file, excluding those corresponding to warnings and errors."""
        # Remove lines corresponding to warnings and errors prior to parsing output quantities
        # because they may be inserted within formatted data blocks.
        return [
            line
            for i, line in enumerate(self._lines)
            if i not in self._message_line_indices
        ]

    @cached_property
    def line_contexts(self) -> list[LineContext]:
        """Program stage, channel and enclosing block at each of the raw lines."""
        return _sections.line_contexts(self.raw_lines)

    @cached_property
    def _clean_line_contexts(self) -> list[LineContext]:
        """Contexts of `clean_lines`."""
        return [
            ctx
            for i, ctx in enumerate(self.line_contexts)
            if i not in self._message_line_indices
        ]

    def _blocks(self, kind: str) -> list[tuple[Block, int, int]]:
        """Blocks of a kind as `(block, start, stop)`, where `clean_lines[start:stop]` is the block."""
        return list(_sections.blocks(self._clean_line_contexts, kind))

    @cached_property
    def program_information(self) -> dict:
        """Description of the program (name, version, relativistic treatment, etc.)

        Raises:
            ValueError: If the program name is unknown.

        Returns:
            dict:
                program (str): Name of the program (ONCVPSP or METAPSP).
                version (str): Version of the program.
                date (str): Date of the program run.
                relativistic_treatment (str): Relativistic treatment used.
        """
        program: str = self.clean_lines[0].strip().split()[0]
        if program == "ONCVPSP":
            line = self.clean_lines[1]
            relativistic_treatment = line.strip().split()[0]
        elif program == "METAPSP":
            line = self.clean_lines[2]
            relativistic_treatment = "non-relativistic"
        else:
            raise ValueError(f"Unknown program: {program}")
        version_match: re.Match[str] | None = re.search(r"(\d+\.\d+\.\d+)", line)
        version: str = version_match.group(1) if version_match else "0.0.0"
        date_match: re.Match[str] | None = re.search(r"(\d{2}/\d{2}/\d{4})", line)
        date: str = date_match.group(1) if date_match else "00/00/0000"
        return {
            "program": program,
            "version": version,
            "date": date,
            "relativistic_treatment": relativistic_treatment,
        }

    @cached_property
    def is_relativistic(self) -> bool:
        """Whether the fully-relativistic Dirac equation was solved."""
        return self.program_information["relativistic_treatment"] in (
            "relativistic",
            "fully-relativistic",
        )

    @cached_property
    def input(self) -> dict:
        """Parsed input data used for the pseudopotential generation."""
        if line_match := self.search(r"^# ATOM AND REFERENCE CONFIGURATION$"):
            i, _ = line_match
            return OncvpspInput.from_dat_string(
                "\n".join(self.clean_lines[i:])
            ).model_dump()
        return {}

    def get_reference_quantum_numbers(self, index) -> list[tuple[int, ChannelKey]]:
        """Get the states `(n, channel)` of the reference configuration. `index` acts upon the
        reference configuration _as it is written in the input file_, i.e. per l-channel, before
        splitting into j-channels in the fully-relativistic case.

        Args:
            index (int | slice): State ind(ex|ices)

        Returns:
            list[tuple[int, ChannelKey]]: States `(n, channel)`.
        """
        config = self.input["reference_configuration"]
        quantum_numbers = []
        for enn, ell in zip(config["n"][index], config["l"][index]):
            for channel in reversed(j_channels(ell, self.is_relativistic)):
                quantum_numbers.append((enn, channel))
        return quantum_numbers

    @property
    def core_quantum_numbers(self) -> list[tuple[int, ChannelKey]]:
        """States `(n, channel)` of the core."""
        n_core = self.input["oncvpsp"]["nc"]
        return self.get_reference_quantum_numbers(index=slice(0, n_core))

    @property
    def valence_quantum_numbers(self) -> list[tuple[int, ChannelKey]]:
        """Valence states `(n, channel)`."""
        n_core = self.input["oncvpsp"]["nc"]
        return self.get_reference_quantum_numbers(index=slice(n_core, None))

    @property
    def valence_quantum_numbers_and_ae_eig(self) -> list[tuple[int, ChannelKey, float]]:
        """Valence states `(n, channel, all-electron eigenvalue)`."""
        vqn_eig = []
        for enn, channel in self.valence_quantum_numbers:
            for config in self.reference_configuration:
                if (config["n"], ChannelKey.from_record(config)) == (enn, channel):
                    vqn_eig.append((enn, channel, config["eig_ae"]))
        return vqn_eig

    @cached_property
    def reference_configuration(self) -> list[dict]:
        """Information about the reference configuration states, including all-electron eigenvalues.

        Raises:
            RuntimeError: If the "ATOM AND REFERENCE CONFIGURATION" block cannot be found.

        Returns:
            list[dict]: One per state of each l-channel or j-channel:
                n (int): Principal quantum number.
                l (int): Orbital angular momentum.
                j (float | None): Total angular momentum, None for an l-channel.
                f (float): Occupation number.
                eig_ae (float): All-electron eigenvalue (Ha).
                eig_pbe (float, optional): PBE eigenvalue (Ha), if available.
        """
        nlf_pattern = rf"\s+(?P<n>\d+)\s+(?P<l>\d+)\s+(?P<f>{RE_FLOAT})"
        pattern = re.compile(
            nlf_pattern + rf"\s+(?P<eig>{RE_FLOAT})(?:\s+(?P<eig_dw>{RE_FLOAT}))?"
        )
        n_states = self.input["oncvpsp"]["nc"] + self.input["oncvpsp"]["nv"]
        if self.program_information["program"] == "METAPSP":
            header_match = self.search(
                r"#   n    l    f      MGGA eval \(Ha\)         PBE         delta"
            )
            pattern = re.compile(
                nlf_pattern + rf"\s+(?P<eig>{RE_FLOAT})\s+(?P<eig_pbe>{RE_FLOAT})"
            )
        else:
            header_match = self.search(
                r"#\s+n\s+l\s+f\s+(?:energy \(Ha\)|l\+1/2\s+l-1/2)"
            )
            # ONCVPSP v2 prints a relativistic l+1/2, l-1/2 header after the energy header
            if (
                header_match
                and self.is_relativistic
                and re.match(
                    r"#\s+n\s+l\s+f\s+l\+1/2", self.clean_lines[header_match[0] + 1]
                )
            ):
                header_match = (header_match[0] + 1, header_match[1])
        if not header_match:
            raise RuntimeError(
                "Failed to find reference configuration section in output."
            )
        data: list[dict] = []
        for _, match in self.findall(
            pattern, start=header_match[0] + 1, stop=header_match[0] + 1 + n_states
        ):
            enn = int(match.group("n"))
            ell = int(match.group("l"))
            occ = float(match.group("f"))
            # Fully-relativistic eigenvalues are printed for j = l + 1/2, then j = l - 1/2
            channels = j_channels(ell, self.is_relativistic)
            eig_groups = ["eig", "eig_dw"][: len(channels)]
            for channel, group in zip(channels, eig_groups):
                data.append(
                    {
                        "n": enn,
                        **channel.as_dict(),
                        "f": occ,
                        "eig_ae": fort_float(match.group(group)),
                    }
                )
                if self.program_information["program"] == "METAPSP":
                    data[-1]["eig_pbe"] = fort_float(match.group("eig_pbe"))
        return data

    @cached_property
    def projector_layout(self) -> ProjectorLayout | None:
        """Projector state of every projector as predicted from the input (`None` if the output
        echoes no input). See `pyoncvpsp.projectors`."""
        if not self.input:
            return None
        info = self.program_information
        return predict_projector_layout(
            OncvpspInput.from_dict(self.input),
            info["program"],
            info["version"],
            self.is_relativistic,
        )

    @cached_property
    def _valence_projector_states(
        self,
    ) -> dict[ProjectorKey, tuple[int, ChannelKey, float]]:
        """Projectors built from a valence state solved as a bound state, with the state
        `(n, channel, eig_ae)` of the reference configuration."""
        if self.projector_layout is None:
            return {}
        config = list(self.input["reference_configuration"]["n"])
        eigs = {
            (state["n"], ChannelKey.from_record(state)): state["eig_ae"]
            for state in self.reference_configuration
        }
        states = {}
        for slot in self.projector_layout:
            if slot.is_valence and slot.valence_index is not None:
                key = (config[slot.valence_index], slot.projector.channel)
                if key in eigs:
                    states[slot.projector] = (*key, eigs[key])
        return states

    @cached_property
    def _unoccupied_projector_states(self) -> list[dict]:
        """Projector states other than bound valence states, in the order in which they are
        reported: wellstates and localized bound states (which ONCVPSP reports only in a warning).

        Each reported state fills the next unresolved slot of its channel in `projector_layout`,
        which also gives its `iproj`. In fully-relativistic ONCVPSP v4, the j = l - 1/2 state is
        solved in the well of its j = l + 1/2 partner and takes the partner's kind; the energy
        printed for it is only the target energy (`energy_guess`), so its `eig_ae` is `None`.
        """
        wellstate_pattern = re.compile(
            r"\s+Wellstate for l =\s+(?P<ell>\d)  n =\s+(?P<enn>\d+)(?:\s+kap(?:pa)?=\s+(?P<kappa>-?\d+))?"
        )
        localized_pattern = re.compile(
            r"WARNING wellstate: localized bound state found for n=\s*(?P<enn>\d+)\s+l=\s*(?P<ell>\d+)"
        )
        # ONCVPSP v3 uses the normal bound state if one is found for the maximum well barrier
        negative_energy_pattern = re.compile(
            r"wellstate: Negative-energy state for l=\s*(?P<ell>\d+) found for maximum barrier"
        )
        negative_energy_eig_pattern = re.compile(
            rf"Normal bound state at ep=\s*(?P<eig>{RE_FLOAT})"
        )
        # ONCVPSP v2/v3 solve a negative-energy second projector state as a bound state
        second_projector_pattern = re.compile(
            r"WARNING: negative energy of\s*" + RE_FLOAT
        )
        second_projector_eig_pattern = re.compile(
            rf"\s*Actual bound-state energy of\s*(?P<eig>{RE_FLOAT})"
        )
        # Fully-relativistic ONCVPSP v3 reports n and kappa when renormalizing the state
        renormalization_pattern = re.compile(
            rf"\s+cnorm=\s*{RE_FLOAT}\s+l{{1,2}}=\s*(?P<ell>\d+),?\s+n=\s*(?P<enn>\d+)\s+kap=\s*(?P<kappa>-?\d+)"
        )
        eig_pattern = re.compile(rf"\s+eigenvalue =\s*(?P<eig>{RE_FLOAT})")
        localized_eig_pattern = re.compile(
            rf"WARNING this state with energy\s*(?P<eig>{RE_FLOAT})"
        )
        ap_pattern = re.compile(rf"\s+asymptotic potential =\s*(?P<ap>{RE_FLOAT})")
        hpr_pattern = re.compile(rf"\s+half-point radius =\s*(?P<hpr>{RE_FLOAT})")

        if self.projector_layout is None:
            return []
        unresolved = [slot for slot in self.projector_layout if not slot.is_valence]

        def match_float(pattern: re.Pattern[str], i: int, group: str) -> float | None:
            if i < len(self._lines) and (match := re.match(pattern, self._lines[i])):
                return float(match.group(group))
            return None

        def renormalization(i: int) -> re.Match[str] | None:
            for k in range(i + 1, min(i + 6, len(self._lines))):
                if match := re.match(renormalization_pattern, self._lines[k]):
                    return match
            return None

        def take_slot(
            ell: int, kappa: int | None, solvers: tuple[Solver, ...] | None = None
        ) -> ProjectorSlot | None:
            channel = ChannelKey.from_printed_kappa(ell, kappa) if kappa else None
            for slot in unresolved:
                if (
                    slot.projector.channel.l == ell
                    and channel in (None, slot.projector.channel)
                    and (solvers is None or slot.solver in solvers)
                ):
                    unresolved.remove(slot)
                    return slot
            return None

        states: list[dict] = []
        # Localized bound state warnings are removed from `clean_lines`, so search the raw lines
        for i, line in enumerate(self._lines):
            solvers = None
            if match := re.match(wellstate_pattern, line):
                ell = int(match.group("ell"))
                kappa = int(match.group("kappa")) if match.group("kappa") else None
                state = {
                    "n": int(match.group("enn")),
                    "is_bound": False,
                    "eig_ae": match_float(eig_pattern, i + 1, "eig"),
                    "asymptotic_potential": match_float(ap_pattern, i + 2, "ap"),
                    "half_point_radius": match_float(hpr_pattern, i + 3, "hpr"),
                }
            elif match := re.match(localized_pattern, line):
                ell = int(match.group("ell"))
                # Only reported for the j = l + 1/2 (kappa = -(l + 1)) state if relativistic
                kappa = -(ell + 1) if self.is_relativistic else None
                state = {
                    "n": int(match.group("enn")),
                    "is_bound": True,
                    "eig_ae": match_float(localized_eig_pattern, i + 1, "eig"),
                }
            elif match := re.search(negative_energy_pattern, line):
                ell = int(match.group("ell"))
                renorm = renormalization(i)
                kappa = int(renorm.group("kappa")) if renorm else None
                state = {
                    "n": int(renorm.group("enn")) if renorm else None,
                    "is_bound": True,
                    "eig_ae": match_float(negative_energy_eig_pattern, i + 1, "eig"),
                }
            elif re.match(second_projector_pattern, line):
                renorm = renormalization(i)
                if renorm:
                    ell, kappa = int(renorm.group("ell")), int(renorm.group("kappa"))
                else:
                    # Printed in the semilocal loop over l before the header of l
                    ell, kappa = self.line_contexts[i].pending_l, None
                solvers = (Solver.BY_ENERGY,)
                state = {
                    "n": int(renorm.group("enn")) if renorm else None,
                    "is_bound": True,
                    "eig_ae": match_float(second_projector_eig_pattern, i + 1, "eig"),
                }
            else:
                continue
            if (slot := take_slot(ell, kappa, solvers)) is None:
                continue
            if state["n"] is None:
                state["n"] = slot.n
            if slot.solver == Solver.PARTNER_WELL:
                state["energy_guess"], state["eig_ae"] = state["eig_ae"], None
            states.append(
                {
                    "n": state.pop("n"),
                    **slot.projector.as_dict(),
                    "energy_guess": None,
                    "asymptotic_potential": None,
                    "half_point_radius": None,
                    **state,
                }
            )
        # A state solved in the well of its j = l + 1/2 partner is of the partner's kind
        kinds = {ProjectorKey.from_record(state): state["is_bound"] for state in states}
        for state in states:
            projector = ProjectorKey.from_record(state)
            slot = self.projector_layout.get(projector)
            if slot is not None and slot.solver == Solver.PARTNER_WELL:
                partner = j_channels(projector.channel.l, True)[0]
                state["is_bound"] = kinds.get(
                    ProjectorKey(partner, projector.iproj), state["is_bound"]
                )
        return states

    @cached_property
    def wellstate_metadata(self) -> list[dict]:
        """Information about any wellstate solutions found in the output.

        Returns:
            dict:
                n (int): Principal quantum number.
                l (int): Orbital angular momentum.
                j (float | None): Total angular momentum, None for an l-channel.
                iproj (int): Projector index (1-based), from `projector_layout`.
                eig_ae (float | None): All-electron eigenvalue (Ha); None for the j = l - 1/2 states of
                    fully-relativistic ONCVPSP v4, for which only the target energy is printed.
                energy_guess (float | None): Target energy printed in place of the eigenvalue (Ha),
                    for the j = l - 1/2 states of fully-relativistic ONCVPSP v4; else None.
                asymptotic_potential (float | None): Asymptotic potential (Ha), if reported.
                half_point_radius (float | None): Half-point radius (a.u.), if reported.
        """
        return [
            {k: v for k, v in state.items() if k != "is_bound"}
            for state in self._unoccupied_projector_states
            if not state["is_bound"]
        ]

    @cached_property
    def localized_bound_states(self) -> list[dict]:
        """Unoccupied bound states which are sufficiently localized to be used for projectors
        in place of wellstates.

        Returns:
            dict:
                n (int): Principal quantum number.
                l (int): Orbital angular momentum.
                j (float | None): Total angular momentum, None for an l-channel.
                iproj (int): Projector index (1-based), from `projector_layout`.
                eig_ae (float | None): All-electron eigenvalue (Ha); see `wellstate_metadata`.
                energy_guess (float | None): See `wellstate_metadata`.
                asymptotic_potential (None): Always None.
                half_point_radius (None): Always None.
        """
        return [
            {k: v for k, v in state.items() if k != "is_bound"}
            for state in self._unoccupied_projector_states
            if state["is_bound"]
        ]

    @cached_property
    def _optimize_line_indices(self) -> list[tuple[ProjectorKey, int, int]]:
        """Optimized projector blocks as `(projector, start, stop)` (`clean_lines` indices)."""
        indices: dict[ProjectorKey, tuple[int, int]] = {}
        for block, start, stop in self._blocks(BlockKind.PROJECTOR):
            if block.l is None or block.iproj is None:
                raise RuntimeError(
                    f"Failed to find ang. mom. in `optimize` block starting at line {block.line + 1}"
                )
            # The relativistic loop over kappa does kap = -(l+1) (j = l+1/2) before kap = l
            first, *rest = j_channels(block.l, self.is_relativistic)
            projector = ProjectorKey(first, block.iproj)
            if projector in indices and rest:
                projector = ProjectorKey(rest[0], block.iproj)
            indices[projector] = (start, stop)
        return [
            (projector, start, stop) for projector, (start, stop) in indices.items()
        ]

    @property
    def _optimize_convergence_profiles(self) -> list[dict]:
        convergence_header_pattern = re.compile(r"\s+Ha\s+eV\s+Ha")
        convergence_body_pattern = re.compile(
            rf"\s+(?P<eresid>{RE_FLOAT})\s+{RE_FLOAT}\s+(?P<ecut>{RE_FLOAT})"
        )
        data: dict[ProjectorKey, dict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )
        for projector, start, stop in self._optimize_line_indices:
            header_matches = self.findall(convergence_header_pattern, start, stop)
            if header_matches:
                header_index = header_matches[0][0]
                # ONCVPSP prints an empty table when no residual threshold is reached;
                # keep the (empty) profile so that `ecut_recommended` reflects it
                data[projector].update({"eresid": [], "ecut": []})
                for _, match in self.findall(
                    convergence_body_pattern,
                    start=header_index,
                    stop=min(header_index + 5, stop),
                ):
                    data[projector]["eresid"].append(float(match.group("eresid")))
                    data[projector]["ecut"].append(float(match.group("ecut")))
        return [
            {**projector.as_dict(), **{k: np.array(v) for k, v in value.items()}}
            for projector, value in data.items()
        ]

    @cached_property
    def _run_vkb_indices(self) -> list[tuple[ChannelKey, int, int]]:
        """VKB blocks as `(channel, start, coefficients)`: the `clean_lines` indices of the
        `B matrix` header and of the `Orthonormal projector coefficients` line. Blocks without
        coefficients (truncated output) are skipped."""
        coeff_pattern = re.compile(
            rf"\s+Orthonormal projector coefficients(?:\s+{RE_FLOAT})+"
        )
        indices: list[tuple[ChannelKey, int, int]] = []
        for block, start, stop in self._blocks(BlockKind.VKB):
            if not (coeff_matches := self.findall(coeff_pattern, start + 1, stop)):
                continue
            assert block.channel is not None
            indices.append((block.channel, start, coeff_matches[0][0]))
        return indices

    @cached_property
    def vkb_projector_coefficients(self) -> list[dict]:
        """Orthonormal Vanderbilt-Kleinman-Bylander coefficients `Dii`.

        Returns:
            list[dict]:
                l (int): Orbital angular momentum.
                j (float | None): Total angular momentum, None for an l-channel.
                iproj (int): Projector index.
                coefficient (float): VKB coefficient Dii.
        """
        coefficients: list[dict] = []
        for channel, _, stop in self._run_vkb_indices:
            for iproj, value in enumerate(self.clean_lines[stop].split()[3:], start=1):
                coefficients.append(
                    {
                        **ProjectorKey(channel, iproj).as_dict(),
                        "coefficient": float(value),
                    }
                )
        return coefficients

    @cached_property
    def vkb_scalar_projector_coefficients(self) -> list[dict]:
        r"""Projector coefficients 1/\tilde{b}_i for each angular momentum.

        Returns:
            list[dict]: List of dicts containing:
                l (int): Orbital angular momentum (l-channel).
                j (None): Always None.
                coefficients (np.ndarray): Orthonormal scalar projector coefficients for the given ell.
        """
        coefficients: list[dict] = []
        for i, match in self.findall(
            r" Orthonormal scalar projector coefficients, l =\s+(?P<ell>\d)"
        ):
            ell = int(match.group("ell"))
            coefficients.append(
                {
                    **ChannelKey(ell).as_dict(),
                    "coefficients": np.array(
                        [float(c) for c in self.clean_lines[i + 1].split()]
                    ),
                }
            )
        return coefficients

    @cached_property
    def vkb_spin_orbit_projector_coefficients(self) -> list[dict]:
        """ "Spin-orbit"-convention projector coefficients for each angular momentum.

        Returns:
            list[dict]: List of dicts containing:
                l (int): Orbital angular momentum (l-channel).
                j (None): Always None.
                coefficients (np.ndarray): Orthonormal "spin-orbit" projector coefficients for the given ell.
        """
        coefficients: list[dict] = []
        for i, match in self.findall(
            r" Orthonormal spin-orbit projector coefficients, l =\s+(?P<ell>\d)"
        ):
            ell = int(match.group("ell"))
            coefficients.append(
                {
                    **ChannelKey(ell).as_dict(),
                    "coefficients": np.array(
                        [float(c) for c in self.clean_lines[i + 1].split()]
                    ),
                }
            )
        return coefficients

    @cached_property
    def vkb_hermiticity_errors(self) -> list[dict]:
        """Hermiticity errors for Bij matrces.

        Returns:
            list[dict]: List of dicts containing:
                l (int): Orbital angular momentum.
                j (float | None): Total angular momentum, None for an l-channel.
                iproj (int): Projector index i.
                jproj (int): Projector index j.
                hermiticity_error (float): Hermiticity error for Bij matrix element (i, j) in Ha.
        """
        v3_herm_err_pattern = re.compile(
            rf"Hermiticity error\s+(?P<herm_err>{RE_FLOAT})"
        )
        # TODO: v3_bij_pattern: ^B11, B12, B22   -3.8877D-03    1.0547D-02   -2.8717D-02$
        v4_herm_err_pattern = re.compile(
            rf"\s+(?P<i>\d)\s+(?P<j>\d)\s+(?P<herm_err>{RE_FLOAT})"
        )
        hermiticity_errors: list[dict] = []
        for channel, start, stop in self._run_vkb_indices:
            # Parse hermiticity errors
            if herm_err_match := re.match(
                v3_herm_err_pattern, self.clean_lines[start + 1]
            ):
                hermiticity_errors.append(
                    {
                        **channel.as_dict(),
                        "iproj": 1,
                        "jproj": 2,
                        "hermiticity_error": fort_float(
                            herm_err_match.group("herm_err")
                        ),
                    }
                )
            else:
                herm_err_matches = self.findall(
                    v4_herm_err_pattern, start=start, stop=stop
                )
                for _, herm_err_match in herm_err_matches:
                    hermiticity_errors.append(
                        {
                            **channel.as_dict(),
                            "iproj": int(herm_err_match.group("i")),
                            "jproj": int(herm_err_match.group("j")),
                            "hermiticity_error": fort_float(
                                herm_err_match.group("herm_err")
                            ),
                        }
                    )
        return hermiticity_errors

    @cached_property
    def d2exc(self) -> dict:
        """Second derivatives of the exchange-correlation energy w.r.t. valence state occupations.

        Returns:
            dict: AE/PS d2Exc/dfi with/without non-linear core corrections.
        """
        pattern = re.compile(
            r"d2exc(?P<ae_ps>ae|ps) - (?:all-electron|pseudofunction) derivatives( with(?P<no> no)? core correction)?"
        )
        matches = self.findall(pattern)
        nv = self.input["oncvpsp"]["nv"]
        nlines_per_state = int(np.ceil(nv / 4))
        data = defaultdict(list)
        for i, match in matches:
            j = i + 2
            ae_ps = match.group("ae_ps")
            with_nlcc = match.group("no") is None
            key = f"{ae_ps}_with_nlcc" if (ae_ps == "ps" and with_nlcc) else ae_ps
            for _ in range(nv):
                line = " ".join(self.clean_lines[j : j + nlines_per_state])
                data[key].append([fort_float(x) for x in line.split()])
                j += nlines_per_state
        return {k: np.array(v) for k, v in data.items()}

    @cached_property
    def d2exc_rmse(self) -> float | None:
        """Root-mean-square error of PS d2Exc/dfi with non-linear core corrections w.r.t. AE d2Exc/dfi."""
        pattern = re.compile(rf"rms 2nd derivative error\s+(?P<rmse>{RE_FLOAT})")
        match = self.search(pattern)
        return float(match[1].group("rmse")) if match else None

    @cached_property
    def teter_parameters(self) -> dict:
        """Parameters of the Teter-function non-linear core correction model core charge density.

        Teter function is defined as: A(sin(2πx) / [(2π(x))(1 - 4(x)^2)(1 - (x)^2)])^2
        where x = r / (rcfact * rmatch) and A = fcfact * rhocmatch.

        N.B.: if icmod=2, all of the quantities have _different_ meanings!

        Returns:
            dict:
                fcfact (float): Amplitude prefactor.
                rcfact (float): Scale prefactor.
                rmatch (float): Core-valence charge density crossover radius (a.u.).
                rhocmatch (float): Core charge density at crossover radius (a.u.^-3).
        """
        parameters: dict[str, float] = {}
        if self.input["model_core_charge"]["icmod"] == 3:
            parameters["fcfact"] = self.input["model_core_charge"]["fcfact"]
            parameters["rcfact"] = self.input["model_core_charge"]["rcfact"]
        if self.input["model_core_charge"]["icmod"] == 4:
            param_match = self.search(r"amplitude prefactor, scale prefactor")
            if param_match:
                parameters["fcfact"] = float(
                    self.clean_lines[param_match[0] + 1].split()[0]
                )
                parameters["rcfact"] = float(
                    self.clean_lines[param_match[0] + 1].split()[1]
                )
        match_match = self.search(
            rf"rmatch, rhocmatch\s+(?P<rmatch>{RE_FLOAT})\s+(?P<rhocmatch>{RE_FLOAT})"
        )
        if match_match:
            parameters["rmatch"] = float(match_match[1].group("rmatch"))
            parameters["rhocmatch"] = float(match_match[1].group("rhocmatch"))
        return parameters

    @cached_property
    def teter_coarse_grid(self) -> dict:
        """Coarse grid evaulation of d2Exc/dfi RMSE over different values of Teter prefactors.
        Present only when icmod=4 in ONCVPSP.

        Returns:
            dict:
                fcfact (np.ndarray): Array of fcfact values.
                rcfact (np.ndarray): Array of rcfact values.
                d2exc_rmse (np.ndarray): 2D array of d2Exc/dfi RMSE values.
        """
        match = self.search(r"Coarse scan for minimum error")
        if not match:
            return {}
        grid: dict[str, list] = defaultdict(list)
        grid["fcfact"] = [float(x) for x in self.clean_lines[match[0] + 5].split()]
        i = match[0] + 7
        while words := self.clean_lines[i].strip().split():
            grid["rcfact"].append(float(words[0]))
            grid["d2exc_rmse"].append([float(x) for x in words[1:]])
            i += 1
        return {k: np.array(v) for k, v in grid.items()}

    @cached_property
    def diagnostic_tests_semilocal(self) -> list[dict]:
        """Diagnostic test information when solving for states in the semilocal pseudopotential.

        Returns:
            list[dict]:
                l (int): Orbital angular momentum.
                j (float | None): Total angular momentum, None for an l-channel.
                rcore (float): Core radius (a.u.).
                rmatch (float): Matching radius (a.u.).
                e_in (float): Input all-electron eigenvalue (Ha).
                e_test (float): Output pseudoeigenvalue (Ha).
                norm_test (float): Norm w/in rc of pseudo wavefunction / AE wavefunction.
                slope_test (float): Slope at rc of pseudo wavefuction / AE wavefunction.
        """
        pattern = re.compile(
            r"\s+(?P<ell>\d)(?:\s+(?P<kap>-?\d))?"
            rf"\s+(?P<rcore>{RE_FLOAT})"
            rf"\s+(?P<rmatch>{RE_FLOAT})"
            rf"\s+(?P<e_in>{RE_FLOAT})"
            rf"\s+(?P<e_test>{RE_FLOAT})"
            rf"\s+(?P<norm_test>{RE_FLOAT})"
            rf"\s+(?P<slope_test>{RE_FLOAT})"
        )
        return [
            {
                **ChannelKey.from_printed_kappa(
                    int(match.group("ell")),
                    int(match.group("kap")) if match.group("kap") is not None else None,
                ).as_dict(),
                "rc": float(match.group("rcore")),
                "rmatch": float(match.group("rmatch")),
                "e_in": float(match.group("e_in")),
                "e_test": float(match.group("e_test")),
                "norm_test": float(match.group("norm_test")),
                "slope_test": float(match.group("slope_test")),
            }
            for _, start, stop in self._blocks(BlockKind.DIAG_SEMILOCAL)
            for _, match in self.findall(pattern, start, stop)
        ]

    @cached_property
    def diagnostic_tests_vkb(self) -> list[dict]:
        """Diagnostic test information when solving for states in the VKB pseudopotential.

        Returns:
            list[dict]:
                l (int): Orbital angular momentum.
                j (float | None): Total angular momentum, None for an l-channel.
                rcore (float): Core radius (a.u.).
                rmatch (float): Matching radius (a.u.).
                e_in (float): Input all-electron eigenvalue (Ha).
                e_test (float | None): Output pseudoeigenvalue (Ha), if ONCVPSP v3.
                delta_e (float | None): Eigenvalue error (Ha), if ONCVPSP v4 or METAPSP v1.
                norm_test (float): Norm w/in rc of pseudo wavefunction / AE wavefunction.
                slope_test (float): Slope at rc of pseudo wavefuction / AE wavefunction.
        """
        is_v3 = self.program_information["version"].startswith("3")
        pattern = re.compile(
            r"\s+(?P<ell>\d)(?:\s+(?P<kap>-?\d))?"
            rf"\s+(?P<rcore>{RE_FLOAT})"
            rf"\s+(?P<rmatch>{RE_FLOAT})"
            rf"\s+(?P<e_in>{RE_FLOAT})"
            rf"\s+(?P<e_or_delta_e_test>{RE_FLOAT})"
            rf"\s+(?P<norm_test>{RE_FLOAT})"
            rf"\s+(?P<slope_test>{RE_FLOAT})"
        )
        return [
            {
                **ChannelKey.from_printed_kappa(
                    int(match.group("ell")),
                    int(match.group("kap")) if match.group("kap") is not None else None,
                ).as_dict(),
                "rc": float(match.group("rcore")),
                "rmatch": float(match.group("rmatch")),
                "e_in": float(match.group("e_in")),
                "e_test": float(match.group("e_or_delta_e_test")) if is_v3 else None,
                "delta_e": float(match.group("e_or_delta_e_test"))
                if not is_v3
                else None,
                "norm_test": float(match.group("norm_test")),
                "slope_test": float(match.group("slope_test")),
            }
            for _, start, stop in self._blocks(BlockKind.DIAG_VKB)
            for _, match in self.findall(pattern, start, stop)
        ]

    @cached_property
    def ghosts(self) -> list[dict]:
        """Information about potential ghost states.

        Returns:
            list[dict]:
                l (int): Orbital angular momentum.
                j (float | None): Total angular momentum, None for an l-channel.
                eig (float): Eigenvalue of the ghost state (Ha).
                ecut (float): Cutoff energy where the ghost appears (Ha).
                sign (str): Sign of the ghost state ('+' or '-').
                r_rc (float | None): r / rc, if unbound (sign == '+').
        """
        ghosts = []
        pattern = re.compile(
            r"\s+(?P<ell>\d)(?:\s+(?P<kap>-?\d))?"
            rf"\s+(?:(?P<r_rc>{RE_FLOAT})\s+)?"
            rf"(?P<eig>{RE_FLOAT})"
            rf"\s+(?P<ecut>{RE_FLOAT})\s+WARNING - GHOST(?:\((?P<sign>[+-])\))?"
        )
        matches = self.findall(pattern)
        for _, match in matches:
            ghosts.append(
                {
                    **ChannelKey.from_printed_kappa(
                        int(match.group("ell")),
                        int(match.group("kap"))
                        if match.group("kap") is not None
                        else None,
                    ).as_dict(),
                    "eig": float(match.group("eig")),
                    "ecut": float(match.group("ecut")),
                    # ONCVPSP v3 only tests for bound ghosts, and prints no sign
                    "sign": match.group("sign") or "-",
                    "r_rc": float(match.group("r_rc"))
                    if match.group("r_rc") is not None
                    else None,
                }
            )
        return ghosts

    @cached_property
    def test_configurations(self) -> list[list[dict]]:
        """Test configuration results.

        Returns:
            list[list[dict]]:
                n (int): Principal quantum number.
                l (int): Orbital angular momentum.
                j (float | None): Total angular momentum, None for an l-channel.
                f (float): Occupation number.
                eig_ae (float): All-electron eigenvalue (Ha).
                eig_ps (float | None): Pseudoeigenvalue (Ha), if available.
        """
        test_configurations: list[list[dict]] = []
        # Large negative eigenvalues can run into the preceding column (e.g. `2.0000-1212.17912322`)
        pattern = re.compile(
            r"\s+(?P<enn>\d)\s+(?P<ell>\d)\s+(?:(?P<kappa>-?\d)\s+)?"
            rf"(?P<occ>{RE_FLOAT})"
            rf"(?:\s+|(?=-))(?P<eig_ae>{RE_FLOAT})"
            rf"(?:(?:\s+|(?=-))(?P<eig_ps>{RE_FLOAT}))?"
            rf"(?:\s+(?P<diff>{RE_FLOAT}))?"
        )
        for block, start, stop in self._blocks(BlockKind.TEST_CONFIG):
            test_configuration: list[dict] = []
            for line in self.clean_lines[start:stop]:
                if config_match := re.match(pattern, line):
                    enn = int(config_match.group("enn"))
                    ell = int(config_match.group("ell"))
                    kappa = (
                        int(config_match.group("kappa"))
                        if config_match.group("kappa") is not None
                        else None
                    )
                    test_configuration.append(
                        {
                            "n": enn,
                            **ChannelKey.from_printed_kappa(ell, kappa).as_dict(),
                            "f": float(config_match.group("occ")),
                            "eig_ae": float(config_match.group("eig_ae")),
                            "eig_ps": (
                                float(config_match.group("eig_ps"))
                                if config_match.group("eig_ps") is not None
                                else None
                            ),
                        }
                    )
            assert block.config_index == len(test_configurations)
            test_configurations.append(test_configuration)
        return test_configurations

    def _get_plot_blocks(
        self, pattern: re.Pattern[str]
    ) -> list[list[tuple[int, re.Match[str]]]]:
        matches = self.findall(pattern)
        blocks = []
        block = []
        i_prev = matches[0][0] - 1 if matches else 0
        for i, match in matches:
            if i - i_prev == 1:
                block.append((i, match))
                i_prev = i
            else:
                blocks.append(block)
                block = [(i, match)]
                i_prev = i
        if block:
            blocks.append(block)
        return blocks

    @cached_property
    def unscreened_semilocal_potentials(self) -> list[dict]:
        r"""Unscreened semilocal pseudopotential radial functions.

        Returns:
            list[dict]:
                l (int): Orbital angular momentum (l-channel).
                j (None): Always None.
                r (np.ndarray): Radial grid points (a.u.).
                v_sl (np.ndarray): Unscreened semilocal potential values (Ha).
        """
        pattern = re.compile(
            rf"!p\s+(?P<r>{RE_FLOAT})\s+(?P<rho_val>{RE_FLOAT})"
            + "".join(
                rf"\s+(?P<v_sl_{ell}>{RE_FLOAT})"
                for ell in range(self.input["pseudopotentials"]["lmax"] + 1)
            )
        )
        block = self.findall(pattern)
        data: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for _, match in block:
            r = float(match.group("r"))
            for ell in range(self.input["pseudopotentials"]["lmax"] + 1):
                data[ell]["r"].append(r)
                data[ell]["v_sl"].append(float(match.group(f"v_sl_{ell}")))
        return [
            {**ChannelKey(ell).as_dict(), **{k: np.array(v) for k, v in values.items()}}
            for ell, values in data.items()
        ]

    @cached_property
    def semilocal_potentials(self) -> list[dict]:
        r"""Semilocal pseudopotential radial functions.

        Returns:
            list[dict]:
                l (int): Orbital angular momentum (l-channel).
                j (None): Always None.
                r (np.ndarray): Radial grid points (a.u.).
                v_sl (np.ndarray): Semilocal potential values (Ha).
        """
        pattern = re.compile(
            rf"!ps\s+(?P<r>{RE_FLOAT})"
            + "".join(
                rf"\s+(?P<v_sl_{ell}>{RE_FLOAT})"
                for ell in range(self.input["pseudopotentials"]["lmax"] + 1)
            )
        )
        block = self.findall(pattern)
        data: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for _, match in block:
            r = float(match.group("r"))
            for ell in range(self.input["pseudopotentials"]["lmax"] + 1):
                data[ell]["r"].append(r)
                data[ell]["v_sl"].append(float(match.group(f"v_sl_{ell}")))
        return [
            {**ChannelKey(ell).as_dict(), **{k: np.array(v) for k, v in values.items()}}
            for ell, values in data.items()
        ]

    @cached_property
    def local_potential(self) -> dict:
        """Local potential radial function, if `lloc = 4`.

        Returns:
            dict:
                r (np.ndarray): Radial grid points (a.u.).
                v_loc (np.ndarray): Local potential values (Ha).
        """
        pattern = re.compile(rf" !L\s+(?P<r>{RE_FLOAT})\s+(?P<v_loc>{RE_FLOAT})")
        block = self.findall(pattern)
        data = defaultdict(list)
        for _, match in block:
            data["r"].append(float(match.group("r")))
            data["v_loc"].append(float(match.group("v_loc")))
        return {
            **{k: np.array(v) for k, v in data.items()},
            **self.input.get("local_potential", {}),
        }

    @cached_property
    def charge_densities(self) -> dict:
        r"""Radial charge densities.

        N.B.: defined as int[0,∞] r²ρi(r) dr = 1 for a single-particle density ρi(r).

        Returns:
            dict:
                r (np.ndarray): Radial grid points (a.u.).
                rho_val (np.ndarray): Valence charge density (e/a.u.³).
                rho_core (np.ndarray): Core charge density (e/a.u.³).
                rho_model_core (np.ndarray): Model core charge density (e/a.u.³).
        """
        pattern = re.compile(
            rf"!r\s+(?P<r>{RE_FLOAT})"
            rf"\s+(?P<rho_val>{RE_FLOAT})"
            rf"\s+(?P<rho_core>{RE_FLOAT})"
            rf"\s+(?P<rho_model_core>{RE_FLOAT})"
        )
        block = self.findall(pattern)
        data = defaultdict(list)
        for _, match in block:
            data["r"].append(float(match.group("r")))
            data["rho_val"].append(float(match.group("rho_val")))
            data["rho_core"].append(float(match.group("rho_core")))
            data["rho_model_core"].append(float(match.group("rho_model_core")))
        return {k: np.array(v) for k, v in data.items()}

    @cached_property
    def kinetic_energy_densities(self) -> dict:
        r"""Radial kinetic energy densities.

        Returns:
            dict:
                r (np.ndarray): Radial grid points (a.u.).
                tau_ps (np.ndarray): Pseudodensity kinetic energy density.
                tau_mod (np.ndarray): Model core kinetic energy density.
        """
        pattern = re.compile(
            rf"!t\s+(?P<r>{RE_FLOAT})\s+(?P<tau_ps>{RE_FLOAT})\s+(?P<tau_mod>{RE_FLOAT})"
        )
        block = self.findall(pattern)
        data = defaultdict(list)
        for _, match in block:
            data["r"].append(float(match.group("r")))
            data["tau_ps"].append(float(match.group("tau_ps")))
            data["tau_mod"].append(float(match.group("tau_mod")))
        return {k: np.array(v) for k, v in data.items()}

    @cached_property
    def kinetic_energy_potentials(self) -> dict:
        r"""Radial kinetic energy potentials.

        Returns:
            dict:
                r (np.ndarray): Radial grid points (a.u.).
                vtau_ps (np.ndarray): Pseudodensity kinetic energy potential.
                vtau_mod (np.ndarray): Model core kinetic energy potential.
        """
        pattern = re.compile(
            rf"!vt\s+(?P<r>{RE_FLOAT})\s+(?P<vtau_ps>{RE_FLOAT})\s+(?P<vtau_mod>{RE_FLOAT})"
        )
        block = self.findall(pattern)
        data = defaultdict(list)
        for _, match in block:
            data["r"].append(float(match.group("r")))
            data["vtau_ps"].append(float(match.group("vtau_ps")))
            data["vtau_mod"].append(float(match.group("vtau_mod")))
        return {k: np.array(v) for k, v in data.items()}

    @cached_property
    def wavefunctions(self) -> list[dict]:
        r"""Radial wavefunctions.

        Returns:
            list[dict]:
                r (np.ndarray): Radial grid points (a.u.).
                wfn_ae (np.ndarray): All-electron wavefunction values.
                wfn_ps (np.ndarray): Pseudowavefunction values.
                is_bound (bool | None): Whether the state is a bound state (True),
                a wellstate (False), or unknown (None).
                n (int | None): Principal quantum number, None if unknown.
                l (int): Orbital angular momentum.
                j (float | None): Total angular momentum, None for an l-channel.
                iproj (int): Projector index (1-based).
                f (float, optional): Occupation number, if available.
                eig_ae (float, optional): All-electron eigenvalue (Ha), if available.
                energy_guess (float, optional): See `wellstate_metadata`.
                eig_ps (float, optional): Pseudoeigenvalue (Ha), if available.
                asymptotic_potential (float, optional): Asymptotic potential (Ha), if available.
                half_point_radius (float, optional): Half-point radius (a.u.), if available.
        """
        data_pattern = re.compile(
            r"&\s+(?P<marker>[\s-])(?P<iproj>\d)?(?P<ell>\d)"
            rf"\s+(?P<r>{RE_FLOAT})\s+(?P<wfn_ae>{RE_FLOAT})\s+(?P<wfn_ps>{RE_FLOAT})"
        )
        header_pattern = re.compile(r"n=\s*(?P<enn>\d+),\s+l=\s*(?P<ell>\d)")
        v4_scattering_header_pattern = re.compile(
            r"scattering, iprj=\s*(?P<iproj>\d+),\s+l=\s*(?P<ell>\d)"
        )
        blocks = self._get_plot_blocks(data_pattern)
        # Used to augment state data with metadata from reference configuration
        # Quantum number degeneracy is possible for wellstates, so add the eigenvalue to the key too
        # Test configuration eigenvalues are printed with a different precision than the reference
        # configuration eigenvalues, so they are keyed by quantum numbers only (bound states are unique)
        test_config = (
            {
                (cfg["n"], ChannelKey.from_record(cfg)): cfg
                for cfg in self.test_configurations[0]
            }
            if self.test_configurations
            else {}
        )
        ref_config = {
            (cfg["n"], ChannelKey.from_record(cfg), cfg["eig_ae"]): cfg
            for cfg in self.reference_configuration
        }
        # States from which the projectors are built: first the valence states (in input order),
        # then wellstates / localized bound states
        projector_states: dict[ProjectorKey, tuple[str, Any]] = {
            projector: ("valence", state_key)
            for projector, state_key in self._valence_projector_states.items()
        }
        for state in self._unoccupied_projector_states:
            projector_states[ProjectorKey.from_record(state)] = (
                "localized" if state["is_bound"] else "wellstate",
                state,
            )
        data: list[dict] = []
        for block in blocks:
            block_data: dict[str, Any] = defaultdict(list)
            # Parse the block data
            for _, match in block:
                block_data["r"].append(float(match.group("r")))
                block_data["wfn_ae"].append(float(match.group("wfn_ae")))
                block_data["wfn_ps"].append(float(match.group("wfn_ps")))
            block_data = {k: np.array(v) for k, v in block_data.items()}
            # Parse the header line and first block line for enn, ell, bound vs. scattering state, iproj
            first_ln, first_match = block[0]
            header_line: str = self.clean_lines[first_ln - 2]
            block_data["is_bound"] = None
            if header_match := re.match(header_pattern, header_line):
                if self.program_information["version"].startswith("4"):
                    block_data["is_bound"] = True
                block_data["n"] = int(header_match.group("enn"))
                ell = int(header_match.group("ell"))
                if first_match.group("iproj"):
                    iproj = int(first_match.group("iproj"))
                elif int(first_match.group("ell")) == ell:
                    iproj = 1
                elif int(first_match.group("ell")) == ell + 4:
                    iproj = 2
                else:
                    raise RuntimeError(
                        f"Failed to infer projector index for ONCVPSP v3 from ell={ell} in header line: '{header_line}'"
                    )
            elif header_match := re.match(v4_scattering_header_pattern, header_line):
                block_data["is_bound"] = False
                ell = int(header_match.group("ell"))
                iproj = int(header_match.group("iproj"))
                block_data["n"] = None
            else:
                raise RuntimeError(
                    f"Failed to parse wavefunction header line: '{header_line}'"
                )
            projector = ProjectorKey(
                ChannelKey.from_plot_marker(
                    ell, first_match.group("marker"), self.is_relativistic
                ),
                iproj,
            )
            # Add additional information about the state from the reference configuration,
            # wellstate metadata, or localized bound state
            state_kind, state = projector_states.get(projector, (None, None))
            state_metadata = {
                "is_bound": None,
                "n": 0,
                "f": np.nan,
                "eig_ae": np.nan,
                "eig_ps": np.nan,
                "asymptotic_potential": np.nan,
                "half_point_radius": np.nan,
                "r_rc": np.nan,
            }  # Default empty metadata
            if (
                state_kind == "valence" and block_data["is_bound"] is not False
            ):  # pyrefly: ignore
                # If we're sure this is not a wellstate, get a boundstate
                state_metadata.update(test_config.get(state[:2], {}))
                state_metadata.update(ref_config[state])
                state_metadata["is_bound"] = True
            elif (
                state_kind == "localized" and block_data["is_bound"] is not False
            ):  # pyrefly: ignore
                state_metadata.update(state)
                state_metadata["f"] = 0.0
            elif state_kind == "wellstate":
                # N.B. a wellstate can be bound in the well potential, in which case ONCVPSP v4
                # plots it with a bound-state header
                state_metadata.update(state)
                state_metadata["f"] = 0.0
            block_data.update(state_metadata)
            # Finalize block data
            block_data.update(projector.as_dict())
            data.append(block_data)
        return data

    @cached_property
    def optimized_pseudo_wavefunctions(self) -> list[dict]:
        """Optimized pseudo-wavefunctions of the channels with two projectors.

        N.B.: only ONCVPSP v2/v3 plot them (`@` lines, "projector pseudo wave functions, well or 2nd
        valence"), for every non-local channel with `nproj = 2` (`run_plot.f90`, `run_plot_r.f90`).
        The `n` printed in their header is always `l + 1, l + 2`, so it is not reported.

        Returns:
            list[dict]:
                l (int): Orbital angular momentum.
                j (float | None): Total angular momentum, None for an l-channel.
                iproj (int): Projector index (1-based).
                r (np.ndarray): Radial grid points (a.u.).
                wfn_ps (np.ndarray): Optimized pseudo-wavefunction values.
        """
        pattern = re.compile(
            rf"@\s+(?P<marker>-?)(?P<ell>\d)\s+(?P<r>{RE_FLOAT})"
            rf"\s+(?P<wfn_ps_1>{RE_FLOAT})\s+(?P<wfn_ps_2>{RE_FLOAT})"
        )
        data: list[dict] = []
        for block in self._get_plot_blocks(pattern):
            _, first_match = block[0]
            channel = ChannelKey.from_plot_marker(
                int(first_match.group("ell")),
                first_match.group("marker"),
                self.is_relativistic,
            )
            r = np.array([float(match.group("r")) for _, match in block])
            for iproj in (1, 2):
                data.append(
                    {
                        **ProjectorKey(channel, iproj).as_dict(),
                        "r": r,
                        "wfn_ps": np.array(
                            [
                                float(match.group(f"wfn_ps_{iproj}"))
                                for _, match in block
                            ]
                        ),
                    }
                )
        return data

    @cached_property
    def vkb_projectors(self) -> list[dict]:
        """Vanderbilt-Kleinman-Bylander projectors.

        N.B.: only ONCVPSP v4 and METAPSP plot the projectors (`!J` lines). The `@` lines of earlier
        versions ("projector pseudo wave functions") are pseudo wavefunctions, not projectors.

        Returns:
            list[dict]:
                r (np.ndarray): Radial grid points (a.u.).
                proj (np.ndarray): VKB projector values.
                coeff (float | None): VKB coefficient Dii, if available.
                l (int): Orbital angular momentum.
                j (float | None): Total angular momentum, None for an l-channel.
                iproj (int): Projector index (1-based).
        """
        max_nproj: int = max(self.input["vkb_projectors"]["nproj"])
        pattern = re.compile(
            rf"!J\s+(?P<marker>[\s-])(?P<ell>\d)\s+(?P<r>{RE_FLOAT})\s+(?P<proj_1>{RE_FLOAT})"
            + "".join(
                rf"(?:\s+(?P<proj_{i}>{RE_FLOAT}))?" for i in range(2, max_nproj + 1)
            )
        )
        # Used to look up VKB coefficients
        coeffs = {
            ProjectorKey.from_record(c): c["coefficient"]
            for c in self.vkb_projector_coefficients
        }
        blocks = self._get_plot_blocks(pattern)
        data: list[dict] = []
        for block in blocks:
            _, first_match = block[0]
            channel = ChannelKey.from_plot_marker(
                int(first_match.group("ell")),
                first_match.group("marker"),
                self.is_relativistic,
            )
            block_data: dict[str, Any] = defaultdict(list)
            for _, match in block:
                block_data["r"].append(float(match.group("r")))
                for i in range(1, max_nproj + 1):
                    proj_key = f"proj_{i}"
                    if match.group(proj_key) is not None:
                        block_data[proj_key].append(float(match.group(proj_key)))
            block_data = {k: np.array(v) for k, v in block_data.items()}
            for iproj in range(1, max_nproj + 1):
                proj_key = f"proj_{iproj}"
                if proj_key in block_data:
                    projector = ProjectorKey(channel, iproj)
                    data.append(
                        {
                            **projector.as_dict(),
                            "r": block_data["r"],
                            "proj": block_data[proj_key],
                            "coeff": coeffs.get(projector, np.nan),
                        }
                    )
        return data

    @property
    def _plot_convergence_profiles(self) -> list[dict]:
        pattern = re.compile(
            rf"!C\s+(?P<ell>\d)\s+(?P<ecut>{RE_FLOAT})\s+(?P<eresid>{RE_FLOAT})"
        )
        block = self.findall(pattern)
        data: dict[ProjectorKey, dict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )
        for _, match in block:
            # Plotted only for the first projector, and for j = l + 1/2 if fully-relativistic
            channel = j_channels(int(match.group("ell")), self.is_relativistic)[0]
            data[ProjectorKey(channel, 1)]["ecut"].append(float(match.group("ecut")))
            data[ProjectorKey(channel, 1)]["eresid"].append(
                float(match.group("eresid"))
            )
        return [
            {**projector.as_dict(), **{k: np.array(v) for k, v in value.items()}}
            for projector, value in data.items()
        ]

    @cached_property
    def convergence_profiles(self) -> list[dict]:
        """Convergence behavior of the VKB projectors w.r.t. Ecut.

        Returns:
            list[dict]:
                l (int): Orbital angular momentum.
                j (float | None): Total angular momentum, None for an l-channel.
                iproj (int): Projector index (1-based).
                ecut (np.ndarray): Cutoff energies (Ha).
                eresid (np.ndarray): Kinetic energy residual L1 errors of the VKB projectors.
        """
        return self._optimize_convergence_profiles + self._plot_convergence_profiles

    @cached_property
    def ecut_recommended(self) -> float:
        """Recommended cutoff energy in Hartree. Determined as the maximum, over all angular momenta
        and projectors, of the Ecut at which the kinetic energy residual reaches 1e-5 Ha.

        Returns:
            float: Recommended Ecut (Ha). `inf` if any convergence profile lacks a 1e-5 Ha entry;
                `-inf` if no convergence profiles were found.
        """
        ecut = -np.inf
        for prof in self.convergence_profiles:
            # if prof["iproj"] != 1:
            #     continue
            mask = np.isclose(prof["eresid"], 1e-5, atol=1e-10)
            if np.sum(mask) == 0 or not np.any(mask):
                ecut_proj = np.inf
            else:
                ecut_proj = prof["ecut"][mask][0]
            if ecut_proj > ecut:
                ecut = ecut_proj
        return ecut

    @cached_property
    def log_derivatives(self) -> list[dict]:
        """Phase-shift-like AE/PS logarithmic derivatives.

        Returns:
            list[dict]:
                l (int): Orbital angular momentum.
                j (float | None): Total angular momentum, None for an l-channel.
                rxpsh (float): Radius for logarithmic derivative evaluation (a.u.).
                e (np.ndarray): Energies (Ha).
                log_deriv_ae (np.ndarray): All-electron logarithmic derivative.
                log_deriv_ps (np.ndarray): Pseudopotential logarithmic derivative.
        """
        pattern = re.compile(
            rf"!\s+(?P<marker>[\s-])(?P<ell>\d)\s+(?P<e>{RE_FLOAT})"
            rf"\s+(?P<log_deriv_ae>{RE_FLOAT})\s+(?P<log_deriv_ps>{RE_FLOAT})"
        )
        blocks = self._get_plot_blocks(pattern)
        data: dict[ChannelKey, dict[str, Any]] = defaultdict(lambda: defaultdict(list))
        for block in blocks:
            first_ln, first_match = block[0]
            header_line: str = self.clean_lines[first_ln - 3]
            channel = ChannelKey.from_plot_marker(
                int(first_match.group("ell")),
                first_match.group("marker"),
                self.is_relativistic,
            )
            data[channel]["rxpsh"] = float(header_line.split("=")[-1])
            for _, match in block:
                channel = ChannelKey.from_plot_marker(
                    int(match.group("ell")), match.group("marker"), self.is_relativistic
                )
                data[channel]["e"].append(float(match.group("e")))
                data[channel]["log_deriv_ae"].append(float(match.group("log_deriv_ae")))
                data[channel]["log_deriv_ps"].append(float(match.group("log_deriv_ps")))
        return [
            {**channel.as_dict(), **{k: np.array(v) for k, v in value.items()}}
            for channel, value in data.items()
        ]

    @cached_property
    def plot_data(self) -> dict:
        """Dictionary containing plot data (radial functions).

        Returns:
            dict:
                local_potential (dict): See `OncvpspTextParser.local_potential`.
                charge_densities (dict): See `OncvpspTextParser.charge_densities`.
                kinetic_energy_densities (dict): See `OncvpspTextParser.kinetic_energy_densities`.
                kinetic_energy_potentials (dict): See `OncvpspTextParser.kinetic_energy_potentials`.
                wavefunctions (dict): See `OncvpspTextParser.wavefunctions`.
                optimized_pseudo_wavefunctions (dict): See
                    `OncvpspTextParser.optimized_pseudo_wavefunctions`.
                vkb_projectors (dict): See `OncvpspTextParser.vkb_projectors`.
                convergence_profiles (dict): See `OncvpspTextParser.convergence_profiles`.
                log_derivatives (dict): See `OncvpspTextParser.log_derivatives`.
        """
        return {
            "unscreened_semilocal_potentials": self.unscreened_semilocal_potentials,
            "semilocal_potentials": self.semilocal_potentials,
            "local_potential": self.local_potential,
            "charge_densities": self.charge_densities,
            "kinetic_energy_densities": self.kinetic_energy_densities,
            "kinetic_energy_potentials": self.kinetic_energy_potentials,
            "wavefunctions": self.wavefunctions,
            "optimized_pseudo_wavefunctions": self.optimized_pseudo_wavefunctions,
            "vkb_projectors": self.vkb_projectors,
            "convergence_profiles": self.convergence_profiles,
            "log_derivatives": self.log_derivatives,
        }

    @cached_property
    def plot_string(self) -> str:
        """String containing data for plotting."""
        return "\n".join(self._get_lines_between("DATA FOR PLOTTING", "GNUSCRIPT"))

    @cached_property
    def gnuplot_script(self) -> str:
        """String containing gnuplot script for plotting."""
        return "\n".join(self._get_lines_between("GNUSCRIPT", "END_GNU"))

    @cached_property
    def upf_string(self) -> str:
        """String containing the pseudopotential in UPF format."""
        return "\n".join(
            self._get_lines_between(
                '<UPF version="2.0.1">', "</UPF>", include_start=True, include_stop=True
            )
        )

    @cached_property
    def psp8_string(self) -> str:
        """String containing the pseudopotential in Abinit PSP8 format."""
        return "\n".join(self._get_lines_between("Begin PSPCODE8", "</INPUT>"))

    # --- Attribution ---------------------------------------------------------------------------------

    @cached_property
    def messages(self) -> list[OncvpspMessage]:
        """All warnings and errors, attributed to the program stage, angular-momentum channel and
        input parameters that produced them, sorted by line."""
        return _attribution.attribute_messages(self)

    @cached_property
    def failure(self) -> OncvpspFailure | None:
        """Why the run failed (primary error, stage, channel, implicated parameters), or `None` if
        ONCVPSP wrote all of its output."""
        return _attribution.analyze_failure(self)

    @cached_property
    def last_stage_reached(self) -> Stage:
        """Last stage of the program for which any output was found."""
        return self.line_contexts[-1].stage if self.line_contexts else Stage.INPUT

    @cached_property
    def output(self) -> dict:
        """Parsed output as a dictionary."""
        output: dict[str, Any] = {}
        output["completed"] = self.completed
        output["missing_plot_data"] = self.missing_plot_data
        output["errors"] = self.errors
        output["warnings"] = self.warnings
        output["program_information"] = self.program_information
        output["input"] = self.input
        output["reference_configuration"] = self.reference_configuration
        output["wellstate_metadata"] = self.wellstate_metadata
        output["localized_bound_states"] = self.localized_bound_states
        output["vkb_hermiticity_errors"] = self.vkb_hermiticity_errors
        output["vkb_scalar_projector_coefficients"] = (
            self.vkb_scalar_projector_coefficients
        )
        output["vkb_spin_orbit_projector_coefficients"] = (
            self.vkb_spin_orbit_projector_coefficients
        )
        output["d2exc"] = self.d2exc
        output["d2exc_rmse"] = self.d2exc_rmse
        output["teter_parameters"] = self.teter_parameters
        output["teter_coarse_grid"] = self.teter_coarse_grid
        output["diagnostic_tests_semilocal"] = self.diagnostic_tests_semilocal
        output["diagnostic_tests_vkb"] = self.diagnostic_tests_vkb
        output["ghosts"] = self.ghosts
        output["test_configurations"] = self.test_configurations
        output["plot_data"] = self.plot_data
        output["gnuplot_script"] = self.gnuplot_script
        output["upf"] = self.upf_string
        output["psp8"] = self.psp8_string
        return output

    def to_dict(self) -> dict:
        """Parsed output as a serializable dictionary."""

        def serialize(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, dict):
                return {k: serialize(v) for k, v in value.items()}
            if isinstance(value, list):
                return [serialize(v) for v in value]
            return value

        return serialize(self.output)

    def parse(self, check: bool = True, werror: bool = False) -> dict:
        """Parse the output file.

        Args:
            check (bool, optional): Check and raise exceptions for errors. Defaults to True.
            werror (bool, optional): Consider warnings as errors. Defaults to False.

        Returns:
            dict: `self.output` dictionary
        """
        if check:
            self.check(werror=werror)
        return self.output
