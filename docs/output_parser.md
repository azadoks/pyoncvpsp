# The output parser

`pyoncvpsp.io.OncvpspTextParser` reads the stdout of ONCVPSP (v2, v3, v4 and the azadoks/oncvpsp
builds) and METAPSP, and parses it into structured data. This page lists everything it parses,
when each quantity is in the output, and how to get it, one at a time or all together.

Units are those of the programs: energies in Hartree, radii in bohr. Arrays are `numpy` arrays.

## Reading an output

```python
from pyoncvpsp.io import OncvpspTextParser

parser = OncvpspTextParser(path="Si.out")   # or io=..., text=..., lines=[...]
parser.reference_configuration             # one quantity, parsed on first access
parser.output                              # every quantity, as a dict
parser.to_dict()                           # the same, with the arrays as lists (JSON-ready)
```

Exactly one source must be given. Files are read as latin-1, and trailing whitespace is stripped
from every line.

### Checking for errors

By default (`check=True`) the constructor calls `check()`. It raises an `ExceptionGroup` of
`OncvpspOutputError`s if the output has:

- an error, or
- no error but is incomplete: plotting data missing or cut short, or, for the all-electron stage,
  no data file written. This gives the synthetic error `incomplete_output`.

An error printed for a test configuration of a completed run is not raised. ONCVPSP moves on to
the next configuration, so the pseudopotential is unaffected. With `werror=True`, those errors and
all warnings are raised as well.

Pass `check=False` to parse a failed run, or `check(doraise=False)` to get the list without raising.
`parse(check=True, werror=False)` checks, then returns `output`.

Each `OncvpspOutputError` has `name`, `subroutine`, `description`, `line_numbers` (1-based),
`context_lines`, `context_line_numbers` and `error_lines`.

## Laziness

Every quantity is a `functools.cached_property`. It is parsed on first access, then cached, and
nothing else is parsed with it. Two things are not lazy:

- **The message scan.** Every quantity is parsed from `clean_lines`: the output without the lines of
  warnings and errors, which ONCVPSP can print in the middle of a table. So the first access to any
  quantity scans the whole output for all messages first. The scan skips the pseudopotential files
  written into the output, and it is done once per parser.
- **`check=True`.** The check needs `errors` and `completed`. `completed` needs `missing_plot_data`,
  and that parses `log_derivatives`. A parser constructed with `check=False` reads only the lines.

Some quantities build on others, and so parse them too. For example, `wavefunctions` uses `input`,
`reference_configuration`, `test_configurations` and the projector states, and most quantities need
`input` to know `nc`, `nv`, `lmax` or `nproj`. `output` evaluates every property in the table below.

Rough cost per output, measured on the 401 outputs of a full smooooth test run (CPU time, so the
figures are approximate):

| Access | Cost |
| --- | --- |
| Construct with `check=False` (read the lines) | ~2 ms |
| Message scan (first access to anything) | ~10 ms |
| Construct with `check=True` | ~25 ms |
| `output` (everything) | ~80 ms |

To parse less, construct with `check=False` and access only the properties you need. Do not call
`output` or `to_dict()`.

## Conventions

**Channels.** Per-channel records carry `l` and `j`. `j` is `None` for an l-channel: every
non- and scalar-relativistic output, and the scalar and spin-orbit coefficients of a
fully-relativistic one. It is `l ± 1/2` for a j-channel. Per-projector records add `iproj`, starting
at 1. See `CONTEXT.md` for the terms.

**Absent quantities.** A quantity that is not in the output comes back empty, never as a guess:

| Type | Empty value |
| --- | --- |
| scalar, or a dict of one block | `None` |
| list | `[]` |
| dict of arrays or parameters | `{}` |
| string | `""` |
| missing field of a record | `None` (`nan` in the metadata fields of `wavefunctions` and `vkb_projectors`) |

Exceptions:

- `reference_configuration` raises `RuntimeError` if its table is missing.
- `program_information` raises `ValueError` for a program other than ONCVPSP or METAPSP.
- Quantities that need the input raise `KeyError` when the output has no input echo. These are
  `d2exc`, `teter_parameters`, the potentials and the projectors.

## Run status and messages

| Property | In `output` | Type | Present when |
| --- | --- | --- | --- |
| `completed` | `completed` | `bool` | Always. True if all plotting data was written, or, for the all-electron stage, the data file. The pseudopotential file is not required, since `psfile` may be `none`. |
| `missing_plot_data` | `missing_plot_data` | `list[str]` | Non-empty only for an incomplete run. Possible entries: `DATA FOR PLOTTING`, `unscreened_semilocal_potentials`, `charge_densities`, `wavefunctions`, `convergence_profiles`, `log_derivatives`, `gnuplot_script`. |
| `stage` | `stage` | `"ae"`, `"ps"` or `None` | `"ae"` or `"ps"` only for the split executables of azadoks/oncvpsp (`oncvpsp*_ae.x`, `oncvpsp*_ps.x`); `None` for the whole program. |
| `ae_data_file` | `ae_data_file` | `str \| None` | All-electron stage only. |
| `errors` | `errors` | `list[dict]` | When errors are printed. Each dict has the keys of `OncvpspOutputError` (above). Matched against the 75 definitions in `ERRORS`. |
| `warnings` | `warnings` | `list[dict]` | Likewise, against the 30 definitions in `WARNINGS`. |
| `messages` | — | `list[OncvpspMessage]` | The warnings and errors in line order, each attributed to a `stage`, `scope`, channel (`l`, `j`, `iproj`, `n`), `config_index` and the input parameters it implicates (`implicated_parameters`, flattened keys such as `pseudopotentials__rc__1`). |
| `failure` | — | `OncvpspFailure \| None` | `None` for a completed run. Otherwise: the first error (`primary`), all errors, and the last stage and `l` reached. A run that stops without a recognized error fails with its last `<routine>: ERROR ...` line as `unrecognized_error`, attributed by its position, or else a synthetic `incomplete_output`. |
| `last_stage_reached` | — | `Stage` | Always. |
| `test_configuration_status` | `test_configuration_status` | `list[dict]` | One per test configuration, the reference first (`config_index` 0): `completed`, and the `messages` printed while it was calculated. |

A message is a warning or an error by what the program does next, not by its wording. A "WARNING"
after which the program stops is an error; an "ERROR" after which it continues is a warning.

## Program and input

| Property | In `output` | Type | Present when |
| --- | --- | --- | --- |
| `program_information` | `program_information` | `dict` | Always. `program`, `version`, `date`, `relativistic_treatment`, `build`. |
| `build_information` | `program_information["build"]` | `dict \| None` | azadoks/oncvpsp builds only: `executable`, `relativistic_treatment`, `commit`, `source` (`clean`, `modified` or `unknown`), `source_diff`, `compiled`. |
| `is_relativistic` | — | `bool` | Always. True for fully-relativistic outputs. |
| `input` | `input` | `dict` | Whenever the input is echoed, which every run does. It is the `OncvpspInput` model dumped as a dict: `oncvpsp`, `reference_configuration`, `pseudopotentials`, `local_potential`, `vkb_projectors`, `model_core_charge`, `log_derivative_analysis`, `test_configurations`, `output_grid`, plus the option blocks of azadoks/oncvpsp (e.g. `log_mesh`). |
| `projector_layout` | — | `ProjectorLayout \| None` | Whenever `input` is non-empty. The projector state of every projector (valence state, wellstate or localized bound state) as the program will choose it, predicted from the input. See `pyoncvpsp.projectors`. |
| `core_quantum_numbers`, `valence_quantum_numbers` | — | `list[(n, ChannelKey)]` | Whenever `input` is non-empty. |

The relativistic treatment comes from the build line when there is one. The header of the
non-relativistic ONCVPSP 4.0.1 executable wrongly says scalar-relativistic.

## All-electron reference

| Property | In `output` | Type | Present when |
| --- | --- | --- | --- |
| `reference_configuration` | `reference_configuration` | `list[dict]` | Always (it raises if missing). One per state and channel: `n`, `l`, `j`, `f`, `eig_ae`, plus `eig_pbe` for METAPSP. |
| `ae_reference_energies` | `ae_reference_energies` | `dict \| None` | Every version prints `iterations` and `total_energy`. `screening_residual` and `terms` (`total`, `kinetic`, `electron_nucleus`, `hartree`, `exchange_correlation`) come only from azadoks/oncvpsp builds from 35f1150 on; otherwise they are `None`. |

## Semilocal stage

| Property | In `output` | Type | Present when |
| --- | --- | --- | --- |
| `wellstate_metadata` | `wellstate_metadata` | `list[dict]` | When a projector is built from an unbound state (a `Wellstate for l =` block). Fields: `n`, `l`, `j`, `iproj`, `eig_ae`, `energy_guess`, `asymptotic_potential`, `half_point_radius`. In fully-relativistic v4, a j = l − 1/2 state is solved in the well of its partner: its `eig_ae` is `None`, and the printed target energy is `energy_guess`. |
| `localized_bound_states` | `localized_bound_states` | `list[dict]` | When an unoccupied bound state is used in place of a wellstate. The program reports it in a warning, or, in v2/v3, with a "negative energy" or "Normal bound state" note. Same fields as `wellstate_metadata`; the potential and radius are always `None`. |
| `optimization_diagnostics` | `optimization_diagnostics` | `list[dict]` | One per optimized pseudo-wavefunction, in all versions. Fields: `constraints` (`pswf`, `aewf`, `diff` arrays), `norm_fraction`, `qcut`, `ecut`, `q_infinity_ke`, `e_inf`, `residual_ke_error`, `ke_fourier`, `ke_rspace`, `ke_ratio`, `v_ps_rc`, `v_ae_rc`, `v_diff_rc`. A field that is not printed is `None`. |
| `convergence_profiles` | `plot_data["convergence_profiles"]` | `list[dict]` | `l`, `j`, `iproj`, `ecut`, `eresid`. Contains the tables printed by `optimize` for every projector, then the `!C` plotting data, which covers only the first projector of each `l` (j = l + 1/2 if relativistic). A table with no row is kept as empty arrays. |
| `ecut_recommended` | — | `float` | The largest Ecut, over all profiles, at which `eresid` reaches 1e-5 Ha. It is `inf` if a profile never reaches 1e-5, and `-inf` if there are no profiles. |

## VKB projectors

| Property | In `output` | Type | Present when |
| --- | --- | --- | --- |
| `vkb_projector_coefficients` | — (in `vkb_projectors[*]["coeff"]`) | `list[dict]` | One per projector of each channel with a VKB block: `l`, `j`, `iproj`, `coefficient`. Empty when the program prints no VKB blocks, as with v3 and `nproj = 1` everywhere. |
| `vkb_hermiticity_errors` | `vkb_hermiticity_errors` | `list[dict]` | Channels with two or more projectors: `l`, `j`, `iproj`, `jproj`, `hermiticity_error`. v3 prints only the (1, 2) element. |
| `vkb_scalar_projector_coefficients` | same name | `list[dict]` | Fully-relativistic only. One per l-channel: `l`, `j=None`, `coefficients`. |
| `vkb_spin_orbit_projector_coefficients` | same name | `list[dict]` | Likewise. |

## Model core charge

| Property | In `output` | Type | Present when |
| --- | --- | --- | --- |
| `d2exc` | `d2exc` | `dict` | ONCVPSP with `icmod` 1–4. Arrays of nv × nv: `ae`, `ps` and `ps_with_nlcc`. `{}` for `icmod` 0, METAPSP (`icmod` 5), and the budgeted cores (6, 7). |
| `d2exc_rmse` | `d2exc_rmse` | `float \| None` | Likewise. |
| `teter_parameters` | `teter_parameters` | `dict` | `icmod` 3 gives `fcfact` and `rcfact` (from the input), plus `rmatch` and `rhocmatch`. `icmod` 4 gives the optimized `fcfact` and `rcfact`. Other `icmod`s give at most `rmatch` and `rhocmatch`. For `icmod` 2 the meanings differ. |
| `teter_coarse_grid` | `teter_coarse_grid` | `dict` | `icmod` 4 only: `fcfact`, `rcfact`, and `d2exc_rmse` (2-D). |
| `budgeted_core` | `budgeted_core` | `dict \| None` | azadoks/oncvpsp with `icmod` 6 or 7. Gives the budget, the radius, the fill and its coefficients, and, for `icmod` 7, the scan of `candidates`; the docstring lists every field. Fields printed only by some builds are `None`. |

## Local potential tail (azadoks/oncvpsp)

| Property | In `output` | Type | Present when |
| --- | --- | --- | --- |
| `ionic_tail` | `ionic_tail` | `dict \| None` | Builds with the ionic tail: `r_start`, `r_end`, `steps`. |
| `coulomb_tail` | `coulomb_tail` | `dict \| None` | Likewise: `alpha_z`, `truncation` (list of `r`, `change`), `r_coulomb`, `max_nonionic`. |

## Diagnostics and ghosts

| Property | In `output` | Type | Present when |
| --- | --- | --- | --- |
| `pseudoatom_total_energy` | `pseudoatom_total_energy` | `float \| None` | Every completed run, and runs that fail after the diagnostics. |
| `diagnostic_tests_semilocal` | same name | `list[dict]` | v2 and v3 only. `l`, `j`, `rc`, `rmatch`, `e_in`, `e_test`, `norm_test`, `slope_test`. |
| `diagnostic_tests_vkb` | same name | `list[dict]` | All versions. Same fields, but `e_test` (v3) is replaced by `delta_e` (v4, METAPSP); the other is `None`. |
| `ghosts` | `ghosts` | `list[dict]` | Rows flagged `WARNING - GHOST`: `l`, `j`, `eig`, `ecut`, `sign`, `r_rc` (unbound ghosts only). v3 tests only for bound ghosts and prints no sign; it is reported as `-`. |
| `bound_ghost_tests` | same name | `list[dict]` | Every row of the bound-ghost tables (v4, METAPSP, later v3): `set` (`+`/`-` for the j-sets of fully-relativistic outputs), `n`, `l`, `eig`, `eig_basis`, `ecut`. |
| `ghost_basis_cuts` | same name | `list[dict]` | azadoks/oncvpsp, when the ghost test's basis is cut below a barrier state it cannot converge (warning `ghost_barrier_state_not_converged`): `n`, `l`, `e_barrier`, and `ecut`, the energy the ghost test reaches. |
| `diagnostics` | `diagnostics` | `dict` | azadoks/oncvpsp builds that print the `Begin DIAGNOSTICS` block. The values are at full precision: `ecut_crossings`, `log_derivatives`, `ghost_margins`, `ghost_localization`, `ecut_weighted`, `projector_conditioning`, `test_configurations`, `model_core`, `vloc_cutoff`. A part that is not printed is `[]` or `None`; the docstring lists every field. |
| `diagnostics_ld_table` | — | `dict[str, np.ndarray]` | The `ld_r`/`ld` lines of the DIAGNOSTICS block as a tidy table, the same as `oncvpsp_ld.x` gives (see below): columns `l`, `ikap`, `r`, `e_ha`, `phase_ae`, `phase_ps`, empty without the block. |

## Test configurations

| Property | In `output` | Type | Present when |
| --- | --- | --- | --- |
| `test_configurations` | same name | `list[list[dict]]` | ONCVPSP (METAPSP prints none). One list per configuration, the reference first, of `n`, `l`, `j`, `f`, `eig_ae`, `eig_ps`. |
| `test_configuration_energies` | same name | `list[dict \| None]` | Indexed like `test_configurations`. Fields: `etot_ae_ref`, `etot_ae_test`, `etot_ae_diff`, the same three for `ps`, and `excitation_error`. `None` for a configuration without results. |
| `test_configuration_status` | same name | `list[dict]` | See "Run status and messages". |

## Plotting data

Everything below is in `output["plot_data"]` under the same name. `plot_data` is itself a property.
All of it is missing from a run that stops before `DATA FOR PLOTTING`.

| Property | Type | Present when |
| --- | --- | --- |
| `unscreened_semilocal_potentials` | `list[dict]` | Always (`!p`). One per l up to `lmax`: `l`, `j=None`, `r`, `v_sl`. |
| `semilocal_potentials` | `list[dict]` | Screened (`!ps`). No version in the test data prints these, so it is `[]`. |
| `local_potential` | `dict` | `r` and `v_loc` (`!L`), merged with the input's `local_potential` parameters. So it is never empty, even without the data. |
| `charge_densities` | `dict` | `r`, `rho_val`, `rho_core`, `rho_model_core` (`!r`). |
| `kinetic_energy_densities` | `dict` | METAPSP only (`!t`): `r`, `tau_ps`, `tau_mod`. |
| `kinetic_energy_potentials` | `dict` | METAPSP only (`!vt`): `r`, `vtau_ps`, `vtau_mod`. |
| `wavefunctions` | `list[dict]` | One per plotted state (`&`): `r`, `wfn_ae`, `wfn_ps`, and the state's `n`, `l`, `j`, `iproj` and `is_bound`. Also `f`, `eig_ae`, `eig_ps`, `asymptotic_potential`, `half_point_radius` and `r_rc`, which are `nan` when unknown. |
| `optimized_pseudo_wavefunctions` | `list[dict]` | v2 and v3 only (`@`), channels with `nproj = 2`: `l`, `j`, `iproj`, `r`, `wfn_ps`. |
| `vkb_projectors` | `list[dict]` | v4 and METAPSP only (`!J`): `l`, `j`, `iproj`, `r`, `proj`, `coeff`. |
| `convergence_profiles` | `list[dict]` | See "Semilocal stage". |
| `log_derivatives` | `list[dict]` | `l`, `j`, `rxpsh`, `e`, `log_deriv_ae`, `log_deriv_ps`. The number of channels depends on the version. |

## Text blocks

| Property | In `output` | Present when |
| --- | --- | --- |
| `gnuplot_script` | `gnuplot_script` | Every completed run. |
| `plot_string` | — | The raw `DATA FOR PLOTTING` block. Present when `gnuplot_script` is. |
| `upf_string` | `upf` | `psfile = upf` or `both`. From `<UPF version="2.0.1">` to `</UPF>`. |
| `psp8_string` | `psp8` | `psfile = psp8` or `both`, in versions that end the file with its `</INPUT>` echo (v3.3, v4, METAPSP). |
| `psml_string` | `psml` | azadoks/oncvpsp builds that print a `Begin PSML` … `END_PSP` block. |

To parse the files themselves, use `pyoncvpsp.io.parse_psp8` and `parse_psml`.

## Lines and contexts

| Property | Type | Contents |
| --- | --- | --- |
| `raw_lines` | `list[str]` | The output as read. |
| `clean_lines` | `list[str]` | The output without the message lines. Indices into it are not indices into `raw_lines`. |
| `line_contexts` | `list[LineContext]` | One per raw line: the `stage`, the channel being processed, the test configuration, and the enclosing block. |
| `search(pattern, start=0)` | `(index, Match) \| None` | The first line of `clean_lines` that matches. |
| `findall(pattern, start=0, stop=-1)` | `list[(index, Match)]` | Every line of `clean_lines[start:stop]` that matches. |

`search` and `findall` first narrow the lines with a literal that every match must contain, so a
pattern with a long literal is fast.

## `oncvpsp_ld.x`

`oncvpsp_ld.x` (azadoks/oncvpsp) computes the log derivatives at any radii and energies from the
data files of a split run (`oncvpsp*_ae.x -a`, `oncvpsp*_ps.x -p`). Its output is parsed by
`pyoncvpsp.io.parse_oncvpsp_ld(text, returncode=None)`. That returns the same table as
`diagnostics_ld_table`, one row per energy, channel and radius, so one function can take either.
It raises `OncvpspLdError` for an error line (`<routine>: ERROR ...`), a nonzero `returncode`, or
an output without log derivatives. `ld_table(lines)` reads the `ld_r`/`ld` lines of any text, and
`ld_command(ae_file, ps_file, radii=, channels=, energies=, executable=)` builds the command line.

```python
import subprocess
from pyoncvpsp.io import ld_command, parse_oncvpsp_ld

run = subprocess.run(ld_command("Si.ae", "Si.ps", radii=[2.5, 3.0]), capture_output=True, text=True)
table = parse_oncvpsp_ld(run.stdout, run.returncode)
```

## Known gaps

- **Old psp8 files.** `psp8_string` is `""` for v2 and v3 before 3.3. Their psp8 file has no
  `</INPUT>` line, and the parser searches for one.
- **METAPSP PSML.** `psml_string` is `""` for METAPSP with `psfile = psml`. It writes the XML
  without the `Begin PSML` line.
- **Slower message scan.** For the same two reasons, the scan does not skip these files and
  searches them for messages. The results are unchanged; the scan is only slower.
- **Messages inside the pseudopotential files.** The scan skips each complete file in the output.
  A warning printed while one is written would be missed, but no version prints any.
- **`semilocal_potentials`.** It is never filled from the test outputs.
