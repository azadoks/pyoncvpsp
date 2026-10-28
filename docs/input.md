# The input model

`pyoncvpsp.OncvpspInput` holds the input of ONCVPSP (v3, v4 and the azadoks/oncvpsp builds) and
METAPSP. It reads and writes both the `.dat` text input and the TOML input that azadoks/oncvpsp
reads with `oncvpsp.x -t`, and checks the input against the rules of the programs. This page lists
the sections and their keys, the options only azadoks/oncvpsp reads, and the checks.

Units are those of the programs: energies in Hartree, radii in bohr.

## Reading and writing

```python
from pyoncvpsp import OncvpspInput

inp = OncvpspInput.from_dat_file("Si.dat")     # or from_dat_string, from_toml_file, from_toml_string
inp.to_toml_file("Si.toml")                    # or as_toml_string()
inp.to_dat_file("Si.dat")                      # or as_dat_string(with_comment=True)
inp.as_dict()                                  # nested dict; as_dict(flat=True) for flat keys
OncvpspInput.from_dict(data)
inp.merge({"pseudopotentials": {"qcut": [6.0, 6.0, 6.0]}})   # a new input, these keys replaced
```

The input of a run is also in its output: `OncvpspTextParser(...).input` is `as_dict()` of the
echoed input, so `OncvpspInput.from_dict(parser.input)` rebuilds it.

**Flat keys.** `as_dict(flat=True)` joins the keys with `__`, one key per list element:
`pseudopotentials__rc__1` is `rc` of l = 1. The `implicated_parameters` of the output parser's
messages use the same keys.

**Writing.** The options of azadoks/oncvpsp (below) are written only when they are not their
defaults, so an input without them gives a `.dat` file that ONCVPSP 4.0.1 reads. They go at the
end of their line, and a later one forces the earlier ones out: on the lmax line, `well_norm`
forces out `wave_regularization` at 0; on the icmod line, `slack` forces out `rcfact`, at 1e-5
if it is 0, since the program rejects 0 there. The TOML is in the layout ONCVPSP reads (see
below); TOML written by older pyoncvpsp, with the model's own section names, is still read.

## Sections

The TOML sections are the fields of `OncvpspInput`, except where the table says otherwise. Lists
have one value per l (`pseudopotentials`, `vkb_projectors`) or per state (configurations).

| Section (model field) | TOML | Keys | `.dat` |
| --- | --- | --- | --- |
| `oncvpsp` | `[oncvpsp]` | `atsym`, `z`, `nc`, `nv`, `iexc`, `c_light`; `psfile` goes in `[pp_output]` | first line |
| `reference_configuration` | same | lists `n`, `l`, `f` | the `nc + nv` state lines |
| `pseudopotentials` | same | `lmax`; lists `l`, `rc`, `ep`, `ncon`, `nbas`, `qcut`; the well options below | lmax line, then one line per l |
| `local_potential` | same | `lloc`, `lpopt`, `rcloc`, `dvloc0` | one line |
| `vkb_projectors` | same | lists `l`, `nproj`, `debl`, `well_n`, `ecut_weight` | one line per l |
| `model_core_charge` | same | `icmod`, `fcfact`, `rcfact`, `slack`, `budget_cutoff`, the icmod 4 grid | icmod line, with the grid of icmod 4 on it |
| `log_derivative_analysis` | same | `epsh1`, `epsh2`, `depsh`, `rxpsh` | one line |
| `output_grid` | `[linear_mesh]` | `rlmax` as `rmax`, `drl` as `a`; `upf_mesh` goes in `[pp_output]` | one line |
| `test_configurations` | `[[test_configurations]]` | lists `n`, `l`, `f` of the valence states | `ncnf`, then each configuration |
| `log_mesh` | `[log_mesh]` | `amesh`, `rr1`, `rmax`, `adams_order` (4 to 8); left out when none is given | optional last line (`adams_order` its optional fourth value) |

In the TOML, the icmod 4 grid is `fcfact_min`, `fcfact_max`, `fcfact_step`, `rcfact_min`,
`rcfact_max`, `rcfact_step` (the model's `teter_amp_*` and `teter_scale_*`). `teter_relative`
is written but ONCVPSP ignores it; the `.dat` input has relative grids only.

The docstring of each section's model (`PseudoPotentialsInput`, `ModelCoreChargeInput`, ...)
describes every key.

## Options of azadoks/oncvpsp

ONCVPSP 4.0.1 and METAPSP do not read these. At their defaults the programs behave as 4.0.1.
`doc/changes_from_4.0.1.md` of azadoks/oncvpsp gives the physics of each.

| Key | Section | `.dat` | Values (default first) |
| --- | --- | --- | --- |
| `c_light` | `oncvpsp` | 7th value of the first line | `None` (137.036); 0 or less reads as `None` |
| `iexc` 6, 7 | `oncvpsp` | | Slater + VWN5; 7 with relativistic exchange at `c_light` |
| `ep_as_given` | `pseudopotentials` | 2nd value of the lmax line, 0/1 | `false`: the well states take `ep` as given, not ONCVPSP's replacement |
| `well_onset` | `pseudopotentials` | 3rd value | `None` (1.006**5); the well's onset in units of rc, at least 1 |
| `well_tail` | `pseudopotentials` | 4th value, 0/1/2 | `"well"`, `"free"`, `"confined"`: the tail of the well states beyond rc |
| `wave_regularization` | `pseudopotentials` | 5th value | `0.0`; the weight of E_resid at qcut/4 in the optimization |
| `well_norm` | `pseudopotentials` | 6th value, 0/1/2 | `None` (that of `well_tail`), `"space"`, `"rc"` |
| `well_n` | `vkb_projectors` | 4th column | `None`; with `"confined"`, the n of each l's first well state, 0 for the program's choice |
| `ecut_weight` | `vkb_projectors` | 5th column | `None`; with `icmod` 6/7 and `fcfact <= 0`, the weight of a channel with no occupied state in the cutoff, negative for 2(2l+1) |
| `icmod` 6, 7 | `model_core_charge` | | budgeted and hardness-matched model cores |
| `slack` | `model_core_charge` | 4th value of the icmod line | `1.0`; with `icmod` 6/7 only |
| `budget_cutoff` | `model_core_charge` | 5th value, 0/1/2 | `"max"`, `"weighted"`, `"max_all"`; with `icmod` 6/7 only |
| `upf_mesh` | `output_grid` | 3rd value of the output-grid line | `0` (the linear grid); most points of the UPF file on the log mesh |
| `log_mesh` | | last line | all `None` (the built-in mesh) |
| `adams_order` | `log_mesh` | 4th value of the last line | `None` (order 4); the order, 4 to 8, of the Adams-Moulton steps |

A list of `well_n` that is all 0, or of `ecut_weight` that is all negative, reads as `None`. When
either is set, the `.dat` writer gives every projector line the column (and `well_n` at 0 before an
`ecut_weight`).

## Checks

The models check the input when they are built, and raise pydantic's `ValidationError`:

- ranges of single values (`ncon` 3–5, `nbas` 5–10, `nproj` 0–5, `icmod` 0–7, `amesh > 1`, ...);
- lists of the same length, and `l` covering 0 to its largest value;
- `nc + nv` equal to the number of reference states, from 1 to 30, and no negative ion;
- `rc >= rcloc` for `lloc = 4`, and `rlmax` not below the largest rc;
- the log mesh: `rr1` below the largest rc, `rmax` above `rlmax`;
- `fcfact > 0` for `icmod` 1 and 3, `rcfact > 0` for `icmod` 3, and a negative `fcfact`, a
  `slack` other than 1 or a `budget_cutoff` other than `"max"` only with `icmod` 6 or 7;
- `well_norm = "space"` not with `well_tail = "free"`, and `well_n` only with `"confined"`;
- with `well_tail = "free"` and `ep_as_given`, no negative `ep` for a channel whose first
  projector state is a well state (azadoks/oncvpsp from fd5aaf0 needs every free-tail well state
  at 0 Ha or above). Without `ep_as_given`, `wellstate` replaces every negative energy of a well
  state by 0.25 Ha, so the case does not arise.

They do not check what only the program can: that `well_n` is not below the program's own choice,
that `ecut_weight` is negative for every channel with an occupied state, or, with `ep_as_given`,
that a free-tail well state at a valence energy plus `debl` is not below 0 Ha. The program stops with an error for
these, which the output parser reads (`well_n_below_first_well_state`,
`ecut_weight_occupied_channel`, `free_tail_negative_ep`).

## Projector states

`pyoncvpsp.projectors.predict_projector_layout(inp, program, version, relativistic)` predicts,
from the input, the projector state of every projector: a valence state, a wellstate or a
localized bound state, its n, and whether its energy comes from the valence state, `ep` or `debl`.
It follows the rules of the source of each version, and `well_n` with the confined well.
`OncvpspTextParser.projector_layout` is the same for the input of an output.
