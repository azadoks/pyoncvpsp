# pyoncvpsp

Reads and writes the inputs of the ONCVPSP and METAPSP pseudopotential generators, and parses
their text outputs into structured data, including where and why a generation failed. It covers
ONCVPSP v2, v3 and v4, METAPSP, and the azadoks/oncvpsp builds.

## Installation

```sh
pip install git+https://github.com/azadoks/pyoncvpsp
```

Python 3.11 or later. For development, `uv sync` installs the package and its dev tools, and
`uv run pytest` runs the tests.

## Use

```python
from pyoncvpsp import OncvpspInput, OncvpspTextParser

inp = OncvpspInput.from_dat_file("Si.dat")
inp.to_toml_file("Si.toml")

parser = OncvpspTextParser(path="Si.out", check=False)
parser.failure                     # None for a completed run, else the first error and its cause
parser.optimization_diagnostics    # one quantity, parsed on first access
parser.output                      # everything, as a dict
```

The same from the command line:

| Command | Does |
| --- | --- |
| `pyoncvpsp dat2toml -i Si.dat -o Si.toml` | `.dat` input to TOML |
| `pyoncvpsp toml2dat -i Si.toml -o Si.dat` | TOML input to `.dat` |
| `pyoncvpsp format-dat -i Si.dat -o Si.dat` | rewrites a `.dat` input in the standard layout, with comments |
| `pyoncvpsp out2json -i Si.out -o Si.json [--pretty] [--gzip]` | parses an output to JSON |
| `pyoncvpsp extract_upf -i Si.out -o Si.upf` | the UPF file written into an output |

Without `-o`, each prints its result. `out2json` and `extract_upf` stop with the output's errors
for a failed run.

## Documentation

- [The input model](docs/input.md): sections, keys, the options of azadoks/oncvpsp, the checks.
- [The output parser](docs/output_parser.md): everything parsed, when it is present, the run
  status and messages, and what each costs; also `oncvpsp_ld.x` and `oncvpsp_confine.x`.
- [CONTEXT.md](CONTEXT.md): the terms (l-channel, projector state, wellstate, ...).
