"""Shared fixtures and helpers for the pyoncvpsp test suite."""

import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyoncvpsp.io.output import OncvpspTextParser

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = DATA_DIR / "outputs"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
OUTPUT_NAMES = sorted(
    path.name.removesuffix(".out.gz") for path in OUTPUT_DIR.glob("*.out.gz")
)


def pytest_addoption(parser):
    parser.addoption(
        "--update-snapshots",
        action="store_true",
        default=False,
        help="Rewrite the parsed-output snapshots instead of comparing against them.",
    )


def read_output(name: str) -> str:
    """Text of the test output `name` (without the `.out.gz` suffix)."""
    with gzip.open(OUTPUT_DIR / f"{name}.out.gz", "rt", encoding="latin-1") as f:
        return f.read()


_PARSERS: dict[str, OncvpspTextParser] = {}


def parse_output(name: str) -> OncvpspTextParser:
    """Shared (cached) parser for the test output `name`. Do not mutate its results."""
    if name not in _PARSERS:
        _PARSERS[name] = OncvpspTextParser(text=read_output(name), check=False)
    return _PARSERS[name]


def pytest_generate_tests(metafunc):
    """Run tests taking `output_name` for every test output, and tests taking
    `completed_output_name` for every test output which completed successfully."""
    if "output_name" in metafunc.fixturenames:
        metafunc.parametrize("output_name", OUTPUT_NAMES)
    if "completed_output_name" in metafunc.fixturenames:
        completed = [name for name in OUTPUT_NAMES if parse_output(name).completed]
        metafunc.parametrize("completed_output_name", completed)


@pytest.fixture(name="read_output")
def read_output_fixture():
    """Function returning the text of a test output by name."""
    return read_output


@pytest.fixture(name="parse_output")
def parse_output_fixture():
    """Function returning the (shared) parser of a test output by name."""
    return parse_output


@pytest.fixture
def snapshot(request):
    """Function comparing a parsed output against its stored snapshot (or rewriting the snapshot
    with `--update-snapshots`)."""
    update = request.config.getoption("--update-snapshots")

    def check(name: str, output: Any) -> None:
        summary = summarize(output)
        if update:
            write_snapshot(name, summary)
        # Compare through JSON so that the summary has the same types as the snapshot
        assert_close(json.loads(json.dumps(summary)), load_snapshot(name))

    return check


def summarize(value: Any) -> Any:
    """Reduce parsed output to a compact, JSON-serializable form for snapshot comparison.
    Arrays are reduced to their shape, NaN count and sums; long strings to their length and hash.
    """
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=float)
        return {
            "shape": list(array.shape),
            "nan": int(np.isnan(array).sum()),
            "sum": summarize(float(np.nansum(array))),
            "abs_sum": summarize(float(np.nansum(np.abs(array)))),
        }
    if isinstance(value, dict):
        return {str(key): summarize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [summarize(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str) and len(value) > 200:
        return {"len": len(value), "sha256": hashlib.sha256(value.encode()).hexdigest()}
    return value


def assert_close(actual: Any, expected: Any, path: str = "$") -> None:
    """Recursively compare snapshot summaries, allowing for floating-point round-off."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert actual.keys() == expected.keys(), f"{path}: keys differ"
        for key in expected:
            assert_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list), path
        assert len(actual) == len(expected), (
            f"{path}: length {len(actual)} != {len(expected)}"
        )
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_close(a, e, f"{path}[{i}]")
    elif isinstance(expected, float) and not isinstance(actual, bool):
        assert isinstance(actual, (int, float)), path
        assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12), (
            f"{path}: {actual} != {expected}"
        )
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def load_snapshot(name: str) -> Any:
    with open(SNAPSHOT_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def write_snapshot(name: str, summary: Any) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SNAPSHOT_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
        f.write("\n")
