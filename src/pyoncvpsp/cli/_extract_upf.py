"""CLI command to extract UPF files from ONCVPSP output files."""

from os import PathLike

import click

from pyoncvpsp.io import OncvpspTextParser


@click.command(name="extract_upf")
@click.option(
    "--input",
    "-i",
    "out_filepath",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the ONCVPSP output file to convert.",
)
@click.option(
    "--output",
    "-o",
    "upf_filepath",
    type=click.Path(dir_okay=False),
    required=False,
    help="Path to save the extracted UPF file.",
)
def extract_upf(out_filepath: PathLike, upf_filepath: PathLike | None) -> None:
    """Extract the UPF file from an ONCVPSP output file.

    Args:
        out_filepath (PathLike): Path to the ONCVPSP output file.
        upf_filepath (PathLike): Path to save the extracted UPF file.
    """
    output = OncvpspTextParser(path=out_filepath).parse()
    if upf_filepath is None:
        print(output["upf"])
    else:
        with open(upf_filepath, "w", encoding="utf-8") as f:
            f.write(output["upf"])
