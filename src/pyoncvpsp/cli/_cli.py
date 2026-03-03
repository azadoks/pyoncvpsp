"""pyoncvpsp command line interface implementation"""

import click

from ._dat2toml import dat2toml
from ._format_dat import format_dat
from ._toml2dat import toml2dat
from ._out2json import out2json
from ._extract_upf import extract_upf


@click.group()
def cli() -> None:
    """Main CLI entry point for pyoncvpsp."""


cli.add_command(dat2toml)
cli.add_command(toml2dat)
cli.add_command(format_dat)
cli.add_command(out2json)
cli.add_command(extract_upf)

if __name__ == "__main__":
    cli()
