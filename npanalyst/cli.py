import json
from pathlib import Path
from typing import Optional

import click

from npanalyst import __version__ as VERSION
from npanalyst import core, configuration
from npanalyst.logging import setup_logging, get_logger

logger = get_logger()

# from npanalyst import activity, core, utils
# from npanalyst.convert import default, gnps, mzmine

# Note each task has a Click function (controller) calling
# the underlying businsess logic function to allow for
# calling purely python functions instead of system calls
# in the web implementation


############################
##      CLI Definition
############################
@click.group()
@click.version_option(version=VERSION)
def cli():
    """NPAnalyst CLI entrypoint"""
    click.echo("Welcome to NPAnalyst!")


############################
##      Get Config
############################
@cli.command("get_config")
@click.option(
    "--output_path",
    "-o",
    type=Path,
    required=False,
    help="Output directory",
    default=".",
    show_default=True,
)
def get_config(output_path: Optional[Path] = None):
    """Helper function to generate a config file"""
    click.echo("Generating configuration file `./config.json`")
    if output_path is None:
        conf_path = Path() / "config.json"
    else:
        conf_path = output_path / 'config.json'
    if conf_path.exists():
        click.echo("ERROR: File already exists.")
        raise click.Abort()
    elif not output_path.exists():
        output_path.mkdir(parents=True)
    with conf_path.open("w") as f:
        f.write(json.dumps(configuration.load_raw_config(), indent=2))


############################
##      REPLICATE COMPARISON
############################
@cli.command("replicate")
@click.option(
    "--input_path",
    "-i",
    type=Path,
    required=True,
    help="Path to directory with input mzML files",
)
@click.option(
    "--output_path",
    "-o",
    type=Path,
    required=False,
    help="Output directory",
    default=".",
    show_default=True,
)
@click.option(
    "--config",
    type=Path,
    help="Configuration file to use. "
    "Arguments will overwrite an overlapping config file options",
)
@click.option(
    "--workers",
    "-w",
    help="Number of parallel workers",
    type=int,
    default=-2,
    show_default=True,
)
@click.option(
    "--verbose/--no-verbose",
    "-v/",
    default=False,
    show_default=True,
    help="Verbose or quiet logging",
)
def run_replicate_command(
    input_path: Path,
    workers: int,
    verbose: bool,
    output_path: Optional[Path] = None,
    config: Optional[Path] = None,
):
    """Run replication comparison on input mzML data."""
    if output_path is None:
        output_path = Path()
    run_replicate(
        input_path=input_path,
        output_path=output_path,
        workers=workers,
        verbose=verbose,
        config=config,
    )


def run_replicate(
    input_path: Path,
    output_path: Path,
    workers: int,
    verbose: bool,
    config: Optional[Path] = None,
):
    """Run replication comparison on input mzML data."""
    if not output_path.exists():
        output_path.mkdir(parents=True)
    setup_logging(verbose=verbose, fpath=output_path / "npanalyst.log")
    configd = configuration.load_config(config_path=config)
    core.process_replicates(
        input_path,
        output_path,
        configd,
        max_workers=workers,
    )


############################
##      BASKETING
############################
@cli.command("basket")
@click.option(
    "--input_path",
    "-i",
    type=Path,
    required=True,
    help="Path to directory with `replicated` folder",
)
@click.option(
    "--output_path",
    "-o",
    type=Path,
    required=False,
    default=".",
    show_default=True,
    help="Output directory",
)
@click.option(
    "--config",
    type=Path,
    help="Configuration file to use. "
    "Arguments will overwrite an overlapping config file options",
)
@click.option(
    "--verbose/--no-verbose",
    "-v/",
    default=False,
    show_default=True,
    help="Verbose or quiet logging",
)
def run_basketing_command(
    input_path: Path,
    verbose: bool,
    output_path: Optional[Path] = None,
    config: Optional[Path] = None,
):
    """Run basketting from replicate compared input data."""
    if output_path is None:
        output_path = Path()
    run_basketing(
        input_path=input_path, output_path=output_path, verbose=verbose, config=config
    )


def run_basketing(
    input_path: Path,
    output_path: Path,
    verbose: bool,
    config: Optional[Path] = None,
):
    """Run basketting from replicate compared input data."""
    if not output_path.exists():
        output_path.mkdir(parents=True)
    setup_logging(verbose=verbose, fpath=output_path / "npanalyst.log")
    configd = configuration.load_config(config_path=config)
    core.basket_replicated(input_path, output_path, configd)


############################
##      IMPORT
############################
# TODO: Implement
@cli.command("import")
@click.option(
    "--input_path",
    "-i",
    type=Path,
    required=True,
    help="Path to input file",
)
@click.option(
    "--output_path",
    "-o",
    help="Output directory",
    type=Path,
    default=".",
    show_default=True,
)
@click.option(
    "--mstype",
    "-t",
    type=click.Choice(["GNPS", "MZmine"], case_sensitive=False),
    help="Select an import data format",
)
@click.option(
    "--verbose/--no-verbose",
    "-v/",
    default=False,
    show_default=True,
    help="Verbose or quiet logging",
)
def run_import_command(
    input_path: Path,
    mstype: str,
    verbose: bool,
    output_path: Optional[Path] = None,
):
    """Run import of MZmine for GNPS input formats to standard basket format.
    Use this prior to the `activity` step.
    """
    if output_path is None:
        output_path = Path()
    run_import(
        input_path=input_path, output_path=output_path, verbose=verbose, mstype=mstype
    )


def run_import(
    input_path: Path,
    output_path: Path,
    mstype: str,
    verbose: bool,
):
    """Run import of MZmine for GNPS input formats to standard basket format."""
    if not output_path.exists():
        output_path.mkdir(parents=True)
    setup_logging(verbose=verbose, fpath=output_path / "npanalyst.log")
    core.import_data(input_path, output_path, mstype)


############################
##      ACTIVITY
############################
@cli.command("activity")
@click.option(
    "--input_path",
    "-i",
    type=Path,
    required=True,
    help="Path to basketed input file",
)
@click.option(
    "--output_path",
    "-o",
    type=Path,
    required=False,
    help="Output directory",
    default=".",
    show_default=True,
)
@click.option(
    "--activity",
    "-a",
    type=Path,
    required=True,
    help="Path to activity input file",
)
@click.option(
    "--config",
    type=Path,
    help="Configuration file to use. "
    "Arguments will overwrite an overlapping config file options",
)
@click.option(
    "--verbose/--no-verbose",
    "-v/",
    default=False,
    show_default=True,
    help="Verbose or quiet logging",
)
@click.option(
    "--include_web_output/--no_include_web_output",
    "-w/",
    default="False",
    show_default=True,
    help="Include web formats during final save stage of outputs?",
)
def run_activity_command(
    input_path: Path,
    activity: Path,
    verbose: bool,
    include_web_output: bool,
    output_path: Optional[Path] = None,
    config: Optional[Path] = None,
):
    """Run activity integration from standard input.

    Input format is ~= to basketed format.

    If not using mzML pipeline, first run `import` for MZmine or GNPS inputs.
    """
    if output_path is None:
        output_path = Path()
    run_activity(
        input_path=input_path,
        output_path=output_path,
        activity_path=activity,
        verbose=verbose,
        config=config,
        include_web_output=include_web_output,
    )


def run_activity(
    input_path: Path,
    output_path: Path,
    activity_path: Path,
    verbose: bool,
    include_web_output: bool,
    config: Optional[Path] = None,
):
    """Run activity integration from standard input format."""
    if not output_path.exists():
        output_path.mkdir(parents=True)
    setup_logging(verbose=verbose, fpath=output_path / "npanalyst.log")
    configd = configuration.load_config(config_path=config)
    core.bioactivity_mapping(
        basket_path=input_path,
        output_dir=output_path,
        activity_path=activity_path,
        configd=configd,
        include_web_output=include_web_output,
    )
