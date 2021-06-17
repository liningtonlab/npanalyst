from pathlib import Path
from typing import Optional

import click

from npanalyst import __version__ as VERSION
from npanalyst import core
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
    default=-1,
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
        output_path = input_path / "output"
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
    setup_logging(verbose=verbose, fpath=Path() / "log.txt")
    configd = core.load_config(config_path=config)
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
    "--config",
    type=Path,
    help="Configuration file to use. "
    "Arguments will overwrite an overlapping config file options",
)
@click.option(
    "--verbose/--no-verbose",
    default=False,
    show_default=True,
    help="Verbose or quiet logging",
)
def run_basketing_command(
    input_path: Path,
    verbose: bool,
    config: Optional[Path] = None,
):
    """Run basketting from replicate compared input data."""
    run_basketing(input_path=input_path, verbose=verbose, config=config)


def run_basketing(
    input_path: Path,
    verbose: bool,
    config: Optional[Path] = None,
):
    """Run basketting from replicate compared input data."""
    setup_logging(verbose=verbose)
    configd = core.load_config(config_path=config)
    core.basket_replicated(input_path, configd)


############################
##      IMPORT
############################
# TODO: Implement
# @cli.command("import")
# @click.argument("input_path")
# @click.option(
#     "--output_path",
#     "-o",
#     help="Output directory",
#     default=".",
#     show_default=True,
# )
# def run_import(
#     **kwargs,
#     # input_path: Path,
# ):
#     """Run import of MZmine for GNPS input formats to standard basket format.
#     Use this prior to the `activity` step.

#     INPUT_Path is the path to the input file.
#     """
#     print(kwargs)


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
    default=False,
    show_default=True,
    help="Verbose or quiet logging",
)
def run_activity_command(
    input_path: Path,
    activity: Path,
    verbose: bool,
    config: Optional[Path] = None,
):
    """Run activity integration from standard input format.

    Input format is ~= to basketed format.

    If not using mzML pipeline, first run `import` for MZmine or GNPS inputs.
    """
    run_activity(
        input_path=input_path,
        activity_path=activity,
        verbose=verbose,
        config=config,
    )


def run_activity(
    input_path: Path,
    activity_path: Path,
    verbose: bool,
    config: Optional[Path] = None,
):
    """Run activity integration from standard input format."""
    setup_logging(verbose=verbose)
    configd = core.load_config(config_path=config)
    core.load_and_generate_act_outputs(
        basket_path=input_path, act_path=activity_path, configd=configd
    )


# TODO: Determine if needed
############################
##      HELPER FXNS
############################
# def gt_zero(ctx, param, value):
#     try:
#         assert value > 0
#         return value
#     except AssertionError:
#         raise click.BadParameter(f"{param} must be greater than 0.")


# def gte_zero(ctx, param, value):
#     try:
#         assert value >= 0
#         return value
#     except AssertionError:
#         raise click.BadParameter(f"{param} must be greater than or equal to 0.")
