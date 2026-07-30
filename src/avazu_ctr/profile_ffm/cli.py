"""CLI composition for the profile FFM workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from avazu_ctr.profile_ffm.config import load_profile_ffm
from avazu_ctr.profile_ffm.pipeline import fit_predict_profile_ffm
from avazu_ctr.profile_ffm.preprocessing import prepare_profile_ffm

app = typer.Typer(no_args_is_help=True)
console = Console()


def _progress(message: str) -> None:
    console.print(f"[cyan]{message}[/cyan]")


@app.command("prepare")
def prepare_command(
    config_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Build checksummed sparse inputs from the raw Kaggle populations."""

    config = load_profile_ffm(config_path)
    manifest = prepare_profile_ffm(
        config,
        overwrite=overwrite,
        progress=_progress,
    )
    console.print(f"[green]Prepared profile FFM inputs[/green] {manifest}")


@app.command("fit-predict")
def fit_predict_command(
    config_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    preparation_manifest: Annotated[
        Path | None,
        typer.Option("--preparation-manifest", exists=True, dir_okay=False),
    ] = None,
    output: Annotated[Path | None, typer.Option("--output")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    clean_prepared: Annotated[bool, typer.Option("--clean-prepared")] = False,
) -> None:
    """Fit prediction sources sequentially and write the final submission."""

    config = load_profile_ffm(config_path)
    submission = fit_predict_profile_ffm(
        config,
        preparation_manifest=preparation_manifest,
        output=output,
        overwrite=overwrite,
        clean_prepared=clean_prepared,
        progress=_progress,
    )
    console.print(f"[green]Wrote profile FFM submission[/green] {submission}")


@app.command("reproduce")
def reproduce_command(
    config_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option("--output")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    clean_prepared: Annotated[bool, typer.Option("--clean-prepared")] = False,
) -> None:
    """Prepare, fit, and compose the configured profile FFM submission."""

    config = load_profile_ffm(config_path)
    manifest = prepare_profile_ffm(
        config,
        overwrite=overwrite,
        progress=_progress,
    )
    submission = fit_predict_profile_ffm(
        config,
        preparation_manifest=manifest,
        output=output,
        overwrite=overwrite,
        clean_prepared=clean_prepared,
        progress=_progress,
    )
    console.print(f"[green]Wrote profile FFM submission[/green] {submission}")
