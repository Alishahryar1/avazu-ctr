"""Installed command-line interface."""

from __future__ import annotations

import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from avazu_ctr.config import load_experiment
from avazu_ctr.config.schema import ExperimentConfig
from avazu_ctr.data import (
    DatasetPurpose,
    load_manifest,
    preprocess_evaluation,
    preprocess_production,
    temporal_windows,
)
from avazu_ctr.exploration import dataset_report, raw_report, run_report
from avazu_ctr.inference import Predictor, export_production_bundle
from avazu_ctr.profile_ffm.cli import app as profile_ffm_app
from avazu_ctr.tracking import (
    HoldoutEvidence,
    RunStore,
    deploy_bundle,
    load_confirmation,
    load_selection,
    write_confirmation,
    write_selection,
)
from avazu_ctr.tracking.promotion import activate_selection
from avazu_ctr.training import CandidateTrainer, ProductionRefitter
from avazu_ctr.tuning import StagedTuner, confirm_configuration

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
console = Console()
app.add_typer(profile_ffm_app, name="profile-ffm")


def _active_selected_config(control: ExperimentConfig) -> ExperimentConfig:
    active = load_selection(control.tracking.selection_dir)
    recorded = RunStore(control.tracking.database).active_selection()
    if (
        recorded is None
        or recorded["candidate_run_id"] != active.evidence.selection_id
        or recorded["candidate_evidence_sha256"] != active.evidence_sha256
    ):
        raise typer.BadParameter("selection directory is not the active recorded selection")
    selected = active.evidence.confirmation.config
    if selected.tracking.database.resolve() != control.tracking.database.resolve():
        raise typer.BadParameter("active selection belongs to a different experiment store")
    if selected.tracking.selection_dir.resolve() != control.tracking.selection_dir.resolve():
        raise typer.BadParameter("active selection declares a different selection directory")
    if selected.deployment.champion_dir.resolve() != control.deployment.champion_dir.resolve():
        raise typer.BadParameter("active selection declares a different champion directory")
    return selected


@app.command("preprocess")
def preprocess_command(
    config_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    window: Annotated[str, typer.Option("--window")] = "final_holdout",
    all_windows: Annotated[bool, typer.Option("--all-windows")] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Build provenance-tracked evaluation folds without reading competition test data."""

    config = load_experiment(config_path)
    names = [item.name for item in temporal_windows(config)] if all_windows else [window]
    for name in names:
        manifest = preprocess_evaluation(
            config,
            window_name=name,
            overwrite=overwrite,
        )
        console.print(f"[green]Wrote evaluation dataset[/green] {manifest}")


@app.command("confirm")
def confirm_command(
    config_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    fold_manifests: Annotated[
        list[Path],
        typer.Option("--fold-manifest", exists=True, dir_okay=False),
    ],
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    """Evaluate one configuration across every walk-forward fold."""

    config = load_experiment(config_path)
    evidence = confirm_configuration(config, fold_manifests)
    destination = output or config.data.artifact_root / "tuning" / "confirmation.json"
    write_confirmation(evidence, destination)
    console.print(
        f"[green]Confirmed[/green] mean walk-forward logloss "
        f"{evidence.mean_logloss:.6f}; wrote {destination}"
    )


@app.command("tune")
def tune_command(
    config_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    screening_manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    confirmation_manifests: Annotated[
        list[Path],
        typer.Option("--confirm-manifest", exists=True, dir_okay=False),
    ],
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    """Run staged screening and emit typed evidence for the best confirmed config."""

    config = load_experiment(config_path)
    tuner = StagedTuner(config, screening_manifest)
    _, study = tuner.run()
    console.print(f"[green]Search completed[/green]; best screening logloss {study.best_value:.6f}")
    confirmed = tuner.confirm(study, confirmation_manifests)
    if not confirmed:
        raise RuntimeError("tuning produced no configuration eligible for confirmation")
    best = confirmed[0]
    destination = output or config.data.artifact_root / "tuning" / "confirmation.json"
    write_confirmation(best, destination)
    console.print(
        f"[green]Best walk-forward mean {best.mean_logloss:.6f}[/green]; wrote {destination}"
    )


@app.command("candidate")
def candidate_command(
    confirmation_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    holdout_manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    """Train the final holdout candidate and retain evidence, never its weights."""

    confirmation = load_confirmation(confirmation_path)
    config = confirmation.config
    manifest = load_manifest(holdout_manifest, verify_shards=True)
    if manifest.purpose is not DatasetPurpose.EVALUATION or manifest.name != "final_holdout":
        raise typer.BadParameter("candidate requires the final_holdout evaluation manifest")
    if manifest.validation_range is None or manifest.validation_population_sha256 is None:
        raise typer.BadParameter("final holdout has incomplete validation provenance")
    result = CandidateTrainer(config, holdout_manifest).fit(kind="candidate")
    holdout = HoldoutEvidence(
        run_id=result.run_id,
        manifest_sha256=result.manifest_sha256,
        labelled_source_sha256=manifest.labelled_source.sha256,
        training_range=manifest.training_range,
        validation_range=manifest.validation_range,
        population_sha256=manifest.validation_population_sha256,
        rows=manifest.validation_rows,
        best_epoch=result.best_epoch,
        metrics=result.validation.metrics,
    )
    destination = output or (config.data.artifact_root / "selection-candidates" / result.run_id)
    evidence_path = write_selection(
        confirmation,
        holdout,
        result.validation.row_losses,
        destination,
    )
    console.print(
        f"[green]Candidate {result.run_id} completed[/green] "
        f"(best epoch {result.best_epoch}, "
        f"logloss {result.validation.metrics['logloss']:.6f}); "
        f"retained evidence at {evidence_path}"
    )


@app.command("promote")
def promote_command(
    config_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    candidate: Annotated[Path, typer.Argument(exists=True)],
) -> None:
    """Apply the paired gate and activate configuration evidence."""

    config = load_experiment(config_path)
    loaded = load_selection(candidate)
    candidate_config = loaded.evidence.confirmation.config
    if candidate_config.tracking.database.resolve() != config.tracking.database.resolve():
        raise typer.BadParameter("candidate evidence belongs to a different experiment store")
    decision = activate_selection(
        candidate,
        config.tracking.selection_dir,
        config.promotion,
        seed=config.training.seed,
        store=RunStore(config.tracking.database),
    )
    if decision.selected:
        console.print(f"[green]Selected {loaded.evidence.selection_id}[/green]: {decision.reason}")
    else:
        console.print(
            f"[yellow]Rejected {loaded.evidence.selection_id}[/yellow]: {decision.reason}"
        )


@app.command("prepare-production")
def prepare_production_command(
    config_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Fit production features from the active selection on all labelled rows."""

    control = load_experiment(config_path)
    selected = _active_selected_config(control)
    manifest = preprocess_production(selected, overwrite=overwrite)
    console.print(f"[green]Wrote production dataset[/green] {manifest}")


@app.command("refit")
def refit_command(
    config_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    production_manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Refit the active selection and atomically replace the deployed champion."""

    control = load_experiment(config_path)
    selected = _active_selected_config(control)
    store = RunStore(control.tracking.database)
    result = ProductionRefitter(
        production_manifest,
        control.tracking.selection_dir,
        store=store,
    ).fit()
    champion = control.deployment.champion_dir
    staged = champion.parent / f".champion-{result.run_id}-{uuid.uuid4().hex}.staging"
    try:
        export_production_bundle(
            result.model,
            selected,
            result.manifest,
            production_manifest,
            staged,
            refit_run_id=result.run_id,
            selection_id=result.selection_id,
            selection_sha256=result.selection_sha256,
            epochs=result.epochs,
            steps=result.steps,
        )
        deployed = deploy_bundle(
            staged,
            champion,
            selection_path=control.tracking.selection_dir,
            store=store,
        )
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    console.print(
        f"[green]Deployed production refit {deployed.metadata['refit_run_id']}[/green] "
        f"at {champion}"
    )


@app.command("predict")
def predict_command(
    bundle: Annotated[Path, typer.Argument(exists=True)],
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")] = Path("submission.csv"),
    device: Annotated[str, typer.Option("--device")] = "cpu",
) -> None:
    written = Predictor(bundle, device=device).write_submission(manifest, output)
    console.print(f"[green]Wrote[/green] {written}")


@app.command("report")
def report_command(
    config_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")] = Path("artifacts/reports/runs"),
    manifest: Annotated[
        Path | None,
        typer.Option("--manifest", exists=True, dir_okay=False),
    ] = None,
    raw: Annotated[Path | None, typer.Option("--raw", exists=True, dir_okay=False)] = None,
) -> None:
    config = load_experiment(config_path)
    _, html_path = run_report(config.tracking.database, output)
    console.print(f"[green]Wrote[/green] {html_path}")
    if manifest is not None:
        _, manifest_html = dataset_report(manifest, output / "dataset")
        console.print(f"[green]Wrote[/green] {manifest_html}")
    if raw is not None:
        _, raw_html = raw_report(raw, output / "raw")
        console.print(f"[green]Wrote[/green] {raw_html}")


@app.command("tensorboard")
def tensorboard_command(
    config_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 6006,
) -> None:
    config = load_experiment(config_path)
    command = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(config.tracking.tensorboard_dir),
        "--port",
        str(port),
    ]
    raise typer.Exit(subprocess.call(command))
