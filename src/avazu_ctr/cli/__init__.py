"""Installed command-line interface."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import typer
from rich.console import Console
from torch.utils.data import DataLoader

from avazu_ctr.config import load_experiment
from avazu_ctr.data import preprocess, temporal_windows
from avazu_ctr.data.dataset import ParquetBatchDataset
from avazu_ctr.data.manifest import sha256_file
from avazu_ctr.exploration import dataset_report, raw_report, run_report
from avazu_ctr.inference import Predictor, export_bundle, load_bundle
from avazu_ctr.tracking import RunStore
from avazu_ctr.tracking.promotion import (
    PromotionDecision,
    decide_promotion,
    promote_bundle,
)
from avazu_ctr.training import Trainer
from avazu_ctr.training.evaluation import evaluate
from avazu_ctr.tuning import StagedTuner

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
console = Console()


@app.command("preprocess")
def preprocess_command(
    config_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    window: Annotated[str, typer.Option("--window")] = "final_holdout",
    all_windows: Annotated[bool, typer.Option("--all-windows")] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    config = load_experiment(config_path)
    names = [item.name for item in temporal_windows(config)] if all_windows else [window]
    for name in names:
        manifest = preprocess(
            config,
            window_name=name,
            include_test=name == "final_holdout",
            overwrite=overwrite,
        )
        console.print(f"[green]Wrote[/green] {manifest}")


@app.command("train")
def train_command(
    config_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    manifest_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    resume_from: Annotated[
        Path | None, typer.Option("--resume", exists=True, dir_okay=False)
    ] = None,
    export_candidate: Annotated[bool, typer.Option("--export-candidate")] = False,
) -> None:
    config = load_experiment(config_path)
    store = RunStore(config.tracking.database)
    result = Trainer(config, manifest_path, store=store).fit(resume_from=resume_from)
    console.print(
        f"[green]Run {result.run_id} completed[/green] "
        f"(best epoch {result.best_epoch}, "
        f"logloss {result.validation.metrics['logloss']:.6f})"
    )
    first_champion = not config.tracking.champion_dir.exists()
    if first_champion or export_candidate:
        candidate = config.data.artifact_root / "candidates" / result.run_id
        bundle = export_bundle(
            result.model,
            config,
            result.manifest,
            manifest_path,
            candidate,
            run_id=result.run_id,
            validation_metrics=result.validation.metrics,
        )
        if not first_champion:
            store.record_artifact(
                result.run_id,
                kind="candidate_bundle",
                path=bundle,
                sha256=sha256_file(bundle),
            )
            console.print(
                f"[green]Exported candidate[/green] at {candidate}; "
                "evaluate it and run the promote command to apply the statistical gate"
            )
            return
        initial = PromotionDecision(
            promoted=True,
            reason="first valid champion",
            mean_difference=float("-inf"),
            upper_confidence_bound=float("-inf"),
            candidate_fold_mean=result.validation.metrics["logloss"],
            incumbent_fold_mean=float("inf"),
        )
        promote_bundle(
            candidate,
            config.tracking.champion_dir,
            initial,
            store=store,
            candidate_run_id=result.run_id,
            incumbent_run_id=None,
        )
        champion_bundle = config.tracking.champion_dir / bundle.name
        store.record_artifact(
            result.run_id,
            kind="champion_bundle",
            path=champion_bundle,
            sha256=sha256_file(champion_bundle),
        )
        console.print(f"[green]Promoted first champion[/green] at {config.tracking.champion_dir}")


def _evaluation_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"schema_version", "run_id", "row_ids", "labels", "row_losses"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"{path} is missing evaluation fields: {sorted(missing)}")
        if int(payload["schema_version"].item()) != 2:
            raise ValueError(f"{path} has an unsupported evaluation schema")
        arrays = {name: payload[name].copy() for name in required}
    row_ids = arrays["row_ids"]
    labels = arrays["labels"]
    row_losses = arrays["row_losses"]
    if row_ids.ndim != 1 or labels.ndim != 1 or row_losses.ndim != 1:
        raise ValueError(f"{path} evaluation rows must be one-dimensional")
    if not row_ids.size or not (row_ids.size == labels.size == row_losses.size):
        raise ValueError(f"{path} evaluation arrays have inconsistent lengths")
    if not np.isfinite(labels).all() or not np.isfinite(row_losses).all():
        raise ValueError(f"{path} evaluation values must be finite")
    return arrays


@app.command("promote")
def promote_command(
    config_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    candidate: Annotated[Path, typer.Argument(exists=True)],
    candidate_evaluation: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    incumbent_evaluation: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    candidate_fold_loss: Annotated[
        list[float] | None, typer.Option("--candidate-fold-loss")
    ] = None,
    incumbent_fold_loss: Annotated[
        list[float] | None, typer.Option("--incumbent-fold-loss")
    ] = None,
) -> None:
    config = load_experiment(config_path)
    candidate_dir = candidate.parent if candidate.is_file() else candidate
    candidate_bundle = load_bundle(candidate_dir)
    incumbent_bundle = load_bundle(config.tracking.champion_dir)
    candidate_arrays = _evaluation_arrays(candidate_evaluation)
    incumbent_arrays = _evaluation_arrays(incumbent_evaluation)
    candidate_run_id = str(candidate_arrays["run_id"].item())
    incumbent_run_id = str(incumbent_arrays["run_id"].item())
    if candidate_run_id != str(candidate_bundle.metadata["run_id"]):
        raise typer.BadParameter("candidate evaluation belongs to a different run")
    if incumbent_run_id != str(incumbent_bundle.metadata["run_id"]):
        raise typer.BadParameter("incumbent evaluation belongs to a different run")
    if not np.array_equal(candidate_arrays["row_ids"], incumbent_arrays["row_ids"]):
        raise typer.BadParameter("candidate and incumbent evaluations contain different rows")
    if not np.array_equal(candidate_arrays["labels"], incumbent_arrays["labels"]):
        raise typer.BadParameter("candidate and incumbent evaluations contain different labels")
    candidate_folds = candidate_fold_loss or []
    incumbent_folds = incumbent_fold_loss or []
    expected_folds = config.data.split.walk_forward_folds
    if len(candidate_folds) != expected_folds or len(incumbent_folds) != expected_folds:
        raise typer.BadParameter(
            f"provide exactly {expected_folds} candidate and incumbent fold losses"
        )
    decision = decide_promotion(
        candidate_arrays["row_losses"],
        incumbent_arrays["row_losses"],
        candidate_folds,
        incumbent_folds,
        config.promotion,
        seed=config.training.seed,
    )
    store = RunStore(config.tracking.database)
    promoted = promote_bundle(
        candidate_dir,
        config.tracking.champion_dir,
        decision,
        store=store,
        candidate_run_id=candidate_run_id,
        incumbent_run_id=incumbent_run_id,
    )
    store.delete_artifacts(candidate_run_id, kind="candidate_bundle")
    if promoted:
        store.delete_artifacts(incumbent_run_id, kind="champion_bundle")
        champion_bundle_path = config.tracking.champion_dir / "bundle.json"
        store.record_artifact(
            candidate_run_id,
            kind="champion_bundle",
            path=champion_bundle_path,
            sha256=sha256_file(champion_bundle_path),
        )
        console.print(
            f"[green]Promoted {candidate_run_id}[/green]: {decision.reason} "
            f"(paired mean {decision.mean_difference:.8f}, "
            f"upper bound {decision.upper_confidence_bound:.8f})"
        )
    else:
        console.print(f"[yellow]Rejected {candidate_run_id}[/yellow]: {decision.reason}")


@app.command("tune")
def tune_command(
    config_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    screening_manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    confirmation_manifests: Annotated[list[Path] | None, typer.Option("--confirm-manifest")] = None,
) -> None:
    config = load_experiment(config_path)
    tuner = StagedTuner(config, screening_manifest)
    best, study = tuner.run()
    console.print(f"[green]Search completed[/green]; best screening logloss {study.best_value:.6f}")
    if confirmation_manifests:
        confirmed = tuner.confirm(study, confirmation_manifests)
        if confirmed:
            best = confirmed[0].config
            console.print(f"Best walk-forward mean: {confirmed[0].mean_logloss:.6f}")
    output = config.data.artifact_root / "tuning" / "best-config.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(best.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"Wrote {output}")


@app.command("evaluate")
def evaluate_command(
    bundle: Annotated[Path, typer.Argument(exists=True)],
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")] = Path("artifacts/reports/evaluation"),
    device: Annotated[str, typer.Option("--device")] = "cpu",
) -> None:
    predictor = Predictor(bundle, device=device)
    predictor.validate_manifest_contract(manifest)
    loader = DataLoader(
        ParquetBatchDataset(
            manifest,
            "validation",
            predictor.bundle.config.training.batch_size,
            shuffle=False,
        ),
        batch_size=None,
    )
    amp_dtype = (
        torch.float16 if predictor.bundle.config.training.amp_dtype == "float16" else torch.bfloat16
    )
    result = evaluate(
        predictor.model,
        loader,
        torch.device(device),
        amp=predictor.bundle.config.training.amp,
        amp_dtype=amp_dtype,
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(result.metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    np.savez_compressed(
        output / "row_losses.npz",
        schema_version=np.asarray(2, dtype=np.int64),
        run_id=np.asarray(str(predictor.bundle.metadata["run_id"])),
        row_ids=np.asarray(result.row_ids),
        labels=result.labels,
        probabilities=result.probabilities,
        row_losses=result.row_losses,
    )
    _, report_html = dataset_report(manifest, output / "dataset")
    console.print(
        f"[green]Logloss {result.metrics['logloss']:.6f}[/green]; "
        f"wrote {output / 'metrics.json'} and {report_html}"
    )


@app.command("predict")
def predict_command(
    bundle: Annotated[Path, typer.Argument(exists=True)],
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")] = Path("submission.csv"),
    device: Annotated[str, typer.Option("--device")] = "cpu",
    compile_model: Annotated[bool, typer.Option("--compile")] = False,
) -> None:
    written = Predictor(bundle, device=device, compile_model=compile_model).write_submission(
        manifest, output
    )
    console.print(f"[green]Wrote[/green] {written}")


@app.command("report")
def report_command(
    config_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")] = Path("artifacts/reports/runs"),
    manifest: Annotated[
        Path | None, typer.Option("--manifest", exists=True, dir_okay=False)
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
