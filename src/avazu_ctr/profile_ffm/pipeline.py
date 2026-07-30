"""Ephemeral profile FFM fitting and deterministic prediction composition."""

from __future__ import annotations

import csv
import math
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path

import polars as pl

from avazu_ctr.profile_ffm.artifacts import copy_file, publish_directory
from avazu_ctr.profile_ffm.config import ProfileFFMConfig, profile_ffm_config_sha256
from avazu_ctr.profile_ffm.contracts import (
    CompositionMetrics,
    FileArtifact,
    ProfileFFMRunManifest,
    file_artifact,
    line_file_artifact,
    load_preparation_manifest,
    sha256_file,
    write_run_manifest,
)
from avazu_ctr.profile_ffm.solver import (
    SolverJob,
    build_solver,
    run_solver_job,
)

Progress = Callable[[str], None]


def _prepared_path(
    manifest_path: Path,
    artifact: FileArtifact,
) -> Path:
    return manifest_path.parent / artifact.path


def _prediction_stats(path: Path, *, expected_rows: int) -> tuple[int, float, float, float]:
    rows = 0
    minimum = 1.0
    maximum = 0.0
    total = 0.0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = float(line)
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"invalid prediction at row {rows} in {path}")
            rows += 1
            minimum = min(minimum, value)
            maximum = max(maximum, value)
            total += value
    if rows != expected_rows:
        raise ValueError(f"{path} has {rows} predictions; expected {expected_rows}")
    return rows, minimum, maximum, total / rows


def _selector_value(row: dict[str, str], column: str) -> bool:
    value = row[column]
    if value not in {"0", "1"}:
        raise ValueError(f"selector {column!r} must contain only 0 or 1")
    return value == "1"


def _validated_prediction(value: str, *, row_id: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 < parsed < 1.0:
        raise ValueError(f"invalid prediction for {row_id}")
    return parsed


def compose_profile_ffm_predictions(
    *,
    app_selector: Path,
    site_selector: Path,
    app_profile: Path,
    app_history: Path,
    site_profile: Path,
    site_cold: Path,
    output: Path,
) -> CompositionMetrics:
    """Compose aligned prediction sources in app-then-site submission order."""

    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    prediction_total = 0.0
    prediction_minimum = 1.0
    prediction_maximum = 0.0
    source_rows = {
        "app_profile": 0,
        "app_history": 0,
        "site_profile": 0,
        "site_cold": 0,
    }
    with output.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination, lineterminator="\n")
        writer.writerow(("id", "click"))
        for (
            selector_path,
            baseline_path,
            specialist_path,
            selector_column,
            baseline_name,
            specialist_name,
        ) in (
            (
                app_selector,
                app_profile,
                app_history,
                "use_history",
                "app_profile",
                "app_history",
            ),
            (
                site_selector,
                site_profile,
                site_cold,
                "use_cold_publisher",
                "site_profile",
                "site_cold",
            ),
        ):
            with (
                selector_path.open("r", encoding="utf-8", newline="") as selectors,
                baseline_path.open("r", encoding="utf-8") as baseline,
                specialist_path.open("r", encoding="utf-8") as specialist,
            ):
                streams = zip(
                    csv.DictReader(selectors),
                    baseline,
                    specialist,
                    strict=True,
                )
                for selector, baseline_line, specialist_line in streams:
                    use_specialist = _selector_value(selector, selector_column)
                    chosen = specialist_line.strip() if use_specialist else baseline_line.strip()
                    source_rows[specialist_name if use_specialist else baseline_name] += 1
                    parsed = _validated_prediction(chosen, row_id=selector["id"])
                    prediction_total += parsed
                    prediction_minimum = min(prediction_minimum, parsed)
                    prediction_maximum = max(prediction_maximum, parsed)
                    writer.writerow((selector["id"], chosen))
                    rows += 1
    if rows == 0:
        raise ValueError("prediction composition produced no rows")
    validation = (
        pl.scan_csv(
            output,
            schema_overrides={"id": pl.String, "click": pl.Float64},
        )
        .select(
            pl.len().alias("rows"),
            pl.col("id").n_unique().alias("unique_ids"),
            pl.col("id").null_count().alias("null_ids"),
            pl.col("click").is_null().sum().alias("null_predictions"),
            pl.col("click").is_nan().sum().alias("nan_predictions"),
        )
        .collect()
        .row(0, named=True)
    )
    if (
        validation["rows"] != rows
        or validation["unique_ids"] != rows
        or validation["null_ids"]
        or validation["null_predictions"]
        or validation["nan_predictions"]
    ):
        raise ValueError("composed submission IDs or predictions are invalid")
    return CompositionMetrics(
        rows=rows,
        app_profile_rows=source_rows["app_profile"],
        app_causal_history_rows=source_rows["app_history"],
        site_profile_rows=source_rows["site_profile"],
        site_cold_publisher_rows=source_rows["site_cold"],
        prediction_minimum=prediction_minimum,
        prediction_maximum=prediction_maximum,
        prediction_mean=prediction_total / rows,
    )


def _prepared_directory_for_cleanup(config: ProfileFFMConfig, prepared: Path) -> Path:
    artifact_root = config.data.artifact_root.resolve()
    resolved = prepared.resolve()
    if resolved.parent != artifact_root or resolved.name != "prepared":
        raise RuntimeError("refusing to clean a preparation directory outside the artifact root")
    return resolved


def fit_predict_profile_ffm(
    config: ProfileFFMConfig,
    *,
    preparation_manifest: str | Path | None = None,
    output: str | Path | None = None,
    overwrite: bool = False,
    clean_prepared: bool = False,
    progress: Progress | None = None,
) -> Path:
    """Fit each prediction source sequentially and publish the composed result."""

    emit = progress or (lambda _: None)
    preparation_path = Path(
        preparation_manifest or config.data.artifact_root / "prepared" / "manifest.json"
    )
    prepared = load_preparation_manifest(
        preparation_path,
        verify_artifacts=True,
    )
    config_digest = profile_ffm_config_sha256(config)
    if prepared.config_sha256 != config_digest:
        raise ValueError("preparation manifest belongs to a different profile FFM config")

    target = config.data.artifact_root / "run"
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} already exists")
    requested_output = Path(output) if output is not None else target / "submission.csv"
    if (
        requested_output.resolve() != (target / "submission.csv").resolve()
        and requested_output.exists()
        and not overwrite
    ):
        raise FileExistsError(f"{requested_output} already exists")
    prepared_to_clean: Path | None = None
    if clean_prepared:
        prepared_to_clean = _prepared_directory_for_cleanup(config, preparation_path.parent)
        if requested_output.resolve().is_relative_to(prepared_to_clean):
            raise ValueError(
                "submission output cannot be inside a preparation selected for cleanup"
            )
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.parent / f".run-{uuid.uuid4().hex}.staging"
    stage.mkdir()
    try:
        emit("Compiling the profile FFM solver")
        build = build_solver(config, stage / "native" / "profile_ffm_solver")
        prediction_root = stage / "predictions"
        logs_root = stage / "logs"
        jobs = {
            "app_profile": SolverJob(
                name="app_profile",
                training=_prepared_path(
                    preparation_path,
                    prepared.artifacts["train_app_profile"],
                ),
                scoring=_prepared_path(
                    preparation_path,
                    prepared.artifacts["score_app_profile"],
                ),
                output=prediction_root / "app.profile.txt",
            ),
            "site_profile": SolverJob(
                name="site_profile",
                training=_prepared_path(
                    preparation_path,
                    prepared.artifacts["train_site_profile"],
                ),
                scoring=_prepared_path(
                    preparation_path,
                    prepared.artifacts["score_site_profile"],
                ),
                output=prediction_root / "site.profile.txt",
            ),
            "site_cold_publisher": SolverJob(
                name="site_cold_publisher",
                training=_prepared_path(
                    preparation_path,
                    prepared.artifacts["train_site_profile"],
                ),
                scoring=_prepared_path(
                    preparation_path,
                    prepared.artifacts["score_site_profile"],
                ),
                output=prediction_root / "site.cold-publisher.txt",
                publisher_mask_basis_points=(config.cold_publisher.training_mask_basis_points),
                score_cold_publisher=True,
            ),
            "app_causal_history": SolverJob(
                name="app_causal_history",
                training=_prepared_path(
                    preparation_path,
                    prepared.artifacts["train_app_history"],
                ),
                scoring=_prepared_path(
                    preparation_path,
                    prepared.artifacts["score_app_history"],
                ),
                output=prediction_root / "app.causal-history.txt",
            ),
        }
        expected_rows = {
            "app_profile": prepared.rows.scoring_app,
            "site_profile": prepared.rows.scoring_site,
            "site_cold_publisher": prepared.rows.scoring_site,
            "app_causal_history": prepared.rows.scoring_app,
        }
        prediction_artifacts: dict[str, FileArtifact] = {}
        fit_commands: dict[str, tuple[str, ...]] = {}
        logs: dict[str, FileArtifact] = {}
        for name, job in jobs.items():
            emit(f"Fitting {name.replace('_', ' ')}")
            stdout_path = logs_root / f"{name}.stdout.log"
            stderr_path = logs_root / f"{name}.stderr.log"
            fit_commands[name] = run_solver_job(
                build,
                job,
                config,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
            rows, _, _, _ = _prediction_stats(
                job.output,
                expected_rows=expected_rows[name],
            )
            prediction_artifacts[name] = file_artifact(
                stage,
                job.output,
                rows=rows,
            )
            logs[f"{name}_stdout"] = line_file_artifact(stage, stdout_path)
            logs[f"{name}_stderr"] = line_file_artifact(stage, stderr_path)

        emit("Composing app, site, publisher, and history predictions")
        submission_path = stage / "submission.csv"
        composition = compose_profile_ffm_predictions(
            app_selector=_prepared_path(
                preparation_path,
                prepared.artifacts["score_app_selector"],
            ),
            site_selector=_prepared_path(
                preparation_path,
                prepared.artifacts["score_site_selector"],
            ),
            app_profile=jobs["app_profile"].output,
            app_history=jobs["app_causal_history"].output,
            site_profile=jobs["site_profile"].output,
            site_cold=jobs["site_cold_publisher"].output,
            output=submission_path,
        )
        if (
            composition.app_causal_history_rows != prepared.rows.scoring_nonempty_history
            or composition.site_cold_publisher_rows != prepared.rows.scoring_cold_site
        ):
            raise ValueError("composition selectors changed from preparation evidence")
        manifest = ProfileFFMRunManifest(
            name=config.name,
            config=config,
            config_sha256=config_digest,
            preparation_manifest_sha256=sha256_file(preparation_path),
            rows=prepared.rows,
            solver=build.evidence,
            predictions=prediction_artifacts,
            composition=composition,
            submission=file_artifact(
                stage,
                submission_path,
                rows=composition.rows,
            ),
            fit_commands=fit_commands,
            logs=logs,
        )
        write_run_manifest(manifest, stage / "manifest.json")
        publish_directory(stage, target, overwrite=overwrite)
    finally:
        if stage.exists():
            shutil.rmtree(stage)

    published = target / "submission.csv"
    written = copy_file(published, requested_output, overwrite=overwrite)
    if prepared_to_clean is not None:
        shutil.rmtree(prepared_to_clean)
    return written
