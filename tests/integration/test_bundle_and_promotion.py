from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader
from typer.testing import CliRunner

from avazu_ctr.cli import app
from avazu_ctr.config import load_experiment
from avazu_ctr.config.loader import resolved_config, write_resolved_config
from avazu_ctr.config.schema import ExperimentConfig
from avazu_ctr.data.dataset import ParquetBatchDataset
from avazu_ctr.data.manifest import (
    DatasetManifest,
    FittedTableManifest,
    HourRange,
    ShardManifest,
    load_manifest,
    sha256_file,
    sha256_json,
)
from avazu_ctr.inference import Predictor, export_production_bundle, load_bundle
from avazu_ctr.models import create_model
from avazu_ctr.tracking import (
    ConfirmationEvidence,
    FoldEvidence,
    HoldoutEvidence,
    RunStore,
    deploy_bundle,
    load_selection,
    write_selection,
)
from avazu_ctr.tracking.promotion import activate_selection, decide_selection
from avazu_ctr.training import ProductionRefitter


def _record_run(
    store: RunStore,
    config: ExperimentConfig,
    manifest: DatasetManifest,
    run_id: str,
    *,
    kind: str = "candidate",
    manifest_sha256: str,
    validation_logloss: float,
) -> None:
    store.start_run(
        config,
        manifest,
        kind=kind,
        plan={
            "mode": "evaluation",
            "epochs": 1,
            "steps_per_epoch": 1,
            "planned_steps": 1,
            "early_stopping": True,
            "manifest_sha256": manifest_sha256,
        },
        run_id=run_id,
    )
    store.log_metrics(
        run_id,
        step=0,
        split="validation",
        metrics={"logloss": validation_logloss},
    )
    store.finish_run(
        run_id,
        status="completed",
        summary={
            "best_epoch": 0,
            "epochs_completed": 1,
            "steps_completed": 1,
            "validation": {"logloss": validation_logloss},
        },
    )


def _write_candidate_selection(
    root: Path,
    config: ExperimentConfig,
    manifest_path: Path,
    store: RunStore,
    *,
    run_id: str,
    losses: np.ndarray,
    fold_losses: tuple[float, ...] = (0.38, 0.39, 0.40),
) -> Path:
    manifest = load_manifest(manifest_path)
    assert manifest.validation_range is not None
    assert manifest.validation_population_sha256 is not None
    holdout_logloss = float(losses.mean())
    _record_run(
        store,
        config,
        manifest,
        run_id,
        manifest_sha256=sha256_file(manifest_path),
        validation_logloss=holdout_logloss,
    )
    fold_hours = config.data.split.fold_hours
    fold_end = manifest.validation_range.start
    folds: list[FoldEvidence] = []
    for index, loss in enumerate(fold_losses):
        validation_end = fold_end - (len(fold_losses) - index - 1) * fold_hours
        validation_range = HourRange(
            start=validation_end - fold_hours,
            end=validation_end,
        )
        fold_manifest_sha256 = sha256_json({"manifest": index})
        training_population_sha256 = sha256_json({"training_population": index})
        validation_population_sha256 = sha256_json({"validation_population": index})
        fold_run_id = f"{run_id}-fold-{index}"
        fold_manifest = manifest.model_copy(
            update={
                "name": f"walk_forward_{index}",
                "training_range": HourRange(
                    start=manifest.training_range.start,
                    end=validation_range.start,
                ),
                "validation_range": validation_range,
                "training_population_sha256": training_population_sha256,
                "validation_population_sha256": validation_population_sha256,
            }
        )
        _record_run(
            store,
            config,
            fold_manifest,
            fold_run_id,
            kind="confirmation",
            manifest_sha256=fold_manifest_sha256,
            validation_logloss=loss,
        )
        folds.append(
            FoldEvidence(
                window=f"walk_forward_{index}",
                run_id=fold_run_id,
                manifest_sha256=fold_manifest_sha256,
                labelled_source_sha256=manifest.labelled_source.sha256,
                training_range=HourRange(
                    start=manifest.training_range.start,
                    end=validation_range.start,
                ),
                validation_range=validation_range,
                population_sha256=validation_population_sha256,
                rows=fold_manifest.validation_rows,
                logloss=loss,
            )
        )
    confirmation = ConfirmationEvidence(
        config=config,
        config_sha256=sha256_json(resolved_config(config)),
        folds=tuple(folds),
    )
    holdout = HoldoutEvidence(
        run_id=run_id,
        manifest_sha256=sha256_file(manifest_path),
        labelled_source_sha256=manifest.labelled_source.sha256,
        training_range=manifest.training_range,
        validation_range=manifest.validation_range,
        population_sha256=manifest.validation_population_sha256,
        rows=manifest.validation_rows,
        best_epoch=0,
        metrics={"logloss": holdout_logloss},
    )
    write_selection(confirmation, holdout, losses, root)
    return root


def test_production_bundle_round_trip_and_submission_order(
    production_project: tuple[ExperimentConfig, Path],
    tmp_path: Path,
) -> None:
    config, manifest_path = production_project
    manifest = load_manifest(manifest_path)
    model = create_model(config.model, manifest, seed=config.training.seed)
    batch = next(
        iter(
            DataLoader(
                ParquetBatchDataset(manifest_path, "test", 16),
                batch_size=None,
            )
        )
    )
    expected = model(batch).aggregate_logits.detach()
    bundle_path = export_production_bundle(
        model,
        config,
        manifest,
        manifest_path,
        tmp_path / "bundle",
        refit_run_id="round-trip",
        selection_id="selection",
        selection_sha256="0" * 64,
        epochs=1,
        steps=1,
    )
    loaded = load_bundle(bundle_path)
    actual = loaded.model(batch).aggregate_logits.detach()
    assert torch.equal(expected, actual)

    submission = Predictor(bundle_path).write_submission(manifest_path, tmp_path / "submission.csv")
    ids = [line.split(",", 1)[0] for line in submission.read_text().splitlines()[1:]]
    assert ids == [
        row_id
        for shard in manifest.test_shards
        for row_id in pl.read_parquet(manifest_path.parent / shard.path)["id"].to_list()
    ]


def test_evaluation_weights_cannot_be_exported(
    processed_project: tuple[ExperimentConfig, Path],
    tmp_path: Path,
) -> None:
    config, manifest_path = processed_project
    manifest = load_manifest(manifest_path)
    with pytest.raises(ValueError, match="production refit"):
        export_production_bundle(
            create_model(config.model, manifest, seed=42),
            config,
            manifest,
            manifest_path,
            tmp_path / "bundle",
            refit_run_id="evaluation",
            selection_id="selection",
            selection_sha256="0" * 64,
            epochs=1,
            steps=1,
        )


def test_corrupted_bundle_artifacts_are_rejected(
    production_project: tuple[ExperimentConfig, Path],
    tmp_path: Path,
) -> None:
    config, manifest_path = production_project
    manifest = load_manifest(manifest_path)
    bundle = export_production_bundle(
        create_model(config.model, manifest, seed=42),
        config,
        manifest,
        manifest_path,
        tmp_path / "weights-bundle",
        refit_run_id="corrupt",
        selection_id="selection",
        selection_sha256="0" * 64,
        epochs=1,
        steps=1,
    )
    weights = bundle.parent / "model.safetensors"
    with weights.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(bundle)

    metadata_bundle = export_production_bundle(
        create_model(config.model, manifest, seed=42),
        config,
        manifest,
        manifest_path,
        tmp_path / "preprocessor-bundle",
        refit_run_id="corrupt-preprocessor",
        selection_id="selection",
        selection_sha256="0" * 64,
        epochs=1,
        steps=1,
    )
    preprocessor = metadata_bundle.parent / "preprocessor" / "preprocessor.json"
    preprocessor.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="preprocessor metadata"):
        load_bundle(metadata_bundle)


def test_prediction_requires_the_exact_production_manifest(
    production_project: tuple[ExperimentConfig, Path],
    tmp_path: Path,
) -> None:
    config, manifest_path = production_project
    manifest = load_manifest(manifest_path)
    bundle = export_production_bundle(
        create_model(config.model, manifest, seed=42),
        config,
        manifest,
        manifest_path,
        tmp_path / "bundle",
        refit_run_id="contract",
        selection_id="selection",
        selection_sha256="0" * 64,
        epochs=1,
        steps=1,
    )

    def absolute_shards(shards: Sequence[ShardManifest]) -> tuple[ShardManifest, ...]:
        return tuple(
            shard.model_copy(
                update={"path": (manifest_path.parent / shard.path).resolve().as_posix()}
            )
            for shard in shards
        )

    def absolute_tables(
        tables: Sequence[FittedTableManifest],
    ) -> tuple[FittedTableManifest, ...]:
        return tuple(
            table.model_copy(
                update={"path": (manifest_path.parent / table.path).resolve().as_posix()}
            )
            for table in tables
        )

    mismatch = manifest.model_copy(
        update={
            "train_shards": absolute_shards(manifest.train_shards),
            "test_shards": absolute_shards(manifest.test_shards),
            "fitted_tables": absolute_tables(manifest.fitted_tables),
            "package_lock_sha256": "0" * 64,
        }
    )
    mismatch_path = tmp_path / "mismatch.json"
    mismatch_path.write_text(mismatch.model_dump_json(indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="deployed production manifest"):
        Predictor(bundle).validate_manifest_contract(mismatch_path)


def test_paired_selection_rejects_noise_and_accepts_real_improvement() -> None:
    config = load_experiment("configs/champion.yaml")
    promotion = config.promotion.model_copy(
        update={"bootstrap_samples": 500, "bootstrap_block_rows": 51}
    )
    incumbent = np.linspace(0.2, 0.8, 1000)
    candidate = incumbent - 0.01
    decision = decide_selection(
        candidate,
        incumbent,
        [0.37, 0.38, 0.39],
        [0.38, 0.39, 0.40],
        promotion,
        seed=42,
    )
    assert decision.selected
    assert decision.bootstrap_blocks == 20
    assert decision.bootstrap_samples == 500
    noisy = decide_selection(
        incumbent + np.tile([-0.01, 0.01], 500),
        incumbent,
        [0.38, 0.39, 0.40],
        [0.38, 0.39, 0.40],
        promotion,
        seed=42,
    )
    assert not noisy.selected


def test_selection_activation_retains_evidence_not_validation_weights(
    processed_project: tuple[ExperimentConfig, Path],
    tmp_path: Path,
) -> None:
    config, manifest_path = processed_project
    store = RunStore(tmp_path / "runs.sqlite3")
    rows = load_manifest(manifest_path).validation_rows
    candidate = _write_candidate_selection(
        tmp_path / "candidate",
        config,
        manifest_path,
        store,
        run_id="candidate",
        losses=np.full(rows, 0.4),
    )
    active = tmp_path / "selection"
    decision = activate_selection(
        candidate,
        active,
        config.promotion.model_copy(update={"bootstrap_samples": 100}),
        seed=42,
        store=store,
    )
    assert decision.selected
    assert not candidate.exists()
    assert load_selection(active).evidence.selection_id == "candidate"
    assert not list(active.rglob("*.safetensors"))

    rejected = _write_candidate_selection(
        tmp_path / "rejected",
        config,
        manifest_path,
        store,
        run_id="rejected",
        losses=np.full(rows, 0.5),
    )
    decision = activate_selection(
        rejected,
        active,
        config.promotion.model_copy(update={"bootstrap_samples": 100}),
        seed=42,
        store=store,
    )
    assert not decision.selected
    assert not rejected.exists()
    assert load_selection(active).evidence.selection_id == "candidate"
    with store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM selection_decisions").fetchone()[0] == 2


def test_fixed_budget_refit_and_atomic_deployment(
    processed_project: tuple[ExperimentConfig, Path],
    production_project: tuple[ExperimentConfig, Path],
    tmp_path: Path,
) -> None:
    config, evaluation_manifest = processed_project
    _, production_manifest = production_project
    store = RunStore(tmp_path / "runs.sqlite3")
    rows = load_manifest(evaluation_manifest).validation_rows
    candidate = _write_candidate_selection(
        tmp_path / "candidate",
        config,
        evaluation_manifest,
        store,
        run_id="selected",
        losses=np.full(rows, 0.4),
    )
    selection = tmp_path / "selection"
    activate_selection(
        candidate,
        selection,
        config.promotion.model_copy(update={"bootstrap_samples": 100}),
        seed=42,
        store=store,
    )

    result = ProductionRefitter(
        production_manifest,
        selection,
        store=store,
    ).fit()
    staged = tmp_path / "staged"
    export_production_bundle(
        result.model,
        config,
        result.manifest,
        production_manifest,
        staged,
        refit_run_id=result.run_id,
        selection_id=result.selection_id,
        selection_sha256=result.selection_sha256,
        epochs=result.epochs,
        steps=result.steps,
    )
    champion = tmp_path / "champion"
    deployed = deploy_bundle(
        staged,
        champion,
        selection_path=selection,
        store=store,
    )

    assert result.epochs == 1
    assert result.steps > 0
    assert deployed.metadata["selection_id"] == "selected"
    assert deployed.metadata["refit_plan"]["validation"] is False
    assert store.latest_metrics(result.run_id, "validation") == {}
    assert not staged.exists()
    assert (champion / "model.safetensors").is_file()
    with store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM deployments").fetchone()[0] == 1
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM artifacts WHERE kind = 'production_bundle'"
            ).fetchone()[0]
            == 1
        )

    stale = tmp_path / "stale"
    export_production_bundle(
        result.model,
        config,
        result.manifest,
        production_manifest,
        stale,
        refit_run_id=result.run_id,
        selection_id=result.selection_id,
        selection_sha256=result.selection_sha256,
        epochs=result.epochs,
        steps=result.steps,
    )
    replacement = _write_candidate_selection(
        tmp_path / "replacement",
        config,
        evaluation_manifest,
        store,
        run_id="replacement",
        losses=np.full(rows, 0.3),
    )
    replacement_decision = activate_selection(
        replacement,
        selection,
        config.promotion.model_copy(update={"bootstrap_samples": 100}),
        seed=42,
        store=store,
    )
    assert replacement_decision.selected
    with pytest.raises(ValueError, match="active selection changed"):
        deploy_bundle(
            stale,
            champion,
            selection_path=selection,
            store=store,
        )
    assert load_bundle(champion).metadata["selection_id"] == "selected"


def test_promote_cli_activates_complete_evidence(
    processed_project: tuple[ExperimentConfig, Path],
    tmp_path: Path,
) -> None:
    base, manifest_path = processed_project
    tracking = base.tracking.model_copy(
        update={
            "database": tmp_path / "experiments.sqlite3",
            "tensorboard_dir": tmp_path / "tensorboard",
            "selection_dir": tmp_path / "selection",
            "tensorboard": False,
        }
    )
    deployment = base.deployment.model_copy(update={"champion_dir": tmp_path / "champion"})
    config = base.model_copy(update={"tracking": tracking, "deployment": deployment})
    config_path = tmp_path / "config.yaml"
    write_resolved_config(config, config_path)
    store = RunStore(tracking.database)
    rows = load_manifest(manifest_path).validation_rows
    candidate = _write_candidate_selection(
        tmp_path / "candidate",
        config,
        manifest_path,
        store,
        run_id="candidate",
        losses=np.full(rows, 0.4),
    )

    result = CliRunner().invoke(
        app,
        ["promote", str(config_path), str(candidate)],
    )

    assert result.exit_code == 0, result.output
    assert load_selection(tracking.selection_dir).evidence.selection_id == "candidate"
