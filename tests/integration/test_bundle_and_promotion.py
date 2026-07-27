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
from avazu_ctr.config.loader import write_resolved_config
from avazu_ctr.config.schema import ExperimentConfig
from avazu_ctr.data.dataset import ParquetBatchDataset
from avazu_ctr.data.manifest import ShardManifest, load_manifest, sha256_file
from avazu_ctr.inference import Predictor, export_bundle, load_bundle
from avazu_ctr.models import create_model
from avazu_ctr.tracking import RunStore
from avazu_ctr.tracking.promotion import (
    PromotionDecision,
    decide_promotion,
    promote_bundle,
)


def test_safetensors_bundle_round_trip_and_submission_order(
    processed_project: tuple[ExperimentConfig, Path],
    tmp_path: Path,
) -> None:
    config, manifest_path = processed_project
    manifest = load_manifest(manifest_path)
    model = create_model(config.model, manifest, seed=config.training.seed)
    batch = next(
        iter(
            DataLoader(
                ParquetBatchDataset(manifest_path, "validation", 16),
                batch_size=None,
            )
        )
    )
    expected = model(batch).aggregate_logits.detach()
    bundle_path = export_bundle(
        model,
        config,
        manifest,
        manifest_path,
        tmp_path / "bundle",
        run_id="round-trip",
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


def test_corrupted_weights_are_rejected(
    processed_project: tuple[ExperimentConfig, Path],
    tmp_path: Path,
) -> None:
    config, manifest_path = processed_project
    manifest = load_manifest(manifest_path)
    bundle = export_bundle(
        create_model(config.model, manifest, seed=42),
        config,
        manifest,
        manifest_path,
        tmp_path / "bundle",
        run_id="corrupt",
    )
    weights = bundle.parent / "model.safetensors"
    with weights.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(bundle)


def test_corrupted_preprocessor_metadata_is_rejected(
    processed_project: tuple[ExperimentConfig, Path],
    tmp_path: Path,
) -> None:
    config, manifest_path = processed_project
    manifest = load_manifest(manifest_path)
    bundle = export_bundle(
        create_model(config.model, manifest, seed=42),
        config,
        manifest,
        manifest_path,
        tmp_path / "bundle",
        run_id="corrupt-preprocessor",
    )
    preprocessor = bundle.parent / "preprocessor" / "preprocessor.json"
    preprocessor.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="preprocessor metadata"):
        load_bundle(bundle)


def test_prediction_rejects_different_fitted_feature_state(
    processed_project: tuple[ExperimentConfig, Path],
    tmp_path: Path,
) -> None:
    config, manifest_path = processed_project
    manifest = load_manifest(manifest_path)
    bundle = export_bundle(
        create_model(config.model, manifest, seed=42),
        config,
        manifest,
        manifest_path,
        tmp_path / "bundle",
        run_id="contract",
    )

    def absolute_shards(shards: Sequence[ShardManifest]) -> tuple[ShardManifest, ...]:
        return tuple(
            shard.model_copy(
                update={"path": (manifest_path.parent / shard.path).resolve().as_posix()}
            )
            for shard in shards
        )

    first_table = manifest.fitted_tables[0].model_copy(update={"sha256": "0" * 64})
    mismatch = manifest.model_copy(
        update={
            "train_shards": absolute_shards(manifest.train_shards),
            "validation_shards": absolute_shards(manifest.validation_shards),
            "test_shards": absolute_shards(manifest.test_shards),
            "fitted_tables": (first_table, *manifest.fitted_tables[1:]),
        }
    )
    mismatch_path = tmp_path / "mismatch.json"
    mismatch_path.write_text(mismatch.model_dump_json(indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="feature contract"):
        Predictor(bundle).validate_manifest_contract(mismatch_path)


def test_paired_promotion_rejects_noise_and_accepts_real_improvement() -> None:
    config = load_experiment("configs/champion.yaml")
    promotion = config.promotion.model_copy(
        update={"bootstrap_samples": 500, "bootstrap_block_rows": 51}
    )
    incumbent = np.linspace(0.2, 0.8, 1000)
    candidate = incumbent - 0.01
    decision = decide_promotion(
        candidate,
        incumbent,
        [0.37, 0.38, 0.39],
        [0.38, 0.39, 0.40],
        promotion,
        seed=42,
    )
    assert decision.promoted
    assert decision.bootstrap_blocks == 20
    assert decision.bootstrap_samples == 500
    noisy = decide_promotion(
        incumbent + np.tile([-0.01, 0.01], 500),
        incumbent,
        [0.38, 0.39, 0.40],
        [0.38, 0.39, 0.40],
        promotion,
        seed=42,
    )
    assert not noisy.promoted


def test_atomic_first_promotion_moves_candidate(
    processed_project: tuple[ExperimentConfig, Path],
    tmp_path: Path,
) -> None:
    config, manifest_path = processed_project
    manifest = load_manifest(manifest_path)
    candidate = tmp_path / "candidate"
    export_bundle(
        create_model(config.model, manifest, seed=42),
        config,
        manifest,
        manifest_path,
        candidate,
        run_id="candidate",
    )
    store = RunStore(tmp_path / "runs.sqlite3")
    run_id = store.start_run(config, manifest, run_id="candidate")
    store.finish_run(run_id, status="completed")
    decision = PromotionDecision(True, "first", -1.0, -1.0, 0.3, 1.0)
    champion = tmp_path / "champion"
    assert promote_bundle(
        candidate,
        champion,
        decision,
        store=store,
        candidate_run_id=run_id,
        incumbent_run_id=None,
    )
    assert not candidate.exists()
    assert (champion / "model.safetensors").exists()


def test_promote_command_applies_paired_gate_and_replaces_champion(
    processed_project: tuple[ExperimentConfig, Path],
    tmp_path: Path,
) -> None:
    base, manifest_path = processed_project
    manifest = load_manifest(manifest_path)
    tracking = base.tracking.model_copy(
        update={
            "database": tmp_path / "experiments.sqlite3",
            "tensorboard_dir": tmp_path / "tensorboard",
            "champion_dir": tmp_path / "champion",
            "tensorboard": False,
        }
    )
    promotion = base.promotion.model_copy(update={"bootstrap_samples": 100})
    config = base.model_copy(update={"tracking": tracking, "promotion": promotion})
    config_path = tmp_path / "config.yaml"
    write_resolved_config(config, config_path)
    store = RunStore(tracking.database)

    incumbent_run = store.start_run(config, manifest, run_id="incumbent")
    store.finish_run(incumbent_run, status="completed")
    export_bundle(
        create_model(config.model, manifest, seed=1),
        config,
        manifest,
        manifest_path,
        tracking.champion_dir,
        run_id=incumbent_run,
    )
    candidate_run = store.start_run(config, manifest, run_id="candidate")
    store.finish_run(candidate_run, status="completed")
    candidate = tmp_path / "candidate"
    candidate_bundle = export_bundle(
        create_model(config.model, manifest, seed=2),
        config,
        manifest,
        manifest_path,
        candidate,
        run_id=candidate_run,
    )
    store.record_artifact(
        candidate_run,
        kind="candidate_bundle",
        path=candidate_bundle,
        sha256=sha256_file(candidate_bundle),
    )

    rows = np.asarray([f"row-{index}" for index in range(100)])
    labels = np.zeros(100, dtype=np.float32)
    incumbent_evaluation = tmp_path / "incumbent.npz"
    candidate_evaluation = tmp_path / "candidate.npz"
    np.savez_compressed(
        incumbent_evaluation,
        schema_version=np.asarray(2, dtype=np.int64),
        run_id=np.asarray(incumbent_run),
        row_ids=rows,
        labels=labels,
        row_losses=np.full(100, 0.5),
    )
    np.savez_compressed(
        candidate_evaluation,
        schema_version=np.asarray(2, dtype=np.int64),
        run_id=np.asarray(candidate_run),
        row_ids=rows,
        labels=labels,
        row_losses=np.full(100, 0.4),
    )
    folds = [
        "--candidate-fold-loss",
        "0.30",
        "--candidate-fold-loss",
        "0.31",
        "--candidate-fold-loss",
        "0.32",
        "--incumbent-fold-loss",
        "0.40",
        "--incumbent-fold-loss",
        "0.41",
        "--incumbent-fold-loss",
        "0.42",
    ]
    result = CliRunner().invoke(
        app,
        [
            "promote",
            str(config_path),
            str(candidate),
            str(candidate_evaluation),
            str(incumbent_evaluation),
            *folds,
        ],
    )

    assert result.exit_code == 0, result.output
    assert not candidate.exists()
    assert load_bundle(tracking.champion_dir).metadata["run_id"] == candidate_run
    with store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM promotions").fetchone()[0] == 1
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM artifacts WHERE kind = 'candidate_bundle'"
            ).fetchone()[0]
            == 0
        )
