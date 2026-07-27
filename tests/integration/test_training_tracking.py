from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from avazu_ctr.config.schema import ExperimentConfig
from avazu_ctr.tracking import RunStore
from avazu_ctr.training import Trainer


def test_training_records_sqlite_and_tensorboard(
    processed_project: tuple[ExperimentConfig, Path],
) -> None:
    config, manifest_path = processed_project
    store = RunStore(config.tracking.database)
    result = Trainer(config, manifest_path, store=store).fit()
    run = store.run(result.run_id)
    assert run["status"] == "completed"
    assert store.latest_metrics(result.run_id, "validation")["logloss"] > 0
    events = list((config.tracking.tensorboard_dir / result.run_id).glob("events.out.tfevents.*"))
    assert events
    with sqlite3.connect(config.tracking.database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM metrics WHERE run_id = ?", (result.run_id,)
            ).fetchone()[0]
            > 0
        )


def test_successful_run_removes_resume_state(
    processed_project: tuple[ExperimentConfig, Path],
) -> None:
    config, manifest_path = processed_project
    training = config.training.model_copy(update={"epochs": 2, "resume_checkpoint": True})
    config = config.model_copy(update={"training": training, "name": "resume-cleanup"})
    result = Trainer(config, manifest_path).fit()
    resume = config.data.artifact_root / "runs" / result.run_id / "resume.pt"
    assert not resume.exists()


def test_unversioned_run_store_is_rejected(tmp_path: Path) -> None:
    database = tmp_path / "unversioned.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE old_runs (id INTEGER)")
    with pytest.raises(ValueError, match="unsupported"):
        RunStore(database)
