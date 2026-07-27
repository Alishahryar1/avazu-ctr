"""Small authoritative SQLite run store."""

from __future__ import annotations

import json
import platform
import sqlite3
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from avazu_ctr.config.loader import resolved_config
from avazu_ctr.config.schema import ExperimentConfig
from avazu_ctr.data.manifest import DatasetManifest, sha256_file, sha256_json


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def code_fingerprint() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        diff = subprocess.check_output(
            ["git", "diff", "HEAD", "--binary", "--no-ext-diff"],
            stderr=subprocess.DEVNULL,
        )
        untracked_output = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            stderr=subprocess.DEVNULL,
        )
        untracked_paths = [
            Path(value.decode("utf-8", errors="surrogateescape"))
            for value in untracked_output.split(b"\0")
            if value
        ]
        untracked = [
            {"path": path.as_posix(), "sha256": sha256_file(path)}
            for path in sorted(untracked_paths, key=lambda item: item.as_posix())
            if path.is_file()
        ]
        return {
            "commit": commit,
            "dirty": bool(diff or untracked),
            "diff_sha256": sha256_json(diff.hex()),
            "untracked_sha256": sha256_json(untracked),
        }
    except (OSError, subprocess.SubprocessError):
        return {
            "commit": None,
            "dirty": True,
            "diff_sha256": None,
            "untracked_sha256": None,
        }


def environment_snapshot() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "devices": [
            {
                "name": torch.cuda.get_device_name(index),
                "capability": torch.cuda.get_device_capability(index),
            }
            for index in range(torch.cuda.device_count())
        ],
    }


class RunStore:
    SCHEMA_VERSION = 3

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self.connect() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            tables = {
                str(row[0])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
                if not str(row[0]).startswith("sqlite_")
            }
            if version == 0 and tables:
                raise ValueError(f"{self.path} is an unsupported unversioned experiment store")
            if version not in {0, self.SCHEMA_VERSION}:
                raise ValueError(f"unsupported experiment-store schema {version}")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    parent_run_id TEXT REFERENCES runs(run_id),
                    kind TEXT NOT NULL,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    config_json TEXT NOT NULL,
                    config_sha256 TEXT NOT NULL,
                    dataset_json TEXT NOT NULL,
                    code_json TEXT NOT NULL,
                    environment_json TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    summary_json TEXT,
                    study_name TEXT,
                    trial_number INTEGER,
                    error TEXT
                );
                CREATE TABLE IF NOT EXISTS metrics (
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    step INTEGER NOT NULL,
                    split TEXT NOT NULL,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, step, split, name)
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    bytes INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, kind, path)
                );
                CREATE TABLE IF NOT EXISTS selection_decisions (
                    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_run_id TEXT NOT NULL REFERENCES runs(run_id),
                    incumbent_run_id TEXT REFERENCES runs(run_id),
                    candidate_evidence_sha256 TEXT NOT NULL,
                    incumbent_evidence_sha256 TEXT,
                    selected INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    statistics_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS deployments (
                    deployment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    selection_run_id TEXT NOT NULL REFERENCES runs(run_id),
                    refit_run_id TEXT NOT NULL REFERENCES runs(run_id),
                    replaced_refit_run_id TEXT REFERENCES runs(run_id),
                    bundle_path TEXT NOT NULL,
                    bundle_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            connection.execute(f"PRAGMA user_version = {self.SCHEMA_VERSION}")

    def start_run(
        self,
        config: ExperimentConfig,
        manifest: DatasetManifest,
        *,
        kind: str,
        plan: dict[str, Any],
        parent_run_id: str | None = None,
        run_id: str | None = None,
        study_name: str | None = None,
        trial_number: int | None = None,
    ) -> str:
        identifier = run_id or uuid.uuid4().hex
        config_dict = resolved_config(config)
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, parent_run_id, kind, name, status, started_at,
                    config_json, config_sha256, dataset_json, code_json,
                    environment_json, plan_json, study_name, trial_number
                ) VALUES (?, ?, ?, ?, 'running', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    parent_run_id,
                    kind,
                    config.name,
                    utc_now(),
                    json.dumps(config_dict, sort_keys=True),
                    sha256_json(config_dict),
                    manifest.model_dump_json(),
                    json.dumps(code_fingerprint(), sort_keys=True),
                    json.dumps(environment_snapshot(), sort_keys=True),
                    json.dumps(plan, sort_keys=True),
                    study_name,
                    trial_number,
                ),
            )
        return identifier

    def log_metrics(
        self,
        run_id: str,
        *,
        step: int,
        split: str,
        metrics: dict[str, float],
    ) -> None:
        now = utc_now()
        rows = [(run_id, step, split, name, value, now) for name, value in metrics.items()]
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO metrics (run_id, step, split, name, value, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (run_id, step, split, name)
                DO UPDATE SET value = excluded.value, recorded_at = excluded.recorded_at
                """,
                rows,
            )

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        error: str | None = None,
        summary: dict[str, Any] | None = None,
    ) -> None:
        if status not in {"completed", "failed", "pruned"}:
            raise ValueError(f"invalid terminal status: {status}")
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE runs
                SET status = ?, finished_at = ?, error = ?, summary_json = ?
                WHERE run_id = ?
                """,
                (
                    status,
                    utc_now(),
                    error,
                    json.dumps(summary, sort_keys=True) if summary is not None else None,
                    run_id,
                ),
            )

    def record_artifact(
        self,
        run_id: str,
        *,
        kind: str,
        path: Path,
        sha256: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO artifacts
                (run_id, kind, path, sha256, bytes, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, kind, str(path), sha256, path.stat().st_size, utc_now()),
            )

    def delete_artifacts(self, run_id: str, *, kind: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "DELETE FROM artifacts WHERE run_id = ? AND kind = ?",
                (run_id, kind),
            )

    def record_selection_decision(
        self,
        candidate_run_id: str,
        incumbent_run_id: str | None,
        *,
        candidate_evidence_sha256: str,
        incumbent_evidence_sha256: str | None,
        selected: bool,
        reason: str,
        statistics: dict[str, float | bool | int | None],
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO selection_decisions (
                    candidate_run_id, incumbent_run_id, candidate_evidence_sha256,
                    incumbent_evidence_sha256, selected, reason, statistics_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate_run_id,
                    incumbent_run_id,
                    candidate_evidence_sha256,
                    incumbent_evidence_sha256,
                    int(selected),
                    reason,
                    json.dumps(statistics, sort_keys=True),
                    utc_now(),
                ),
            )

    def active_selection(self) -> dict[str, Any] | None:
        with self.connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT * FROM selection_decisions
                WHERE selected = 1
                ORDER BY decision_id DESC
                LIMIT 1
                """
            ).fetchone()
        return dict(row) if row is not None else None

    def record_deployment(
        self,
        *,
        selection_run_id: str,
        refit_run_id: str,
        replaced_refit_run_id: str | None,
        bundle_path: Path,
        bundle_sha256: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO deployments (
                    selection_run_id, refit_run_id, replaced_refit_run_id,
                    bundle_path, bundle_sha256, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    selection_run_id,
                    refit_run_id,
                    replaced_refit_run_id,
                    str(bundle_path),
                    bundle_sha256,
                    utc_now(),
                ),
            )
            connection.execute("DELETE FROM artifacts WHERE kind = 'production_bundle'")
            connection.execute(
                """
                INSERT INTO artifacts
                (run_id, kind, path, sha256, bytes, created_at)
                VALUES (?, 'production_bundle', ?, ?, ?, ?)
                """,
                (
                    refit_run_id,
                    str(bundle_path),
                    bundle_sha256,
                    bundle_path.stat().st_size,
                    utc_now(),
                ),
            )

    def latest_deployment(self) -> dict[str, Any] | None:
        with self.connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM deployments ORDER BY deployment_id DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row is not None else None

    def latest_metrics(self, run_id: str, split: str) -> dict[str, float]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT name, value FROM metrics
                WHERE run_id = ? AND split = ?
                AND step = (
                    SELECT MAX(step) FROM metrics WHERE run_id = ? AND split = ?
                )
                """,
                (run_id, split, run_id, split),
            ).fetchall()
        return {str(name): float(value) for name, value in rows}

    def run(self, run_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return dict(row)
