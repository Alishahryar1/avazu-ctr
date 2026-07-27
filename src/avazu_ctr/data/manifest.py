"""Processed dataset manifests and integrity verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ShardManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    rows: int = Field(gt=0)
    sha256: str


class FittedTableManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    feature: str
    kind: str
    path: str
    rows: int = Field(ge=0)
    sha256: str


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    name: str
    raw_path: str
    raw_sha256: str
    train_start: int
    train_end: int
    valid_start: int
    valid_end: int
    categorical_columns: tuple[str, ...]
    numerical_columns: tuple[str, ...]
    cardinalities: dict[str, int]
    embedding_kinds: dict[str, str]
    train_shards: tuple[ShardManifest, ...]
    validation_shards: tuple[ShardManifest, ...]
    test_shards: tuple[ShardManifest, ...] = ()
    fitted_tables: tuple[FittedTableManifest, ...] = ()
    config_sha256: str
    package_lock_sha256: str | None = None

    @property
    def train_rows(self) -> int:
        return sum(shard.rows for shard in self.train_shards)

    @property
    def validation_rows(self) -> int:
        return sum(shard.rows for shard in self.validation_shards)


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_manifest(manifest: DatasetManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def load_manifest(path: str | Path, *, verify_shards: bool = False) -> DatasetManifest:
    manifest_path = Path(path)
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if verify_shards:
        root = manifest_path.parent
        for shard in (
            *manifest.train_shards,
            *manifest.validation_shards,
            *manifest.test_shards,
        ):
            shard_path = root / shard.path
            if sha256_file(shard_path) != shard.sha256:
                raise ValueError(f"checksum mismatch for {shard_path}")
    return manifest
