"""Iterable typed batches over processed Parquet shards."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import IterableDataset

from avazu_ctr.contracts import FeatureBatch
from avazu_ctr.data.manifest import DatasetManifest, ShardManifest


class ParquetBatchDataset(IterableDataset[FeatureBatch]):
    """Streams already-typed shards without materializing the full dataset."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        batch_size: int,
        *,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.manifest_path = Path(manifest_path)
        self.manifest = DatasetManifest.model_validate_json(
            self.manifest_path.read_text(encoding="utf-8")
        )
        choices: dict[str, Sequence[ShardManifest]] = {
            "train": self.manifest.train_shards,
            "validation": self.manifest.validation_shards,
            "test": self.manifest.test_shards,
        }
        if split not in choices:
            raise ValueError(f"unknown split {split!r}")
        self.shards = tuple(choices[split])
        if not self.shards:
            raise ValueError(f"manifest has no {split} shards")
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _worker_shards(self) -> list[ShardManifest]:
        shards = list(self.shards)
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(shards)
        worker = torch.utils.data.get_worker_info()
        return shards if worker is None else shards[worker.id :: worker.num_workers]

    def __iter__(self) -> Iterator[FeatureBatch]:
        rng = np.random.default_rng(self.seed + self.epoch)
        for shard in self._worker_shards():
            frame = pl.read_parquet(self.manifest_path.parent / shard.path)
            order = np.arange(frame.height)
            if self.shuffle:
                rng.shuffle(order)
            for start in range(0, frame.height, self.batch_size):
                positions = order[start : start + self.batch_size]
                batch = frame[positions]
                categorical = torch.from_numpy(
                    batch.select(self.manifest.categorical_columns)
                    .to_numpy()
                    .astype(np.int64, copy=True)
                )
                numerical = torch.from_numpy(
                    batch.select(self.manifest.numerical_columns)
                    .to_numpy()
                    .astype(np.float32, copy=True)
                )
                labels = None
                if "click" in batch.columns:
                    labels = torch.from_numpy(
                        batch["click"].to_numpy().astype(np.float32, copy=True)
                    ).reshape(-1, 1)
                yield FeatureBatch(
                    categorical=categorical,
                    numerical=numerical,
                    labels=labels,
                    row_ids=batch["id"].to_list(),
                    timestamps=torch.from_numpy(
                        batch["_timestamp_hour"].to_numpy().astype(np.int64, copy=True)
                    ),
                )
