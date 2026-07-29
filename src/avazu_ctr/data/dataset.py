"""Iterable typed batches over processed Parquet shards."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import torch
from torch.utils.data import IterableDataset

from avazu_ctr.contracts import FeatureBatch
from avazu_ctr.data.manifest import DatasetManifest, ShardManifest


class ParquetBatchDataset(IterableDataset[FeatureBatch]):
    """Stream projected, globally coalesced batches from typed Parquet shards."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        batch_size: int,
        *,
        shuffle: bool = False,
        seed: int = 42,
        include_row_ids: bool = False,
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
        self.include_row_ids = include_row_ids
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _worker_shards(self) -> tuple[list[ShardManifest], int]:
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        shards = list(self.shards[worker_id::worker_count])
        if self.shuffle:
            random.Random(self.seed + self.epoch * 1_000_003 + worker_id).shuffle(shards)
        return shards, worker_id

    def _columns(self) -> list[str]:
        columns = [
            *self.manifest.categorical_columns,
            *self.manifest.numerical_columns,
        ]
        if self.split != "test":
            columns.append("click")
        if self.include_row_ids:
            columns.append("id")
        return columns

    def _batch_from_frame(self, batch: pl.DataFrame) -> FeatureBatch:
        categorical = torch.from_numpy(
            batch.select(self.manifest.categorical_columns).to_numpy(
                order="fortran",
                writable=True,
            )
        )
        numerical = torch.from_numpy(
            batch.select(self.manifest.numerical_columns).to_numpy(
                order="fortran",
                writable=True,
            )
        )
        labels = None
        if self.split != "test":
            labels = torch.from_numpy(batch["click"].to_numpy(writable=True)).reshape(-1, 1)
        row_ids = cast(list[str], batch["id"].to_list()) if self.include_row_ids else None
        return FeatureBatch(
            categorical=categorical,
            numerical=numerical,
            labels=labels,
            row_ids=row_ids,
        )

    def __iter__(self) -> Iterator[FeatureBatch]:
        shards, worker_id = self._worker_shards()
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + worker_id)
        carry: pl.DataFrame | None = None

        for shard in shards:
            frame = pl.read_parquet(
                self.manifest_path.parent / shard.path,
                columns=self._columns(),
            )
            if self.shuffle:
                order = np.arange(frame.height)
                rng.shuffle(order)
                frame = frame[order]

            if carry is not None:
                needed = self.batch_size - carry.height
                if frame.height < needed:
                    carry = pl.concat((carry, frame), how="vertical", rechunk=False)
                    continue
                batch = pl.concat(
                    (carry, frame.head(needed)),
                    how="vertical",
                    rechunk=False,
                )
                yield self._batch_from_frame(batch)
                frame = frame.slice(needed)
                carry = None

            complete_rows = frame.height - frame.height % self.batch_size
            for start in range(0, complete_rows, self.batch_size):
                yield self._batch_from_frame(frame.slice(start, self.batch_size))
            if complete_rows < frame.height:
                carry = frame.slice(complete_rows)

        if carry is not None:
            yield self._batch_from_frame(carry)
