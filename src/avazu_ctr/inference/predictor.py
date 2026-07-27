"""Typed batch prediction and deterministic Avazu submission writing."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from avazu_ctr.contracts import FeatureBatch
from avazu_ctr.data.dataset import ParquetBatchDataset
from avazu_ctr.data.manifest import load_manifest
from avazu_ctr.inference.bundle import LoadedBundle, load_bundle


class Predictor:
    def __init__(
        self,
        bundle_path: str | Path,
        *,
        device: str = "cpu",
        compile_model: bool = False,
    ) -> None:
        self.bundle: LoadedBundle = load_bundle(bundle_path, device=device)
        self.device = torch.device(device)
        self.model = self.bundle.model
        self.runtime_model = (
            torch.compile(self.model) if compile_model and hasattr(torch, "compile") else self.model
        )

    def validate_manifest_contract(self, manifest_path: str | Path) -> None:
        manifest = load_manifest(manifest_path, verify_shards=True)
        trained = self.bundle.manifest
        fitted_state = tuple(
            (table.feature, table.kind, table.sha256) for table in manifest.fitted_tables
        )
        trained_state = tuple(
            (table.feature, table.kind, table.sha256) for table in trained.fitted_tables
        )
        if (
            manifest.categorical_columns != trained.categorical_columns
            or manifest.numerical_columns != trained.numerical_columns
            or manifest.cardinalities != trained.cardinalities
            or manifest.embedding_kinds != trained.embedding_kinds
            or manifest.config_sha256 != trained.config_sha256
            or fitted_state != trained_state
        ):
            raise ValueError("prediction manifest does not match the promoted feature contract")

    @torch.inference_mode()
    def predict_batch(self, batch: FeatureBatch) -> torch.Tensor:
        moved = batch.to(self.device)
        return self.runtime_model(moved).probabilities().float().cpu()

    def iter_predictions(
        self,
        manifest_path: str | Path,
        *,
        split: str = "test",
        batch_size: int | None = None,
    ) -> Iterator[tuple[list[str], np.ndarray]]:
        self.validate_manifest_contract(manifest_path)
        dataset = ParquetBatchDataset(
            manifest_path,
            split,
            batch_size or self.bundle.config.training.batch_size,
            shuffle=False,
            seed=self.bundle.config.training.seed,
        )
        loader = DataLoader(dataset, batch_size=None)
        for batch in loader:
            if batch.row_ids is None:
                raise ValueError("prediction rows require source IDs")
            probabilities = self.predict_batch(batch).numpy().reshape(-1)
            yield batch.row_ids, probabilities

    def write_submission(
        self,
        manifest_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("id", "click"))
            for row_ids, probabilities in self.iter_predictions(manifest_path):
                writer.writerows(
                    (row_id, format(float(probability), ".10g"))
                    for row_id, probability in zip(row_ids, probabilities, strict=True)
                )
        return output
