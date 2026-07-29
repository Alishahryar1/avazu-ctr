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
from avazu_ctr.data.manifest import DatasetPurpose, load_manifest, sha256_file
from avazu_ctr.inference.bundle import LoadedBundle, load_bundle
from avazu_ctr.inference.execution import InferenceRuntime


class Predictor:
    def __init__(
        self,
        bundle_path: str | Path,
        *,
        device: str = "cpu",
    ) -> None:
        self.bundle: LoadedBundle = load_bundle(bundle_path, device=device)
        self.device = torch.device(device)
        self.model = self.bundle.model
        self.runtime = InferenceRuntime(self.model, self.device)

    def validate_manifest_contract(self, manifest_path: str | Path) -> None:
        path = Path(manifest_path)
        manifest = load_manifest(path, verify_shards=True)
        if manifest.purpose is not DatasetPurpose.PRODUCTION:
            raise ValueError("prediction requires a production dataset")
        if sha256_file(path) != self.bundle.metadata["source_manifest_sha256"]:
            raise ValueError("prediction manifest is not the deployed production manifest")

    @torch.inference_mode()
    def predict_batch(self, batch: FeatureBatch) -> torch.Tensor:
        return self.runtime.predict(batch)

    def iter_predictions(
        self,
        manifest_path: str | Path,
        *,
        batch_size: int | None = None,
    ) -> Iterator[tuple[list[str], np.ndarray]]:
        self.validate_manifest_contract(manifest_path)
        dataset = ParquetBatchDataset(
            manifest_path,
            "test",
            batch_size or self.bundle.config.training.batch_size,
            shuffle=False,
            seed=self.bundle.config.training.seed,
            include_row_ids=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=None,
            pin_memory=self.device.type == "cuda",
        )
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
