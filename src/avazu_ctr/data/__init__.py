"""Leakage-safe Avazu data processing."""

from avazu_ctr.data.dataset import ParquetBatchDataset
from avazu_ctr.data.manifest import DatasetManifest, DatasetPurpose, load_manifest
from avazu_ctr.data.preprocessing import (
    preprocess_evaluation,
    preprocess_production,
    temporal_windows,
)

__all__ = [
    "DatasetManifest",
    "DatasetPurpose",
    "ParquetBatchDataset",
    "load_manifest",
    "preprocess_evaluation",
    "preprocess_production",
    "temporal_windows",
]
