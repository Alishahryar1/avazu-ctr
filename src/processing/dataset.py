"""
PyTorch Dataset for Avazu CTR data stored in Parquet format.

This module provides a dataset that loads the entire parquet file into memory
for maximum training performance.
"""

import torch
from torch.utils.data import Dataset
import polars as pl
from pathlib import Path


class ParquetFullDataset(Dataset):
    """
    Dataset that loads the entire Parquet file into memory using native Polars to_torch().

    FASTEST: Loads everything into CPU RAM (or GPU if mapped).
    High memory usage, but zero I/O during training.

    Args:
        parquet_path: Path to the parquet file
        feature_cols: List of feature column names
        label_col: Name of the label column (None for test data)
    """

    def __init__(
        self,
        parquet_path: str | Path,
        feature_cols: list[str],
        label_col: str | None = "click",
    ):
        self.parquet_path = Path(parquet_path)
        self.feature_cols = feature_cols
        self.label_col = label_col

        # Type declarations for attributes
        self.X: torch.Tensor
        self.y: torch.Tensor | None

        print(f"Loading full dataset from {self.parquet_path} into memory...")

        # Read entire file at once
        df = pl.scan_parquet(self.parquet_path).collect(engine="streaming")

        # Convert features to tensor using native Polars to_torch()
        print("Converting features to tensor...")
        self.X = df.select(self.feature_cols).to_torch(dtype=pl.Float32)

        # Convert labels if present using native Polars to_torch()
        if self.label_col and self.label_col in df.columns:
            print("Converting labels to tensor...")
            self.y = df.select(self.label_col).to_torch(dtype=pl.Float32).squeeze()
        else:
            self.y = None

        # Free polars dataframe memory
        del df

        self.n_samples = len(self.X)
        print(f"Loaded {self.n_samples:,} samples.")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        X = self.X[index]

        if self.y is not None:
            return X, self.y[index]

        return X
