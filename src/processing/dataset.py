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
    Dataset that loads the entire Parquet file into memory.

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

        # Convert features to tensor
        print("Converting features to tensor...")
        self.X = torch.tensor(df.select(self.feature_cols).to_numpy(), dtype=torch.long)

        # Convert labels if present
        if self.label_col and self.label_col in df.columns:
            print("Converting labels to tensor...")
            self.y = torch.tensor(df[self.label_col].to_numpy(), dtype=torch.float32)
        else:
            self.y = None

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
