"""
Memory-efficient PyTorch Dataset for Avazu CTR data stored in Parquet format.

This module provides datasets that read data on-demand from parquet files,
avoiding loading the entire dataset into memory.
"""

import torch
from torch.utils.data import Dataset, Sampler
import polars as pl
import numpy as np
from pathlib import Path
from typing import Iterator, Sized


class ParquetDataset(Dataset):
    """
    PyTorch Dataset that reads from a Parquet file on-demand.

    MEMORY-EFFICIENT: Only loads one batch at a time into memory.
    Uses Polars' efficient parquet scanning with slice operations.

    Args:
        parquet_path: Path to the parquet file
        feature_cols: List of feature column names
        label_col: Name of the label column (None for test data)
        cache_size: Number of rows to cache (default 0 = no caching)
    """

    def __init__(
        self,
        parquet_path: str | Path,
        feature_cols: list[str],
        label_col: str | None = 'click',
        cache_size: int = 0
    ):
        self.parquet_path = Path(parquet_path)
        self.feature_cols = feature_cols
        self.label_col = label_col

        # Get total row count
        self._length = pl.scan_parquet(self.parquet_path).select(pl.len()).collect().item()

        # Optional caching for small datasets or validation
        self.cache_size = cache_size
        self._cache = {}
        self._cache_start = -1

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        # Check cache first
        if self.cache_size > 0 and self._cache_start <= index < self._cache_start + self.cache_size:
            cache_idx = index - self._cache_start
            if self.label_col:
                return self._cache['X'][cache_idx], self._cache['y'][cache_idx]
            return self._cache['X'][cache_idx]

        # Read single row from parquet (this is slow for individual items)
        # Note: For better performance, use ParquetBatchDataset with a custom sampler
        row = (
            pl.scan_parquet(self.parquet_path)
            .slice(index, 1)
            .collect()
        )

        X = torch.tensor(row.select(self.feature_cols).to_numpy()[0], dtype=torch.long)

        if self.label_col:
            y = torch.tensor(row[self.label_col].to_numpy()[0], dtype=torch.float32)
            return X, y

        return X

    def get_batch(self, start_idx: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Efficiently read a contiguous batch of rows.

        This is much more efficient than __getitem__ for individual rows.

        Args:
            start_idx: Starting row index
            batch_size: Number of rows to read

        Returns:
            X: Feature tensor of shape (batch_size, n_features)
            y: Label tensor of shape (batch_size,) or None for test data
        """
        actual_size = min(batch_size, self._length - start_idx)

        batch = (
            pl.scan_parquet(self.parquet_path)
            .slice(start_idx, actual_size)
            .collect()
        )

        X = torch.tensor(batch.select(self.feature_cols).to_numpy(), dtype=torch.long)

        if self.label_col:
            y = torch.tensor(batch[self.label_col].to_numpy(), dtype=torch.float32)
            return X, y

        return X, None


class ContiguousBatchSampler(Sampler[list[int]]):
    """
    Sampler that yields contiguous batches of indices.

    This is optimized for parquet reading where sequential access is efficient.
    Shuffling is done at the batch level, not the row level.

    Args:
        data_source: Dataset to sample from
        batch_size: Size of each batch
        shuffle: Whether to shuffle batch order (not row order)
        drop_last: Whether to drop the last incomplete batch
    """

    def __init__(
        self,
        data_source: Sized,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.n_samples = len(data_source)
        self.n_batches = self.n_samples // batch_size
        if not drop_last and self.n_samples % batch_size != 0:
            self.n_batches += 1

    def __iter__(self) -> Iterator[list[int]]:
        # Generate batch start indices
        batch_starts = list(range(0, self.n_samples, self.batch_size))

        if self.drop_last and len(batch_starts) > 0:
            # Check if last batch is incomplete
            if batch_starts[-1] + self.batch_size > self.n_samples:
                batch_starts = batch_starts[:-1]

        if self.shuffle:
            np.random.shuffle(batch_starts)

        for start in batch_starts:
            end = min(start + self.batch_size, self.n_samples)
            yield list(range(start, end))

    def __len__(self) -> int:
        return self.n_batches


class ParquetBatchDataset(Dataset):
    """
    Dataset optimized for batch-level access to parquet files.

    Instead of reading individual rows, this dataset pre-computes batch
    boundaries and reads entire batches at once, which is much more efficient.

    Args:
        parquet_path: Path to the parquet file
        feature_cols: List of feature column names
        label_col: Name of the label column (None for test data)
        batch_size: Size of each batch
        shuffle: Whether to shuffle batch order each epoch
    """

    def __init__(
        self,
        parquet_path: str | Path,
        feature_cols: list[str],
        label_col: str | None = 'click',
        batch_size: int = 4096,
        shuffle: bool = True
    ):
        self.parquet_path = Path(parquet_path)
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Get total row count
        self.n_samples = pl.scan_parquet(self.parquet_path).select(pl.len()).collect().item()

        # Compute batch boundaries
        self.batch_starts = list(range(0, self.n_samples, batch_size))
        self.n_batches = len(self.batch_starts)

        # Shuffle order (recomputed each epoch via reset())
        self._order = list(range(self.n_batches))
        if shuffle:
            np.random.shuffle(self._order)

    def __len__(self) -> int:
        return self.n_batches

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Get a batch by batch index (not row index)."""
        actual_idx = self._order[index]
        start = self.batch_starts[actual_idx]
        actual_size = min(self.batch_size, self.n_samples - start)

        batch = (
            pl.scan_parquet(self.parquet_path)
            .slice(start, actual_size)
            .collect()
        )

        X = torch.tensor(batch.select(self.feature_cols).to_numpy(), dtype=torch.long)

        if self.label_col:
            y = torch.tensor(batch[self.label_col].to_numpy(), dtype=torch.float32)
            return X, y

        return X, None

    def reset(self):
        """Reset shuffle order for new epoch."""
        if self.shuffle:
            np.random.shuffle(self._order)

    @property
    def total_samples(self) -> int:
        """Total number of samples (rows) in the dataset."""
        return self.n_samples



