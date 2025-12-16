"""
Test suite for dataset implementations.

This module tests the ParquetFullDataset class for loading
and handling Parquet files in PyTorch.
"""

import unittest
import torch
import numpy as np


class TestParquetFullDataset(unittest.TestCase):
    """Tests for ParquetFullDataset."""

    @classmethod
    def setUpClass(cls):
        """Create a temporary parquet file for testing."""
        import polars as pl
        from src.processing.dataset import ParquetFullDataset
        import tempfile
        import os

        cls.ParquetFullDataset = ParquetFullDataset

        # Create temp directory and file
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.parquet_path = os.path.join(cls.temp_dir.name, 'test_data.parquet')

        # Create dummy data
        # 100 rows, 2 features, 1 binary label
        data = {
            'feat1': np.random.randint(0, 10, 100),
            'feat2': np.random.randint(0, 10, 100),
            'click': np.random.randint(0, 2, 100).astype(np.float32)
        }
        df = pl.DataFrame(data)
        df.write_parquet(cls.parquet_path)

        cls.feature_cols = ['feat1', 'feat2']
        cls.label_col = 'click'

    @classmethod
    def tearDownClass(cls):
        """Cleanup temp files."""
        cls.temp_dir.cleanup()

    def test_dataset_initialization(self):
        """Test dataset loading and length."""
        dataset = self.ParquetFullDataset(
            self.parquet_path,
            self.feature_cols,
            self.label_col
        )
        self.assertEqual(len(dataset), 100)

    def test_getitem_single_row(self):
        """Test retrieving a single row."""
        dataset = self.ParquetFullDataset(
            self.parquet_path,
            self.feature_cols,
            self.label_col
        )

        X, y = dataset[0]

        self.assertIsInstance(X, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(X.shape, (2,))  # 2 features
        self.assertEqual(X.dtype, torch.long)
        self.assertEqual(y.shape, ())    # Scalar
        self.assertEqual(y.dtype, torch.float32)

    def test_getitem_inference_mode(self):
        """Test retrieving without labels."""
        dataset = self.ParquetFullDataset(
            self.parquet_path,
            self.feature_cols,
            label_col=None
        )

        X = dataset[0]

        self.assertIsInstance(X, torch.Tensor)
        self.assertEqual(X.shape, (2,))

    def test_data_loaded_in_memory(self):
        """Test that data is fully loaded into memory as tensors."""
        dataset = self.ParquetFullDataset(
            self.parquet_path,
            self.feature_cols,
            self.label_col
        )

        # X and y should be full tensors in memory
        self.assertIsInstance(dataset.X, torch.Tensor)
        self.assertIsInstance(dataset.y, torch.Tensor)
        self.assertEqual(dataset.X.shape, (100, 2))
        self.assertEqual(dataset.y.shape, (100,))

    def test_dataloader_integration(self):
        """Test that it works with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        dataset = self.ParquetFullDataset(
            self.parquet_path,
            self.feature_cols,
            self.label_col
        )

        loader = DataLoader(dataset, batch_size=20, shuffle=False)

        batches = list(loader)
        self.assertEqual(len(batches), 5)  # 100 / 20 = 5 batches

        # Verify first batch shape
        X, y = batches[0]
        self.assertEqual(X.shape, (20, 2))
        self.assertEqual(y.shape, (20,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
