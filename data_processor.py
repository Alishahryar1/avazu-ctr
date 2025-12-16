#!/usr/bin/env python
"""Entry point for data processing."""
from src.data.data_processor import process_data_polars

if __name__ == "__main__":
    vocab_sizes, cat_cols, train_rows, test_rows = process_data_polars()
    print(f"\nVocabulary sizes: {len(vocab_sizes)} features")
    print(f"Feature names: {cat_cols[:5]}... ({len(cat_cols)} total)")
