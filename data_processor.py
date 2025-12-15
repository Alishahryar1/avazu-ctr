"""
Avazu CTR Data Processor - Polars Best Practices Implementation

This module uses idiomatic Polars patterns:
- Expression-based transformations (no Python loops in hot paths)
- Lazy evaluation with streaming for memory efficiency
- Vectorized vocabulary building and mapping
- Proper use of chained expressions
"""

import polars as pl
import numpy as np
import gc
from pathlib import Path
from config import CONFIG


# =============================================================================
# Schema Definition
# =============================================================================
SCHEMA = {
    'id': pl.String,
    'click': pl.UInt8,
    'hour': pl.String,
    'C1': pl.String,
    'banner_pos': pl.String,
    'site_id': pl.String,
    'site_domain': pl.String,
    'site_category': pl.String,
    'app_id': pl.String,
    'app_domain': pl.String,
    'app_category': pl.String,
    'device_id': pl.String,
    'device_ip': pl.String,
    'device_model': pl.String,
    'device_type': pl.String,
    'device_conn_type': pl.String,
    'C14': pl.String,
    'C15': pl.String,
    'C16': pl.String,
    'C17': pl.String,
    'C18': pl.String,
    'C19': pl.String,
    'C20': pl.String,
    'C21': pl.String,
}

CATEGORICAL_COLS = [
    'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category',
    'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip',
    'device_model', 'device_type', 'device_conn_type',
    'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'
]


# =============================================================================
# Feature Engineering Expressions
# =============================================================================
def get_time_feature_expressions() -> list[pl.Expr]:
    """Returns Polars expressions for time-based feature extraction."""
    return [
        pl.col("hour").str.slice(6, 2).cast(pl.UInt8).alias("hour_of_day"),
        pl.col("hour").str.slice(4, 2).cast(pl.UInt8).alias("day_of_week"),
    ]


def get_string_cast_expressions(columns: list[str]) -> list[pl.Expr]:
    """Returns expressions to cast columns to String type."""
    return [pl.col(c).cast(pl.String) for c in columns]


# =============================================================================
# Vocabulary Building (Vectorized)
# =============================================================================
def build_vocabularies(lf_train: pl.LazyFrame, cat_cols: list[str], min_freq: int) -> tuple[dict, dict]:
    """
    Build vocabularies using vectorized Polars operations.
    
    Returns:
        vocab_sizes: dict mapping column names to vocabulary sizes
        feat_maps: dict mapping column names to value->id dictionaries
    """
    print("Building vocabularies (vectorized)...")
    
    # Build all frequency counts in a single pass using unpivot + group_by
    # This is more efficient than iterating column by column
    freq_expressions = [
        pl.col(col).cast(pl.String).value_counts(sort=True).alias(col)
        for col in cat_cols
    ]
    
    vocab_sizes = {}
    feat_maps = {}
    
    # Collect frequencies for each column
    # Using select + explode pattern for vectorized counting
    for col in cat_cols:
        counts = (
            lf_train
            .select(pl.col(col).cast(pl.String))
            .group_by(col)
            .len()
            .filter(pl.col("len") >= min_freq)
            .sort(col)  # Deterministic ordering
            .collect()
        )
        
        # Build mapping: value -> sequential ID (starting at 1, 0 = <UNK>)
        values = counts[col].to_list()
        mapping = {val: idx + 1 for idx, val in enumerate(values)}
        
        feat_maps[col] = mapping
        vocab_sizes[col] = len(mapping) + 1  # +1 for <UNK> token
    
    print(f"Built vocabularies for {len(cat_cols)} columns")
    return vocab_sizes, feat_maps


# =============================================================================
# Data Transformation (Expression-Based)
# =============================================================================
def create_mapping_expressions(feat_maps: dict, cat_cols: list[str]) -> list[pl.Expr]:
    """
    Create Polars expressions for mapping categorical values to integer IDs.
    Uses vectorized replace_strict for optimal performance.
    """
    expressions = []
    for col in cat_cols:
        mapping = feat_maps[col]
        expr = (
            pl.col(col)
            .cast(pl.String)
            .replace_strict(mapping, default=0)
            .cast(pl.Int32)
            .alias(col)
        )
        expressions.append(expr)
    return expressions


def transform_dataframe(
    lf: pl.LazyFrame,
    feat_maps: dict,
    cat_cols: list[str],
    is_test: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Transform a lazy frame into numpy arrays using vectorized operations.
    
    Args:
        lf: Input LazyFrame
        feat_maps: Feature value to ID mappings
        cat_cols: List of categorical column names
        is_test: Whether this is test data
        
    Returns:
        X: Feature matrix (n_samples, n_features)
        extra: Labels (train) or IDs (test)
        hour_data: Raw hour strings for temporal splitting (train only)
    """
    # Build all mapping expressions
    mapping_exprs = create_mapping_expressions(feat_maps, cat_cols)
    
    # Select columns based on train/test
    if is_test:
        select_cols = ['id'] + cat_cols
    else:
        select_cols = ['click', 'hour'] + cat_cols
    
    # Apply all transformations in a single expression chain
    df = (
        lf
        .select(select_cols)
        .with_columns(mapping_exprs)
        .collect(engine="gpu")
    )
    
    # Extract arrays
    if is_test:
        extra = df['id'].to_numpy()
        hour_data = None
    else:
        extra = df['click'].to_numpy().astype(np.float32)
        hour_data = df['hour'].to_numpy()
    
    # Extract feature matrix (all mapped categorical columns)
    X = df.select(cat_cols).to_numpy().astype(np.int32)
    
    # Explicit cleanup
    del df
    gc.collect()
    
    return X, extra, hour_data


# =============================================================================
# Main Processing Pipeline
# =============================================================================
def process_data_polars() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, list]:
    """
    Main data processing pipeline using Polars best practices.
    
    Returns:
        X_train: Training feature matrix
        y_train: Training labels
        train_hours: Hour strings for temporal splitting
        X_test: Test feature matrix
        test_ids: Test IDs
        vocab_sizes: Vocabulary sizes per column
        cat_cols: List of categorical column names
    """
    print("Loading data with Polars...")
    
    # Lazy load with explicit schema
    lf_train = pl.scan_csv(CONFIG['train_path'], schema_overrides=SCHEMA)
    lf_test = pl.scan_csv(CONFIG['test_path'], schema_overrides=SCHEMA)
    
    # Apply time feature extraction (lazy)
    time_exprs = get_time_feature_expressions()
    lf_train = lf_train.with_columns(time_exprs)
    lf_test = lf_test.with_columns(time_exprs)
    
    # Extended categorical columns including time features
    cat_cols = CATEGORICAL_COLS + ['hour_of_day', 'day_of_week']
    
    # Build vocabularies from training data
    vocab_sizes, feat_maps = build_vocabularies(
        lf_train, 
        cat_cols, 
        CONFIG['min_freq']
    )
    
    print("Transforming data to numpy arrays...")
    
    # Transform train data
    X_train, y_train, train_hours = transform_dataframe(
        lf_train, feat_maps, cat_cols, is_test=False
    )
    print(f"Train processed: {X_train.shape}")
    
    # Transform test data
    X_test, test_ids, _ = transform_dataframe(
        lf_test, feat_maps, cat_cols, is_test=True
    )
    print(f"Test processed: {X_test.shape}")
    
    return X_train, y_train, train_hours, X_test, test_ids, vocab_sizes, cat_cols


# =============================================================================
# Data Persistence
# =============================================================================
def save_processed_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_hours: np.ndarray,
    X_test: np.ndarray,
    test_ids: np.ndarray,
    vocab_sizes: dict,
    feature_names: list
) -> None:
    """Save processed numpy arrays and metadata to disk."""
    import pickle
    
    path = Path(CONFIG['processed_path'])
    path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving processed data to {path}...")
    
    # Save arrays
    np.save(path / "X_train.npy", X_train)
    np.save(path / "y_train.npy", y_train)
    np.save(path / "train_hours.npy", train_hours)
    np.save(path / "X_test.npy", X_test)
    np.save(path / "test_ids.npy", test_ids)
    
    # Save metadata
    with open(path / "vocab_sizes.pkl", "wb") as f:
        pickle.dump(vocab_sizes, f)
    
    with open(path / "feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)
    
    print("Data saved successfully.")


def load_processed_data(mode: str = 'train') -> tuple[
    np.ndarray | None,  # X_train
    np.ndarray | None,  # y_train
    np.ndarray | None,  # train_hours
    np.ndarray,         # X_test
    np.ndarray,         # test_ids
    dict,               # vocab_sizes
    list                # feature_names
]:
    """
    Load processed data from disk.
    
    Args:
        mode: 'train' loads all data, 'inference' loads only test data
        
    Returns:
        Tuple of arrays and metadata
    """
    import pickle
    
    path = Path(CONFIG['processed_path'])
    print(f"Loading processed data from {path} ({mode} mode)...")
    
    try:
        # Load metadata (always needed)
        with open(path / "vocab_sizes.pkl", "rb") as f:
            vocab_sizes = pickle.load(f)
        with open(path / "feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)
        
        # Load test data (always needed)
        X_test = np.load(path / "X_test.npy", allow_pickle=True)
        test_ids = np.load(path / "test_ids.npy", allow_pickle=True)
        
        if mode == 'train':
            X_train = np.load(path / "X_train.npy", allow_pickle=True)
            y_train = np.load(path / "y_train.npy", allow_pickle=True)
            train_hours = np.load(path / "train_hours.npy", allow_pickle=True)
            return X_train, y_train, train_hours, X_test, test_ids, vocab_sizes, feature_names
        
        elif mode == 'inference':
            return None, None, None, X_test, test_ids, vocab_sizes, feature_names
        
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'inference'.")
    
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Processed data not found in {path}. Run 'python data_processor.py' first."
        ) from e


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    X_train, y_train, train_hours, X_test, test_ids, vocab_sizes, cat_cols = process_data_polars()
    save_processed_data(X_train, y_train, train_hours, X_test, test_ids, vocab_sizes, cat_cols)
