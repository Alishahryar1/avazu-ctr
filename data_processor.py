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

# Base categorical columns from raw data
BASE_CATEGORICAL_COLS = [
    'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category',
    'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip',
    'device_model', 'device_type', 'device_conn_type',
    'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'
]

# Engineered categorical columns (user proxy + interaction features)
ENGINEERED_CATEGORICAL_COLS = [
    'user_proxy',
    'device_id_x_app_id',
    'device_ip_x_C14',
]

# Combined list for vocabulary building
CATEGORICAL_COLS = BASE_CATEGORICAL_COLS + ENGINEERED_CATEGORICAL_COLS


# =============================================================================
# Feature Engineering Expressions
# =============================================================================
def get_time_feature_expressions() -> list[pl.Expr]:
    """Returns Polars expressions for time-based feature extraction.
    
    The 'hour' column format is YYMMDDHH (e.g., 14102100 = 2014-10-21, hour 00).
    """
    return [
        # Extract year (positions 0-1, e.g., "14" -> 14)
        pl.col("hour").str.slice(0, 2).cast(pl.UInt8).alias("year"),
        # Extract month (positions 2-3, e.g., "10" -> 10)
        pl.col("hour").str.slice(2, 2).cast(pl.UInt8).alias("month"),
        # Extract day of month (positions 4-5, e.g., "21" -> 21)
        pl.col("hour").str.slice(4, 2).cast(pl.UInt8).alias("day_of_month"),
        # Extract hour of day (positions 6-7, e.g., "00" -> 0)
        pl.col("hour").str.slice(6, 2).cast(pl.UInt8).alias("hour_of_day"),
        # Calculate actual day of week (0=Monday, 6=Sunday) by parsing as datetime
        # Append "00" for minutes since Polars requires both hour and minute in format string
        (pl.col("hour") + "00").str.to_datetime("%y%m%d%H%M").dt.weekday().cast(pl.UInt8).alias("day_of_week"),
    ]


def get_user_proxy_expression() -> pl.Expr:
    """Creates a user proxy ID from device_ip + device_model.
    
    Since device_id is null for ~82% of rows, we create a proxy user identifier
    by combining device_ip and device_model. This helps identify returning users.
    """
    return pl.concat_str(
        [pl.col("device_ip"), pl.col("device_model")],
        separator="_"
    ).alias("user_proxy")


def get_interaction_feature_expressions() -> list[pl.Expr]:
    """Creates interaction features by combining strong feature pairs.
    
    Key interactions identified from competition winners:
    - device_id + app_id: Captures user-app affinity
    - device_ip + C14: Captures IP-level behavioral patterns
    - user_proxy + app_id: Similar to device_id+app_id but more coverage
    """
    return [
        pl.concat_str(
            [pl.col("device_id"), pl.col("app_id")],
            separator="_"
        ).alias("device_id_x_app_id"),
        pl.concat_str(
            [pl.col("device_ip"), pl.col("C14")],
            separator="_"
        ).alias("device_ip_x_C14"),
    ]


# Columns to compute count features for (high-cardinality)
COUNT_FEATURE_COLS = ["device_ip", "device_id", "C14", "C17", "C21", "user_proxy"]


def get_string_cast_expressions(columns: list[str]) -> list[pl.Expr]:
    """Returns expressions to cast columns to String type."""
    return [pl.col(c).cast(pl.String) for c in columns]


def compute_count_features_from_train(
    lf_train: pl.LazyFrame,
    lf_test: pl.LazyFrame,
    count_cols: list[str]
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Compute count/frequency features based on training data statistics.
    
    For each column in count_cols, adds a new column '{col}_count' representing
    how many times that value appears in the training set. This helps the model
    distinguish between rare and frequent values.
    
    Args:
        lf_train: Training LazyFrame
        lf_test: Test LazyFrame  
        count_cols: Columns to compute counts for
        
    Returns:
        Updated train and test LazyFrames with count features
    """
    print(f"Computing count features for: {count_cols}")
    
    # Compute counts from training data only (to avoid leakage)
    for col in count_cols:
        # Get value counts from training data
        count_df = (
            lf_train
            .select(pl.col(col).cast(pl.String))
            .group_by(col)
            .len()
            .rename({"len": f"{col}_count"})
            .collect()
            .lazy()
        )
        
        # Join counts to both train and test
        # Use left join so missing values get null (which we'll fill with 0)
        lf_train = (
            lf_train
            .join(count_df, on=col, how="left")
            .with_columns(
                pl.col(f"{col}_count").fill_null(0).cast(pl.UInt32)
            )
        )
        lf_test = (
            lf_test
            .join(count_df, on=col, how="left")
            .with_columns(
                pl.col(f"{col}_count").fill_null(0).cast(pl.UInt32)
            )
        )
    
    return lf_train, lf_test


def bin_count_features(count_cols: list[str]) -> list[pl.Expr]:
    """
    Create binned versions of count features for use as categorical features.
    
    Bins: 0, 1, 2-5, 6-10, 11-50, 51-100, 101-500, 501-1000, 1000+
    This converts continuous counts into categorical buckets.
    """
    expressions = []
    for col in count_cols:
        count_col = f"{col}_count"
        binned_col = f"{col}_count_bin"
        expr = (
            pl.when(pl.col(count_col) == 0).then(pl.lit("0"))
            .when(pl.col(count_col) == 1).then(pl.lit("1"))
            .when(pl.col(count_col) <= 5).then(pl.lit("2-5"))
            .when(pl.col(count_col) <= 10).then(pl.lit("6-10"))
            .when(pl.col(count_col) <= 50).then(pl.lit("11-50"))
            .when(pl.col(count_col) <= 100).then(pl.lit("51-100"))
            .when(pl.col(count_col) <= 500).then(pl.lit("101-500"))
            .when(pl.col(count_col) <= 1000).then(pl.lit("501-1000"))
            .otherwise(pl.lit("1000+"))
            .alias(binned_col)
        )
        expressions.append(expr)
    return expressions


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
        .collect()
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
    
    # === Step 1: Create user proxy (must come before interactions that use it) ===
    print("Creating user proxy feature...")
    user_proxy_expr = get_user_proxy_expression()
    lf_train = lf_train.with_columns(user_proxy_expr)
    lf_test = lf_test.with_columns(user_proxy_expr)
    
    # === Step 2: Create interaction features ===
    print("Creating interaction features...")
    interaction_exprs = get_interaction_feature_expressions()
    lf_train = lf_train.with_columns(interaction_exprs)
    lf_test = lf_test.with_columns(interaction_exprs)
    
    # === Step 3: Apply time feature extraction ===
    print("Extracting time features...")
    time_exprs = get_time_feature_expressions()
    lf_train = lf_train.with_columns(time_exprs)
    lf_test = lf_test.with_columns(time_exprs)
    
    # === Step 4: Compute count features (based on train stats) ===
    lf_train, lf_test = compute_count_features_from_train(
        lf_train, lf_test, COUNT_FEATURE_COLS
    )
    
    # === Step 5: Bin count features for categorical encoding ===
    print("Binning count features...")
    bin_exprs = bin_count_features(COUNT_FEATURE_COLS)
    lf_train = lf_train.with_columns(bin_exprs)
    lf_test = lf_test.with_columns(bin_exprs)
    
    # === Build final feature list ===
    # Categorical columns: base + engineered + time + count bins
    count_bin_cols = [f"{col}_count_bin" for col in COUNT_FEATURE_COLS]
    cat_cols = CATEGORICAL_COLS + ['year', 'month', 'day_of_month', 'hour_of_day', 'day_of_week'] + count_bin_cols
    
    print(f"Total categorical features: {len(cat_cols)}")
    
    # Build vocabularies from training data
    vocab_sizes, feat_maps = build_vocabularies(
        lf_train, 
        cat_cols, 
        CONFIG['min_freq']
    )
    
    print("Transforming data to numpy arrays...")
    
    # Transform train data
    X_train, y_train, train_hours_raw = transform_dataframe(
        lf_train, feat_maps, cat_cols, is_test=False
    )
    # train_hours is guaranteed to be non-None when is_test=False
    assert train_hours_raw is not None
    train_hours = train_hours_raw
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
