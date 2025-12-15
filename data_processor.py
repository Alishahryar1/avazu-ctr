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

# Columns to compute cumulative count features for (captures user "maturity")
CUMCOUNT_COLS = ["device_ip", "user_proxy", "device_id"]


def get_cumulative_count_expressions(cols: list[str]) -> list[pl.Expr]:
    """
    Create cumulative count expressions for specified columns.
    
    Cumulative count captures how many times a value has appeared UP TO this point
    in the dataset (when sorted chronologically). This helps distinguish:
    - New users (low cumcount) vs returning users (high cumcount)
    - First impressions vs repeat impressions
    
    Args:
        cols: Columns to compute cumulative counts for
        
    Returns:
        List of Polars expressions for cumulative counts
    """
    return [
        pl.col(col).cum_count().over(col).alias(f"{col}_cumcount")
        for col in cols
    ]


def bin_cumcount_features(cols: list[str]) -> list[pl.Expr]:
    """
    Create binned versions of cumulative count features for categorical encoding.
    
    Bins: 1 (first), 2-3, 4-10, 11-50, 51-100, 100+
    These bins distinguish user engagement levels.
    """
    expressions = []
    for col in cols:
        cumcount_col = f"{col}_cumcount"
        binned_col = f"{col}_cumcount_bin"
        expr = (
            pl.when(pl.col(cumcount_col) == 1).then(pl.lit("first"))
            .when(pl.col(cumcount_col) <= 3).then(pl.lit("2-3"))
            .when(pl.col(cumcount_col) <= 10).then(pl.lit("4-10"))
            .when(pl.col(cumcount_col) <= 50).then(pl.lit("11-50"))
            .when(pl.col(cumcount_col) <= 100).then(pl.lit("51-100"))
            .otherwise(pl.lit("100+"))
            .alias(binned_col)
        )
        expressions.append(expr)
    return expressions


# =============================================================================
# Time-Delta Features (Hours Since Last Click)
# =============================================================================
def compute_time_delta_features(
    lf: pl.LazyFrame,
    group_col: str = "user_proxy"
) -> pl.LazyFrame:
    """
    Compute time delta features - hours since last click for each user.
    
    This captures click velocity patterns:
    - Bots/fraudulent clicks have very short intervals
    - Normal users have longer intervals between clicks
    
    Note: Since the 'hour' column is YYMMDDHH format (hour-level granularity),
    we compute hours since last click, not seconds.
    
    Args:
        lf: Input LazyFrame (must be sorted by 'hour')
        group_col: Column to group by for computing deltas (default: user_proxy)
        
    Returns:
        LazyFrame with time delta features added
    """
    # Parse hour column to datetime for time delta computation
    # Format: YYMMDDHH -> add "00" for minutes
    lf = lf.with_columns(
        (pl.col("hour") + "00")
        .str.to_datetime("%y%m%d%H%M")
        .alias("_timestamp")
    )
    
    # Compute hours since previous click for this user
    lf = lf.with_columns(
        (
            pl.col("_timestamp") - pl.col("_timestamp").shift(1).over(group_col)
        )
        .dt.total_hours()
        .fill_null(0)  # First click has no previous
        .cast(pl.UInt32)
        .alias("hours_since_last_click")
    )
    
    # Drop temporary timestamp column
    lf = lf.drop("_timestamp")
    
    return lf


def bin_time_delta_features() -> pl.Expr:
    """
    Bin hours_since_last_click into categorical buckets.
    
    Bins:
    - 'first': First click (0 hours delta)
    - 'same_hour': Same hour as previous click (would be 0 but we use 'first' for that)
    - '<1h': Within 1 hour
    - '1-6h': 1-6 hours
    - '6-24h': 6-24 hours (same day roughly)
    - '1-7d': 1-7 days
    - '>7d': More than a week
    """
    return (
        pl.when(pl.col("hours_since_last_click") == 0).then(pl.lit("first"))
        .when(pl.col("hours_since_last_click") < 1).then(pl.lit("same_hour"))
        .when(pl.col("hours_since_last_click") < 6).then(pl.lit("1-6h"))
        .when(pl.col("hours_since_last_click") < 24).then(pl.lit("6-24h"))
        .when(pl.col("hours_since_last_click") < 168).then(pl.lit("1-7d"))  # 24*7 = 168
        .otherwise(pl.lit(">7d"))
        .alias("hours_since_last_click_bin")
    )


# =============================================================================
# Previous Click Count Features (Rolling Count)
# =============================================================================
def compute_previous_click_count(
    lf: pl.LazyFrame,
    group_col: str = "user_proxy"
) -> pl.LazyFrame:
    """
    Compute the number of previous clicks for each user up to (but not including) current row.
    
    This is similar to cumulative count but shifted by 1, so it represents
    "how many clicks has this user made BEFORE this one".
    
    This is more useful than cumulative count for prediction because:
    - cumcount includes the current row
    - prev_click_count is what we would know at prediction time
    
    Args:
        lf: Input LazyFrame (must be sorted chronologically)
        group_col: Column to group by
        
    Returns:
        LazyFrame with previous click count feature
    """
    return lf.with_columns(
        (pl.col(group_col).cum_count().over(group_col) - 1)
        .clip(lower_bound=0)
        .cast(pl.UInt32)
        .alias(f"{group_col}_prev_clicks")
    )


def bin_prev_clicks(group_col: str) -> pl.Expr:
    """
    Bin previous click counts into categorical buckets.
    
    Bins: new (0), returning (1-5), regular (6-20), heavy (21-100), power (100+)
    """
    col_name = f"{group_col}_prev_clicks"
    return (
        pl.when(pl.col(col_name) == 0).then(pl.lit("new"))
        .when(pl.col(col_name) <= 5).then(pl.lit("returning"))
        .when(pl.col(col_name) <= 20).then(pl.lit("regular"))
        .when(pl.col(col_name) <= 100).then(pl.lit("heavy"))
        .otherwise(pl.lit("power"))
        .alias(f"{col_name}_bin")
    )


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


def compute_hourly_aggregated_features(
    lf_train: pl.LazyFrame,
    lf_test: pl.LazyFrame
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Compute hourly aggregated features for user activity.
    
    This is an efficient alternative to rolling window features.
    For each row, adds the count of impressions this user had in the same hour.
    
    Features added:
    - user_hourly_impressions: Number of impressions for this user_proxy in this hour
    
    Args:
        lf_train: Training LazyFrame
        lf_test: Test LazyFrame
        
    Returns:
        Updated train and test LazyFrames with hourly features
    """
    print("Computing hourly aggregated features...")
    
    # Compute user impressions per hour from training data
    hourly_counts = (
        lf_train
        .select(["user_proxy", "hour"])
        .group_by(["user_proxy", "hour"])
        .len()
        .rename({"len": "user_hourly_impressions"})
        .collect()
        .lazy()
    )
    
    # Join to both train and test
    lf_train = (
        lf_train
        .join(hourly_counts, on=["user_proxy", "hour"], how="left")
        .with_columns(
            pl.col("user_hourly_impressions").fill_null(1).cast(pl.UInt32)
        )
    )
    lf_test = (
        lf_test
        .join(hourly_counts, on=["user_proxy", "hour"], how="left")
        .with_columns(
            pl.col("user_hourly_impressions").fill_null(1).cast(pl.UInt32)
        )
    )
    
    return lf_train, lf_test


def bin_hourly_impressions() -> pl.Expr:
    """
    Bin hourly impressions into categorical buckets.
    
    Bins: 1 (single), 2-3, 4-10, 11-50, 50+
    """
    return (
        pl.when(pl.col("user_hourly_impressions") == 1).then(pl.lit("single"))
        .when(pl.col("user_hourly_impressions") <= 3).then(pl.lit("2-3"))
        .when(pl.col("user_hourly_impressions") <= 10).then(pl.lit("4-10"))
        .when(pl.col("user_hourly_impressions") <= 50).then(pl.lit("11-50"))
        .otherwise(pl.lit("50+"))
        .alias("user_hourly_impressions_bin")
    )


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
    
    # === Step 6: Compute cumulative count features (user maturity) ===
    print("Computing cumulative count features...")
    # Sort by hour for chronological order, then compute cumulative counts
    lf_train = lf_train.sort("hour")
    lf_test = lf_test.sort("hour")
    
    cumcount_exprs = get_cumulative_count_expressions(CUMCOUNT_COLS)
    lf_train = lf_train.with_columns(cumcount_exprs)
    lf_test = lf_test.with_columns(cumcount_exprs)
    
    # Bin cumulative counts
    cumcount_bin_exprs = bin_cumcount_features(CUMCOUNT_COLS)
    lf_train = lf_train.with_columns(cumcount_bin_exprs)
    lf_test = lf_test.with_columns(cumcount_bin_exprs)
    
    # === Step 7: Compute hourly aggregated features ===
    lf_train, lf_test = compute_hourly_aggregated_features(lf_train, lf_test)
    
    # Bin hourly impressions
    hourly_bin_expr = bin_hourly_impressions()
    lf_train = lf_train.with_columns(hourly_bin_expr)
    lf_test = lf_test.with_columns(hourly_bin_expr)
    
    # === Step 8: Compute time-delta features (hours since last click) ===
    print("Computing time-delta features...")
    lf_train = compute_time_delta_features(lf_train, group_col="user_proxy")
    lf_test = compute_time_delta_features(lf_test, group_col="user_proxy")
    
    # Bin time delta
    time_delta_bin_expr = bin_time_delta_features()
    lf_train = lf_train.with_columns(time_delta_bin_expr)
    lf_test = lf_test.with_columns(time_delta_bin_expr)
    
    # === Step 9: Compute previous click count features ===
    print("Computing previous click count features...")
    lf_train = compute_previous_click_count(lf_train, group_col="user_proxy")
    lf_test = compute_previous_click_count(lf_test, group_col="user_proxy")
    
    # Bin previous clicks
    prev_clicks_bin_expr = bin_prev_clicks("user_proxy")
    lf_train = lf_train.with_columns(prev_clicks_bin_expr)
    lf_test = lf_test.with_columns(prev_clicks_bin_expr)
    
    # === Build final feature list ===
    # Categorical columns: base + engineered + time + count bins + cumcount bins + hourly bins + time delta + prev clicks
    count_bin_cols = [f"{col}_count_bin" for col in COUNT_FEATURE_COLS]
    cumcount_bin_cols = [f"{col}_cumcount_bin" for col in CUMCOUNT_COLS]
    cat_cols = (
        CATEGORICAL_COLS 
        + ['year', 'month', 'day_of_month', 'hour_of_day', 'day_of_week'] 
        + count_bin_cols 
        + cumcount_bin_cols 
        + ['user_hourly_impressions_bin']
        + ['hours_since_last_click_bin']
        + ['user_proxy_prev_clicks_bin']
    )
    
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
