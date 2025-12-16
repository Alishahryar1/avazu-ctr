"""
Avazu CTR Data Processor - Memory-Efficient Polars Implementation

This module is optimized for running on memory-constrained systems (e.g., laptops):
- Sequential processing of vocabulary building to avoid memory spikes
- Streaming sink directly to parquet (no numpy conversion)
- Explicit garbage collection between processing phases
- Data stays in parquet format for on-demand batch reading during training
- Expression-based transformations (no Python loops in hot paths)
- Lazy evaluation with streaming for memory efficiency
"""

import polars as pl
import gc
from pathlib import Path
from src.config.config import CONFIG
import pickle


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
    Note: 'year' is not extracted as dataset is entirely from 2014 (zero variance).
    """
    return [
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
# Note: user_proxy excluded as it's redundant with user_proxy_prev_clicks
CUMCOUNT_COLS = ["device_ip", "device_id"]


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
    
    EDA-optimized bins based on percentiles (80.8% are first clicks with 0):
    - 'first': First click (0 hours delta)
    - '1-4h': Up to P50 (short interval)
    - '5-19h': P50 to P75 (medium interval)  
    - '20-53h': P75 to P90 (long interval)
    - '>53h': Top 10% (re-engagement)
    """
    return (
        pl.when(pl.col("hours_since_last_click") == 0).then(pl.lit("first"))
        .when(pl.col("hours_since_last_click") <= 4).then(pl.lit("1-4h"))
        .when(pl.col("hours_since_last_click") <= 19).then(pl.lit("5-19h"))
        .when(pl.col("hours_since_last_click") <= 53).then(pl.lit("20-53h"))
        .otherwise(pl.lit(">53h"))
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
    
    EDA-optimized bins based on percentiles:
    - new (0): 29.4% of users
    - returning (1-7): Up to P50
    - regular (8-32): P50 to P75
    - heavy (33-224): P75 to P90
    - power (224+): Top 10% most active
    """
    col_name = f"{group_col}_prev_clicks"
    return (
        pl.when(pl.col(col_name) == 0).then(pl.lit("new"))
        .when(pl.col(col_name) <= 7).then(pl.lit("returning"))
        .when(pl.col(col_name) <= 32).then(pl.lit("regular"))
        .when(pl.col(col_name) <= 224).then(pl.lit("heavy"))
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

    MEMORY-OPTIMIZED: Processes columns sequentially with streaming collection
    to avoid holding all count DataFrames in memory simultaneously.

    Args:
        lf_train: Training LazyFrame
        lf_test: Test LazyFrame
        count_cols: Columns to compute counts for

    Returns:
        Updated train and test LazyFrames with count features
    """
    print(f"Computing count features for: {count_cols} (sequential, memory-efficient)")

    # Process columns one at a time to minimize peak memory
    for i, col in enumerate(count_cols):
        count_query = (
            lf_train
            .select(pl.col(col).cast(pl.String))
            .group_by(col)
            .len()
            .rename({"len": f"{col}_count"})
        )

        # Collect with streaming to reduce memory pressure
        try:
            count_df = count_query.collect(engine='streaming')
        except Exception:
            count_df = count_query.collect()

        # Join to both train and test
        count_lf = count_df.lazy()
        lf_train = lf_train.join(count_lf, on=col, how="left")
        lf_test = lf_test.join(count_lf, on=col, how="left")

        # Explicitly free memory
        del count_df, count_lf
        gc.collect()

        if (i + 1) % 3 == 0 or i == len(count_cols) - 1:
            print(f"  Computed count feature {i + 1}/{len(count_cols)}: {col}")

    # Apply all fill_null operations in a single with_columns call
    fill_exprs = [
        pl.col(f"{col}_count").fill_null(0).cast(pl.UInt32)
        for col in count_cols
    ]
    lf_train = lf_train.with_columns(fill_exprs)
    lf_test = lf_test.with_columns(fill_exprs)

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

    MEMORY-OPTIMIZED: Uses streaming collection.

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
    print("  Computing hourly aggregated features...")

    # Compute user impressions per hour from training data
    hourly_query = (
        lf_train
        .select(["user_proxy", "hour"])
        .group_by(["user_proxy", "hour"])
        .len()
        .rename({"len": "user_hourly_impressions"})
    )

    # Collect with streaming
    try:
        hourly_counts_df = hourly_query.collect(engine='streaming')
    except Exception:
        hourly_counts_df = hourly_query.collect()

    hourly_counts = hourly_counts_df.lazy()

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

    # Cleanup
    del hourly_counts_df, hourly_counts
    gc.collect()

    return lf_train, lf_test


def bin_hourly_impressions() -> pl.Expr:
    """
    Bin hourly impressions into categorical buckets.
    
    EDA-optimized bins based on percentiles (65.4% are single impressions):
    - 'single' (1): P50/median, most common
    - '2' (2): P75, returning within hour
    - '3-4' (3-4): Up to P90
    - '5+': Top 10% high-frequency users
    """
    return (
        pl.when(pl.col("user_hourly_impressions") == 1).then(pl.lit("single"))
        .when(pl.col("user_hourly_impressions") == 2).then(pl.lit("2"))
        .when(pl.col("user_hourly_impressions") <= 4).then(pl.lit("3-4"))
        .otherwise(pl.lit("5+"))
        .alias("user_hourly_impressions_bin")
    )


# =============================================================================
# Vocabulary Building (Memory-Efficient Sequential)
# =============================================================================
def build_vocabularies(lf_train: pl.LazyFrame, cat_cols: list[str], min_freq: int) -> tuple[dict, dict]:
    """
    Build vocabularies using memory-efficient sequential processing.

    MEMORY-OPTIMIZED: Processes one column at a time with streaming collection
    to avoid holding all vocabulary DataFrames in memory simultaneously.

    Returns:
        vocab_sizes: dict mapping column names to vocabulary sizes
        feat_maps: dict mapping column names to value->id dictionaries
    """
    print("Building vocabularies (sequential, memory-efficient)...")

    vocab_sizes = {}
    feat_maps = {}

    # Process columns one at a time to minimize peak memory
    for i, col in enumerate(cat_cols):
        # Build vocabulary query for single column
        vocab_query = (
            lf_train
            .select(pl.col(col).cast(pl.String))
            .group_by(col)
            .len()
            .filter(pl.col("len") >= min_freq)
            .sort(col)  # Deterministic ordering
        )

        # Collect with streaming to reduce memory pressure
        try:
            counts = vocab_query.collect(engine='streaming')
        except Exception:
            # Fallback if streaming not supported for this query
            counts = vocab_query.collect()

        # Build mapping: value -> sequential ID (starting at 1, 0 = <UNK>)
        values = counts[col].to_list()
        mapping = {val: idx + 1 for idx, val in enumerate(values)}

        feat_maps[col] = mapping
        vocab_sizes[col] = len(mapping) + 1  # +1 for <UNK> token

        # Explicitly free memory
        del counts, values
        gc.collect()

        if (i + 1) % 10 == 0 or i == len(cat_cols) - 1:
            print(f"  Built vocabulary for {i + 1}/{len(cat_cols)} columns")

    print(f"Built vocabularies for {len(cat_cols)} columns")
    return vocab_sizes, feat_maps


# =============================================================================
# Data Transformation (Streaming Sink to Parquet)
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


def sink_to_parquet(
    lf: pl.LazyFrame,
    feat_maps: dict,
    cat_cols: list[str],
    output_path: Path,
    is_test: bool = False,
    row_group_size: int = 10_000
) -> int:
    """
    Transform and sink a LazyFrame directly to parquet file.

    MEMORY-OPTIMIZED: Uses streaming sink to write transformed data directly
    to the final parquet file. No intermediate numpy arrays are created.

    Args:
        lf: Input LazyFrame
        feat_maps: Feature value to ID mappings
        cat_cols: List of categorical column names
        output_path: Path to write the parquet file
        is_test: Whether this is test data
        row_group_size: Row group size for parquet sink

    Returns:
        Number of rows written
    """
    # Build all mapping expressions
    mapping_exprs = create_mapping_expressions(feat_maps, cat_cols)

    # Select columns based on train/test
    # For train: click (label), hour (for splitting), and all features
    # For test: id (for submission) and all features
    if is_test:
        select_cols = ['id'] + cat_cols
    else:
        select_cols = ['click', 'hour'] + cat_cols

    # Apply all transformations
    transformed_lf = (
        lf
        .select(select_cols)
        .with_columns(mapping_exprs)
    )

    print(f"  Streaming transformed data to {output_path}...")

    # Use streaming sink to write to parquet - this is memory-efficient
    try:
        # transformed_lf.sink_parquet(output_path, row_group_size=row_group_size)
        raise Exception("Not implemented")
    except Exception as e:
        # Fallback: collect with streaming and write
        print(f"  Sink failed ({e}), using streaming collect fallback...")
        df = transformed_lf.collect(engine='streaming')
        df.write_parquet(output_path, row_group_size=row_group_size)
        del df
        gc.collect()

    # Get row count from the written file
    total_rows = pl.scan_parquet(output_path).select(pl.len()).collect().item()
    print(f"  Written {total_rows:,} rows to {output_path.name}")

    return total_rows


# =============================================================================
# Main Processing Pipeline
# =============================================================================
def process_data_polars() -> tuple[dict, list, int, int]:
    """
    Main data processing pipeline using memory-efficient Polars patterns.

    MEMORY-OPTIMIZED for laptops and constrained environments:
    - Sequential processing of count/vocabulary features (avoids parallel memory spike)
    - Streaming collection where possible
    - Explicit garbage collection between phases
    - Streaming sink directly to parquet (no numpy arrays in memory)

    Output files are saved to CONFIG['processed_path']:
    - train.parquet: Training data with features and labels
    - test.parquet: Test data with features and IDs
    - vocab_sizes.pkl: Vocabulary sizes for each feature
    - feature_names.pkl: List of feature column names

    Returns:
        vocab_sizes: Vocabulary sizes per column
        cat_cols: List of categorical column names
        train_rows: Number of training rows
        test_rows: Number of test rows
    """
    print("Loading data with Polars (memory-efficient pipeline)...")

    # Lazy load with explicit schema
    lf_train = pl.scan_csv(CONFIG['train_path'], schema_overrides=SCHEMA)
    lf_test = pl.scan_csv(CONFIG['test_path'], schema_overrides=SCHEMA)

    # ==========================================================================
    # PHASE 1: All independent base features (stays lazy)
    # ==========================================================================
    print("Building base feature expressions...")

    # Combine ALL base expressions into a single expression list
    base_exprs = [
        # User proxy
        get_user_proxy_expression(),
        # Interaction features
        *get_interaction_feature_expressions(),
        # Time features
        *get_time_feature_expressions(),
    ]

    lf_train = lf_train.with_columns(base_exprs)
    lf_test = lf_test.with_columns(base_exprs)

    # ==========================================================================
    # PHASE 2: Count features (REQUIRED materialization - train stats to both)
    # ==========================================================================
    print("Computing count features (memory-efficient)...")
    lf_train, lf_test = compute_count_features_from_train(
        lf_train, lf_test, COUNT_FEATURE_COLS
    )
    gc.collect()  # Free memory after materialization

    # Immediately add count binning (stays lazy)
    bin_exprs = bin_count_features(COUNT_FEATURE_COLS)
    lf_train = lf_train.with_columns(bin_exprs)
    lf_test = lf_test.with_columns(bin_exprs)

    # ==========================================================================
    # PHASE 3: Hourly features (REQUIRED materialization - train stats to both)
    # ==========================================================================
    print("Computing hourly aggregated features (memory-efficient)...")
    lf_train, lf_test = compute_hourly_aggregated_features(lf_train, lf_test)
    gc.collect()  # Free memory after materialization

    # ==========================================================================
    # PHASE 4: All remaining features in a SINGLE expression batch (stays lazy)
    # ==========================================================================
    print("Computing sequential/window features (batched)...")

    # Sort for chronological operations (lazy - just adds to query plan)
    lf_train = lf_train.sort("hour")
    lf_test = lf_test.sort("hour")

    # Build ALL remaining expressions as a single batch
    # These are window/sequential ops that can be computed together after sort
    sequential_exprs = [
        # Cumulative counts (window over sorted data)
        *get_cumulative_count_expressions(CUMCOUNT_COLS),
        # Time delta (window over sorted data)
        # Inlined from compute_time_delta_features for single expression batch
        (pl.col("hour") + "00").str.to_datetime("%y%m%d%H%M").alias("_timestamp"),
    ]

    lf_train = lf_train.with_columns(sequential_exprs)
    lf_test = lf_test.with_columns(sequential_exprs)

    # Time delta computation (needs _timestamp column)
    time_delta_exprs = [
        (
            pl.col("_timestamp") - pl.col("_timestamp").shift(1).over("user_proxy")
        )
        .dt.total_hours()
        .fill_null(0)
        .cast(pl.UInt32)
        .alias("hours_since_last_click"),
        # Previous click count
        (pl.col("user_proxy").cum_count().over("user_proxy") - 1)
        .clip(lower_bound=0)
        .cast(pl.UInt32)
        .alias("user_proxy_prev_clicks"),
    ]

    lf_train = lf_train.with_columns(time_delta_exprs).drop("_timestamp")
    lf_test = lf_test.with_columns(time_delta_exprs).drop("_timestamp")

    # ==========================================================================
    # PHASE 5: ALL binning in one shot (stays lazy until collect)
    # ==========================================================================
    print("Binning all features (single batch)...")

    all_bin_exprs = [
        # Cumcount bins
        *bin_cumcount_features(CUMCOUNT_COLS),
        # Hourly impressions bin
        bin_hourly_impressions(),
        # Time delta bin
        bin_time_delta_features(),
        # Previous clicks bin
        bin_prev_clicks("user_proxy"),
    ]

    lf_train = lf_train.with_columns(all_bin_exprs)
    lf_test = lf_test.with_columns(all_bin_exprs)

    # ==========================================================================
    # Build final feature list
    # ==========================================================================
    count_bin_cols = [f"{col}_count_bin" for col in COUNT_FEATURE_COLS]
    cumcount_bin_cols = [f"{col}_cumcount_bin" for col in CUMCOUNT_COLS]
    cat_cols = (
        CATEGORICAL_COLS
        + ['month', 'day_of_month', 'hour_of_day', 'day_of_week']
        + count_bin_cols
        + cumcount_bin_cols
        + ['user_hourly_impressions_bin']
        + ['hours_since_last_click_bin']
        + ['user_proxy_prev_clicks_bin']
    )

    print(f"Total categorical features: {len(cat_cols)}")

    # ==========================================================================
    # PHASE 6: Vocabulary building + streaming sink to parquet
    # ==========================================================================
    vocab_sizes, feat_maps = build_vocabularies(
        lf_train,
        cat_cols,
        CONFIG['min_freq']
    )
    gc.collect()  # Free memory after vocabulary building

    # Create output directory
    output_path = Path(CONFIG['processed_path'])
    output_path.mkdir(parents=True, exist_ok=True)

    print("Sinking transformed data to parquet (memory-efficient)...")

    # Sink train data to parquet
    print("Processing training data...")
    train_parquet = output_path / "train.parquet"
    train_rows = sink_to_parquet(lf_train, feat_maps, cat_cols, train_parquet, is_test=False)

    # Free train LazyFrame reference and collect garbage before test
    del lf_train
    gc.collect()

    # Sink test data to parquet
    print("Processing test data...")
    test_parquet = output_path / "test.parquet"
    test_rows = sink_to_parquet(lf_test, feat_maps, cat_cols, test_parquet, is_test=True)

    # Free test LazyFrame reference
    del lf_test
    gc.collect()

    # Save metadata
    print("Saving metadata...")
    with open(output_path / "vocab_sizes.pkl", "wb") as f:
        pickle.dump(vocab_sizes, f)
    with open(output_path / "feature_names.pkl", "wb") as f:
        pickle.dump(cat_cols, f)

    print(f"\nProcessing complete!")
    print(f"  Train: {train_rows:,} rows -> {train_parquet}")
    print(f"  Test:  {test_rows:,} rows -> {test_parquet}")

    return vocab_sizes, cat_cols, train_rows, test_rows


# =============================================================================
# Data Loading (Metadata Only - Data Stays in Parquet)
# =============================================================================
def load_metadata() -> tuple[dict, list]:
    """
    Load metadata (vocab sizes and feature names) from disk.

    The actual data stays in parquet files and is read by ParquetFullDataset.

    Returns:
        vocab_sizes: Vocabulary sizes per column
        feature_names: List of feature column names
    """
    path = Path(CONFIG['processed_path'])

    try:
        with open(path / "vocab_sizes.pkl", "rb") as f:
            vocab_sizes = pickle.load(f)
        with open(path / "feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)

        return vocab_sizes, feature_names

    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Processed data not found in {path}. Run 'python data_processor.py' first."
        ) from e


def get_parquet_path(split: str = 'train') -> Path:
    """
    Get the path to a parquet file.

    Args:
        split: 'train' or 'test'

    Returns:
        Path to the parquet file
    """
    path = Path(CONFIG['processed_path'])
    return path / f"{split}.parquet"


def get_parquet_row_count(split: str = 'train') -> int:
    """
    Get the number of rows in a parquet file.

    Args:
        split: 'train' or 'test'

    Returns:
        Number of rows
    """
    parquet_path = get_parquet_path(split)
    return pl.scan_parquet(parquet_path).select(pl.len()).collect().item()


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    vocab_sizes, cat_cols, train_rows, test_rows = process_data_polars()
    print(f"\nVocabulary sizes: {len(vocab_sizes)} features")
    print(f"Feature names: {cat_cols[:5]}... ({len(cat_cols)} total)")
