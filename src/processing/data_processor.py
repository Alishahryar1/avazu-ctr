"""
Avazu CTR Data Processor - Full Lazy Polars Implementation

This module uses FULL LAZY evaluation until the final sink:
- All feature engineering stays lazy (no intermediate collect)
- Statistics (counts, vocabularies) are collected once in a batched manner
- Joins use lazy frames throughout
- Only the final sink_parquet triggers computation
- Expression-based transformations (no Python loops in hot paths)
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
def get_time_delta_expressions() -> list[pl.Expr]:
    """
    Get expressions for time delta features - hours since last click for each user.
    
    This captures click velocity patterns:
    - Bots/fraudulent clicks have very short intervals
    - Normal users have longer intervals between clicks
    
    Returns:
        List of Polars expressions for time delta computation
    """
    return [
        # Parse hour column to datetime for time delta computation
        (pl.col("hour") + "00").str.to_datetime("%y%m%d%H%M").alias("_timestamp"),
    ]


def get_time_delta_window_expressions() -> list[pl.Expr]:
    """
    Get window expressions for time delta that require _timestamp column.
    
    Returns:
        List of expressions for hours_since_last_click
    """
    return [
        (
            pl.col("_timestamp") - pl.col("_timestamp").shift(1).over("user_proxy")
        )
        .dt.total_hours()
        .fill_null(0)
        .cast(pl.UInt32)
        .alias("hours_since_last_click"),
    ]


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
def get_prev_clicks_expression(group_col: str = "user_proxy") -> pl.Expr:
    """
    Get expression for number of previous clicks for each user.
    
    This is similar to cumulative count but shifted by 1, so it represents
    "how many clicks has this user made BEFORE this one".
    
    Args:
        group_col: Column to group by
        
    Returns:
        Polars expression for previous click count
    """
    return (
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
# Backward-Compatible Wrapper Functions (for tests)
# =============================================================================
def build_vocabularies(lf_train: pl.LazyFrame, cat_cols: list[str], min_freq: int) -> tuple[dict, dict]:
    """
    Build vocabularies using memory-efficient sequential processing.

    BACKWARD-COMPATIBLE WRAPPER: This function maintains the original API
    for existing tests while internally using the new lazy implementation.

    Returns:
        vocab_sizes: dict mapping column names to vocabulary sizes
        feat_maps: dict mapping column names to value->id dictionaries
    """
    print("Building vocabularies (sequential, memory-efficient)...")

    vocab_sizes = {}
    feat_maps = {}

    for i, col in enumerate(cat_cols):
        vocab_query = (
            lf_train
            .select(pl.col(col).cast(pl.String))
            .group_by(col)
            .len()
            .filter(pl.col("len") >= min_freq)
            .sort(col)
        )

        try:
            counts = vocab_query.collect(engine='streaming')
        except Exception:
            counts = vocab_query.collect()

        values = counts[col].to_list()
        mapping = {val: idx + 1 for idx, val in enumerate(values)}

        feat_maps[col] = mapping
        vocab_sizes[col] = len(mapping) + 1

        del counts, values
        gc.collect()

        if (i + 1) % 10 == 0 or i == len(cat_cols) - 1:
            print(f"  Built vocabulary for {i + 1}/{len(cat_cols)} columns")

    print(f"Built vocabularies for {len(cat_cols)} columns")
    return vocab_sizes, feat_maps


def compute_count_features_from_train(
    lf_train: pl.LazyFrame,
    lf_test: pl.LazyFrame,
    count_cols: list[str]
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Compute count/frequency features based on training data statistics.

    BACKWARD-COMPATIBLE WRAPPER: This function maintains the original API
    for existing tests while internally using lazy joins.

    Args:
        lf_train: Training LazyFrame
        lf_test: Test LazyFrame
        count_cols: Columns to compute counts for

    Returns:
        Updated train and test LazyFrames with count features
    """
    print(f"Computing count features for: {count_cols} (sequential, memory-efficient)")

    for i, col in enumerate(count_cols):
        count_query = (
            lf_train
            .select(pl.col(col).cast(pl.String))
            .group_by(col)
            .len()
            .rename({"len": f"{col}_count"})
        )

        try:
            count_df = count_query.collect(engine='streaming')
        except Exception:
            count_df = count_query.collect()

        count_lf = count_df.lazy()
        lf_train = lf_train.join(count_lf, on=col, how="left")
        lf_test = lf_test.join(count_lf, on=col, how="left")

        del count_df, count_lf
        gc.collect()

        if (i + 1) % 3 == 0 or i == len(count_cols) - 1:
            print(f"  Computed count feature {i + 1}/{len(count_cols)}: {col}")

    fill_exprs = [
        pl.col(f"{col}_count").fill_null(0).cast(pl.UInt32)
        for col in count_cols
    ]
    lf_train = lf_train.with_columns(fill_exprs)
    lf_test = lf_test.with_columns(fill_exprs)

    return lf_train, lf_test


def compute_hourly_aggregated_features(
    lf_train: pl.LazyFrame,
    lf_test: pl.LazyFrame
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Compute hourly aggregated features for user activity.

    BACKWARD-COMPATIBLE WRAPPER: This function maintains the original API
    for existing tests.

    Args:
        lf_train: Training LazyFrame
        lf_test: Test LazyFrame

    Returns:
        Updated train and test LazyFrames with hourly features
    """
    print("  Computing hourly aggregated features...")

    hourly_query = (
        lf_train
        .select(["user_proxy", "hour"])
        .group_by(["user_proxy", "hour"])
        .len()
        .rename({"len": "user_hourly_impressions"})
    )

    try:
        hourly_counts_df = hourly_query.collect(engine='streaming')
    except Exception:
        hourly_counts_df = hourly_query.collect()

    hourly_counts = hourly_counts_df.lazy()

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

    del hourly_counts_df, hourly_counts
    gc.collect()

    return lf_train, lf_test


def compute_time_delta_features(
    lf: pl.LazyFrame,
    group_col: str = "user_proxy"
) -> pl.LazyFrame:
    """
    Compute time delta features - hours since last click for each user.

    BACKWARD-COMPATIBLE WRAPPER: This function maintains the original API
    for existing tests.

    Args:
        lf: Input LazyFrame (must be sorted by 'hour')
        group_col: Column to group by for computing deltas (default: user_proxy)

    Returns:
        LazyFrame with time delta features added
    """
    lf = lf.with_columns(
        (pl.col("hour") + "00")
        .str.to_datetime("%y%m%d%H%M")
        .alias("_timestamp")
    )

    lf = lf.with_columns(
        (
            pl.col("_timestamp") - pl.col("_timestamp").shift(1).over(group_col)
        )
        .dt.total_hours()
        .fill_null(0)
        .cast(pl.UInt32)
        .alias("hours_since_last_click")
    )

    lf = lf.drop("_timestamp")

    return lf


def compute_previous_click_count(
    lf: pl.LazyFrame,
    group_col: str = "user_proxy"
) -> pl.LazyFrame:
    """
    Compute the number of previous clicks for each user up to (but not including) current row.

    BACKWARD-COMPATIBLE WRAPPER: This function maintains the original API
    for existing tests.

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


# =============================================================================
# Statistics Collection (Single Materialization Point)
# =============================================================================
def collect_all_statistics(
    lf_train: pl.LazyFrame,
    count_cols: list[str],
    cat_cols: list[str],
    min_freq: int
) -> tuple[dict[str, pl.LazyFrame], pl.LazyFrame, dict, dict]:
    """
    Collect ALL required statistics in a single batched operation.
    
    This is the ONLY materialization point before the final sink.
    Collects:
    - Count statistics for count_cols
    - Hourly aggregation statistics
    - Vocabularies for cat_cols
    
    Args:
        lf_train: Training LazyFrame (with all base features already added)
        count_cols: Columns to compute counts for
        cat_cols: Categorical columns for vocabulary building
        min_freq: Minimum frequency for vocabulary inclusion
        
    Returns:
        count_lfs: Dict of column -> LazyFrame with counts
        hourly_lf: LazyFrame with hourly aggregations
        vocab_sizes: Dict of column -> vocabulary size
        feat_maps: Dict of column -> value->id mapping
    """
    print("Collecting statistics (single materialization point)...")
    
    # =========================================================================
    # 1. Count features - batch collect all at once
    # =========================================================================
    print(f"  Computing count features for: {count_cols}")
    count_dfs = {}
    
    for col in count_cols:
        count_query = (
            lf_train
            .select(pl.col(col).cast(pl.String))
            .group_by(col)
            .len()
            .rename({"len": f"{col}_count"})
        )
        try:
            count_dfs[col] = count_query.collect(engine='streaming')
        except Exception:
            count_dfs[col] = count_query.collect()
    
    print(f"  Collected {len(count_dfs)} count statistics")
    
    # =========================================================================
    # 2. Hourly aggregation
    # =========================================================================
    print("  Computing hourly aggregated features...")
    hourly_query = (
        lf_train
        .select(["user_proxy", "hour"])
        .group_by(["user_proxy", "hour"])
        .len()
        .rename({"len": "user_hourly_impressions"})
    )
    
    try:
        hourly_df = hourly_query.collect(engine='streaming')
    except Exception:
        hourly_df = hourly_query.collect()
    
    print(f"  Collected hourly statistics")
    
    # =========================================================================
    # 3. Vocabularies - batch collect
    # =========================================================================
    print(f"  Building vocabularies for {len(cat_cols)} columns...")
    vocab_sizes = {}
    feat_maps = {}
    
    for i, col in enumerate(cat_cols):
        vocab_query = (
            lf_train
            .select(pl.col(col).cast(pl.String))
            .group_by(col)
            .len()
            .filter(pl.col("len") >= min_freq)
            .sort(col)
        )
        
        try:
            counts = vocab_query.collect(engine='streaming')
        except Exception:
            counts = vocab_query.collect()
        
        # Build mapping: value -> sequential ID (starting at 1, 0 = <UNK>)
        values = counts[col].to_list()
        mapping = {val: idx + 1 for idx, val in enumerate(values)}
        
        feat_maps[col] = mapping
        vocab_sizes[col] = len(mapping) + 1  # +1 for <UNK> token
        
        del counts, values
        
        if (i + 1) % 10 == 0 or i == len(cat_cols) - 1:
            print(f"    Built vocabulary for {i + 1}/{len(cat_cols)} columns")
    
    # Convert count DataFrames to LazyFrames for lazy joins
    count_lfs = {col: df.lazy() for col, df in count_dfs.items()}
    hourly_lf = hourly_df.lazy()
    
    # Cleanup eager DataFrames
    del count_dfs, hourly_df
    gc.collect()
    
    print("Statistics collection complete!")
    return count_lfs, hourly_lf, vocab_sizes, feat_maps


def apply_statistics_lazy(
    lf: pl.LazyFrame,
    count_lfs: dict[str, pl.LazyFrame],
    hourly_lf: pl.LazyFrame,
    count_cols: list[str]
) -> pl.LazyFrame:
    """
    Apply pre-computed statistics to a LazyFrame using lazy joins.
    
    All joins are lazy - no materialization happens here.
    
    Args:
        lf: Input LazyFrame
        count_lfs: Dict of column -> LazyFrame with counts
        hourly_lf: LazyFrame with hourly aggregations
        count_cols: Columns to join counts for
        
    Returns:
        LazyFrame with statistics joined
    """
    # Join count features (all lazy)
    for col in count_cols:
        lf = lf.join(count_lfs[col], on=col, how="left")
    
    # Fill nulls and cast counts
    fill_exprs = [
        pl.col(f"{col}_count").fill_null(0).cast(pl.UInt32)
        for col in count_cols
    ]
    lf = lf.with_columns(fill_exprs)
    
    # Join hourly features (lazy)
    lf = (
        lf
        .join(hourly_lf, on=["user_proxy", "hour"], how="left")
        .with_columns(
            pl.col("user_hourly_impressions").fill_null(1).cast(pl.UInt32)
        )
    )
    
    return lf


# =============================================================================
# Two-Pass Hybrid Pipeline Helpers
# =============================================================================
def get_lazy_vocab_map(lf: pl.LazyFrame, col: str, min_freq: int) -> pl.LazyFrame:
    """
    Creates a LazyFrame mapping for a specific column.
    
    Builds a vocabulary by filtering values with frequency >= min_freq,
    then assigns sequential IDs starting at 1 (0 is reserved for <UNK>).
    
    Args:
        lf: Training LazyFrame to build vocabulary from
        col: Column name to build vocabulary for
        min_freq: Minimum frequency threshold for inclusion
        
    Returns:
        LazyFrame with two columns: [col, '{col}_id'] where _id contains Int32 IDs
    """
    return (
        lf
        .group_by(col)
        .len()
        .filter(pl.col("len") >= min_freq)
        .select(pl.col(col))
        .sort(col)
        # Assign IDs starting at 1 (0 is reserved for <UNK>)
        .with_row_index(name=f"{col}_id", offset=1)
        .with_columns(pl.col(f"{col}_id").cast(pl.Int32))
    )


def apply_lazy_vocab(
    lf_data: pl.LazyFrame, 
    lf_vocab: pl.LazyFrame, 
    col: str
) -> pl.LazyFrame:
    """
    Joins the vocabulary LazyFrame to the data LazyFrame.
    
    Replaces the original string column with integer IDs via a lazy join.
    Unseen or low-frequency values get mapped to 0 (<UNK>).
    
    Args:
        lf_data: Data LazyFrame to apply vocabulary to
        lf_vocab: Vocabulary LazyFrame from get_lazy_vocab_map
        col: Column name being transformed
        
    Returns:
        LazyFrame with the original column replaced by integer IDs
    """
    return (
        lf_data
        .join(lf_vocab, on=col, how="left")
        .drop(col)  # Drop original string column
        .rename({f"{col}_id": col})  # Rename ID column to original name
        .with_columns(pl.col(col).fill_null(0))  # Fill <UNK> with 0
    )


def collect_stats_pass(
    lf: pl.LazyFrame, 
    count_cols: list[str], 
    cat_cols: list[str], 
    min_freq: int
) -> tuple[pl.LazyFrame, dict[str, pl.LazyFrame], dict[str, pl.LazyFrame]]:
    """
    Pass 1: Materialize only the statistics (Vocabs, Counts, Hourly).
    
    These are small tables (aggregated stats), so holding them in memory is fine.
    Returns them as LazyFrames backed by in-memory DataFrames to be joined in Pass 2.
    
    This breaks the circular dependency that prevents sink_parquet from working:
    - The sink needs vocab stats to process rows
    - Vocab stats need full scan of source data
    - By materializing stats first, the sink graph becomes streamable
    
    Args:
        lf: Training LazyFrame with base features
        count_cols: Columns to compute counts for
        cat_cols: Columns to build vocabularies for
        min_freq: Minimum frequency for vocabulary inclusion
        
    Returns:
        hourly_lf: LazyFrame with hourly aggregations
        count_lfs: Dict of column -> LazyFrame with counts
        vocab_lfs: Dict of column -> LazyFrame with vocab mappings
    """
    print("  [Pass 1] Collecting global statistics (Counts, Hourly, Vocabs)...")
    
    # 1. Hourly Impressions
    print("    Computing hourly stats...")
    hourly_df = (
        lf
        .group_by(["user_proxy", "hour"])
        .len()
        .rename({"len": "user_hourly_impressions"})
        .with_columns(pl.col("user_hourly_impressions").cast(pl.UInt32))
        .collect()  # Materialize
    )
    
    # 2. Count Features
    print(f"    Computing counts for {len(count_cols)} features...")
    count_lfs = {}
    for col in count_cols:
        count_df = (
            lf
            .group_by(col)
            .len()
            .rename({"len": f"{col}_count"})
            .with_columns(pl.col(f"{col}_count").cast(pl.UInt32))
            .collect()  # Materialize
        )
        count_lfs[col] = count_df.lazy()  # Convert back to Lazy for joining
    
    # 3. Vocabularies (if cat_cols provided)
    vocab_lfs = {}
    if cat_cols:
        print(f"    Building vocabularies for {len(cat_cols)} columns...")
        for i, col in enumerate(cat_cols):
            vocab_df = (
                lf
                .group_by(col)
                .len()
                .filter(pl.col("len") >= min_freq)
                .select(pl.col(col))
                .sort(col)
                .with_row_index(name=f"{col}_id", offset=1)  # ID 1..N
                .with_columns(pl.col(f"{col}_id").cast(pl.Int32))
                .collect()  # Materialize
            )
            vocab_lfs[col] = vocab_df.lazy()  # Convert back to Lazy
            
            if (i + 1) % 5 == 0 or i == len(cat_cols) - 1:
                print(f"      Built {i + 1}/{len(cat_cols)} vocabularies")
    
    return hourly_df.lazy(), count_lfs, vocab_lfs


def apply_transforms_lazy(
    lf: pl.LazyFrame,
    hourly_lf: pl.LazyFrame,
    count_lfs: dict[str, pl.LazyFrame],
    vocab_lfs: dict[str, pl.LazyFrame],
    count_cols: list[str],
    cat_cols: list[str]
) -> pl.LazyFrame:
    """
    Pass 2: Apply the joins using the pre-computed (static) stats.
    
    This graph is fully streamable because the RHS of joins are backed
    by in-memory DataFrames (materialized in Pass 1), not lazy scans.
    
    Args:
        lf: LazyFrame to transform (with base features)
        hourly_lf: Pre-computed hourly stats (from collect_stats_pass)
        count_lfs: Pre-computed count stats per column
        vocab_lfs: Pre-computed vocabulary mappings per column
        count_cols: List of count feature columns
        cat_cols: List of categorical columns for vocab mapping
        
    Returns:
        Transformed LazyFrame ready for sink
    """
    # 1. Join Hourly
    lf = lf.join(hourly_lf, on=["user_proxy", "hour"], how="left")
    
    # 2. Join Counts
    for col in count_cols:
        lf = lf.join(count_lfs[col], on=col, how="left")

    # 3. Fill Nulls (for counts/hourly - unseen values get 0/1)
    fill_exprs = [pl.col("user_hourly_impressions").fill_null(1)] + \
                 [pl.col(f"{c}_count").fill_null(0) for c in count_cols]
    lf = lf.with_columns(fill_exprs)

    # 4. Binning (Pure Expressions - stays lazy)
    all_bin_exprs = [
        *bin_count_features(count_cols),
        *bin_cumcount_features(CUMCOUNT_COLS),
        bin_hourly_impressions(),
        bin_time_delta_features(),
        bin_prev_clicks("user_proxy"),
    ]
    lf = lf.with_columns(all_bin_exprs)

    # 5. Apply Vocabularies (Join & Map)
    for col in cat_cols:
        vocab_lf = vocab_lfs[col]
        lf = (
            lf
            .join(vocab_lf, on=col, how="left")
            .drop(col)  # Drop original string
            .rename({f"{col}_id": col})  # Rename ID to original
            .with_columns(pl.col(col).fill_null(0))  # Fill UNK with 0
        )
        
    return lf

# =============================================================================
# Main Processing Pipeline (Two-Pass Hybrid)
# =============================================================================
def process_data_polars() -> tuple[dict, list, int, int]:
    """
    Main data processing pipeline using Two-Pass Hybrid approach.
    
    This approach breaks the circular dependency that prevents sink_parquet:
    - Pass 1: Materialize statistics (counts, hourly, vocabs) into small in-memory tables
    - Pass 2: Stream data through joins against static tables, then sink to parquet
    
    Phases:
    PASS 1 (Stats Materialization):
    1. Define base LazyFrame with all features
    2. Collect counts/hourly stats (small tables, safe to materialize)
    3. Apply stats, compute bins, then collect vocabularies
    
    PASS 2 (Streamable Sink):
    4. Fresh scan with base features
    5. Apply all transforms via joins against static (in-memory) stats
    6. Sink directly to parquet (fully streamable now)
    7. Recover metadata from written parquet
    
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
    print("Loading data with Polars (Two-Pass Hybrid Pipeline)...")

    # =========================================================================
    # Helper: Get base LazyFrame with all features (shared by both passes)
    # =========================================================================
    def get_base_lf(path: str) -> pl.LazyFrame:
        """Defines the lazy graph up to the window features."""
        lf = pl.scan_csv(path, schema_overrides=SCHEMA)
        
        # Base Expressions
        base_exprs = [
            get_user_proxy_expression(),
            *get_interaction_feature_expressions(),
            *get_time_feature_expressions(),
        ]
        lf = lf.with_columns(base_exprs).sort("hour")
        
        # Window Expressions
        lf = lf.with_columns(get_time_delta_expressions())
        window_exprs = [
            *get_cumulative_count_expressions(CUMCOUNT_COLS),
            *get_time_delta_window_expressions(),
            get_prev_clicks_expression("user_proxy"),
        ]
        return lf.with_columns(window_exprs).drop("_timestamp")

    # Define final categorical columns list
    count_bin_cols = [f"{col}_count_bin" for col in COUNT_FEATURE_COLS]
    cumcount_bin_cols = [f"{col}_cumcount_bin" for col in CUMCOUNT_COLS]
    final_cat_cols = (
        CATEGORICAL_COLS
        + ['month', 'day_of_month', 'hour_of_day', 'day_of_week']
        + count_bin_cols
        + cumcount_bin_cols
        + ['user_hourly_impressions_bin']
        + ['hours_since_last_click_bin']
        + ['user_proxy_prev_clicks_bin']
    )
    
    print(f"Total categorical features: {len(final_cat_cols)}")

    # =========================================================================
    # PASS 1: Collect Statistics (Materialize Metadata Only)
    # =========================================================================
    print("\n--- PASS 1: Statistics Collection ---")
    
    # Initialize base LazyFrame for train
    lf_train_base = get_base_lf(CONFIG['train_path'])
    
    # 1.1 Collect Counts & Hourly (no vocab yet)
    hourly_lf, count_lfs, _ = collect_stats_pass(
        lf_train_base, 
        COUNT_FEATURE_COLS, 
        [],  # No vocabs yet
        CONFIG['min_freq']
    )
    
    # 1.2 Create temp LF for vocab collection (needs counts/hourly/bins applied)
    lf_temp_vocab = apply_transforms_lazy(
        lf_train_base, hourly_lf, count_lfs, {},
        COUNT_FEATURE_COLS, []  # No vocab mapping yet
    )
    
    # 1.3 Collect Vocabularies from the binned features
    _, _, vocab_lfs = collect_stats_pass(
        lf_temp_vocab, 
        [],  # No counts needed (already done)
        final_cat_cols, 
        CONFIG['min_freq']
    )
    
    # Clean up temp graph
    del lf_temp_vocab, lf_train_base
    gc.collect()
    
    print(f"  Collected stats for {len(count_lfs)} count features and {len(vocab_lfs)} vocabularies")

    # =========================================================================
    # PASS 2: Final Transform & Sink (Streamable)
    # =========================================================================
    print("\n--- PASS 2: Final Transform & Sink ---")
    
    output_path = Path(CONFIG['processed_path'])
    output_path.mkdir(parents=True, exist_ok=True)

    # Fresh scans for Pass 2
    lf_train_final = apply_transforms_lazy(
        get_base_lf(CONFIG['train_path']), 
        hourly_lf, count_lfs, vocab_lfs,
        COUNT_FEATURE_COLS, final_cat_cols
    )
    
    lf_test_final = apply_transforms_lazy(
        get_base_lf(CONFIG['test_path']), 
        hourly_lf, count_lfs, vocab_lfs,
        COUNT_FEATURE_COLS, final_cat_cols
    )

    # Column selection
    train_cols = ['click', 'hour'] + final_cat_cols
    test_cols = ['id'] + final_cat_cols
    
    train_parquet = output_path / "train.parquet"
    test_parquet = output_path / "test.parquet"

    # SINK TRAIN (now streamable - RHS of joins are static)
    print(f"  Sinking Train to {train_parquet}...")
    lf_train_final.select(train_cols).sink_parquet(train_parquet)

    # SINK TEST
    print(f"  Sinking Test to {test_parquet}...")
    lf_test_final.select(test_cols).sink_parquet(test_parquet)

    # =========================================================================
    # Metadata Recovery
    # =========================================================================
    print("\n--- Metadata Recovery ---")
    
    # Calculate vocab sizes from the static vocab LFs we already have in memory
    vocab_sizes = {}
    for col, lf in vocab_lfs.items():
        # Get max ID from the materialized vocab (instant since it's in-memory)
        max_id = lf.select(pl.col(f"{col}_id").max()).collect().item()
        vocab_sizes[col] = (max_id if max_id is not None else 0) + 1

    # Save metadata
    print("  Saving metadata...")
    with open(output_path / "vocab_sizes.pkl", "wb") as f:
        pickle.dump(vocab_sizes, f)
    with open(output_path / "feature_names.pkl", "wb") as f:
        pickle.dump(final_cat_cols, f)

    # Get row counts
    train_rows = pl.scan_parquet(train_parquet).select(pl.len()).collect().item()
    test_rows = pl.scan_parquet(test_parquet).select(pl.len()).collect().item()

    print(f"\nProcessing complete!")
    print(f"  Train: {train_rows:,} rows -> {train_parquet}")
    print(f"  Test:  {test_rows:,} rows -> {test_parquet}")

    return vocab_sizes, final_cat_cols, train_rows, test_rows


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
