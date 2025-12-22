"""
Avazu CTR Data Processor - Full Lazy Polars Implementation

Uses lazy evaluation with a 3-stage pipeline:
1. Foundation materialization (sort + window functions)
2. Statistics collection (counts, vocabularies)
3. Streaming sink (joins + final output)
"""

import polars as pl
import gc
from pathlib import Path
from config import CONFIG
import pickle


# --- Schema Definition ---
SCHEMA = {
    "id": pl.String,
    "click": pl.UInt8,
    "hour": pl.String,
    "C1": pl.String,
    "banner_pos": pl.String,
    "site_id": pl.String,
    "site_domain": pl.String,
    "site_category": pl.String,
    "app_id": pl.String,
    "app_domain": pl.String,
    "app_category": pl.String,
    "device_id": pl.String,
    "device_ip": pl.String,
    "device_model": pl.String,
    "device_type": pl.String,
    "device_conn_type": pl.String,
    "C14": pl.String,
    "C15": pl.String,
    "C16": pl.String,
    "C17": pl.String,
    "C18": pl.String,
    "C19": pl.String,
    "C20": pl.String,
    "C21": pl.String,
}

BASE_CATEGORICAL_COLS = [
    "C1",
    "banner_pos",
    "site_id",
    "site_domain",
    "site_category",
    "app_id",
    "app_domain",
    "app_category",
    "device_id",
    "device_ip",
    "device_model",
    "device_type",
    "device_conn_type",
    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
    "C19",
    "C20",
    "C21",
]

ENGINEERED_CATEGORICAL_COLS = [
    "user_proxy",
    "device_id_x_app_id",
    "device_ip_x_C14",
]

CATEGORICAL_COLS = BASE_CATEGORICAL_COLS + ENGINEERED_CATEGORICAL_COLS

COUNT_FEATURE_COLS = ["device_ip", "device_id", "C14", "C17", "C21", "user_proxy"]
CUMCOUNT_COLS = ["device_ip", "device_id"]

# Counts Unique: (group_col, value_col, output_name) - computed from combined train+test
NUNIQUE_SPECS: list[tuple[str, str, str]] = [
    ("device_ip", "app_id", "device_ip_nunique_apps"),
    ("device_ip", "site_id", "device_ip_nunique_sites"),
    ("user_proxy", "app_id", "user_proxy_nunique_apps"),
    ("user_proxy", "site_id", "user_proxy_nunique_sites"),
]

# Likelihood features: columns to compute click probability for (train-only, test has no click)
LIKELIHOOD_COLS = ["app_id", "site_id", "site_domain", "app_domain", "C14", "C17"]

# Smoothing parameter for regularization (prevents overfitting on small counts)
LIKELIHOOD_SMOOTHING_WEIGHT = 20

# K-Fold configuration for target encoding (prevents target leakage)
LIKELIHOOD_KFOLD = 5

# Raw numerical features (kept alongside binned versions for numerical embeddings)
NUMERICAL_FEATURE_COLS = (
    # Likelihood features (6)
    [f"{col}_likelihood" for col in LIKELIHOOD_COLS]
    # Nunique features (4)
    + [name for _, _, name in NUNIQUE_SPECS]
    # Count features (6)
    + [f"{col}_count" for col in COUNT_FEATURE_COLS]
    # Cumcount features (2)
    + [f"{col}_cumcount" for col in CUMCOUNT_COLS]
    # Time delta (1)
    + ["hours_since_last_click"]
    # Prev clicks (1)
    + ["user_proxy_prev_clicks"]
)


# --- Feature Engineering Expressions ---
def get_time_feature_expressions() -> list[pl.Expr]:
    """Extract time features from 'hour' column (format: YYMMDDHH)."""
    return [
        pl.col("hour").str.slice(2, 2).cast(pl.UInt8).alias("month"),
        pl.col("hour").str.slice(4, 2).cast(pl.UInt8).alias("day_of_month"),
        pl.col("hour").str.slice(6, 2).cast(pl.UInt8).alias("hour_of_day"),
        (pl.col("hour") + "00")
        .str.to_datetime("%y%m%d%H%M")
        .dt.weekday()
        .cast(pl.UInt8)
        .alias("day_of_week"),
    ]


def get_user_proxy_expression() -> pl.Expr:
    """Create user proxy ID from device_ip + device_model."""
    return pl.concat_str(
        [pl.col("device_ip"), pl.col("device_model")], separator="_"
    ).alias("user_proxy")


def get_interaction_feature_expressions() -> list[pl.Expr]:
    """Create device_id x app_id and device_ip x C14 interaction features."""
    return [
        pl.concat_str([pl.col("device_id"), pl.col("app_id")], separator="_").alias(
            "device_id_x_app_id"
        ),
        pl.concat_str([pl.col("device_ip"), pl.col("C14")], separator="_").alias(
            "device_ip_x_C14"
        ),
    ]


def get_cumulative_count_expressions(cols: list[str]) -> list[pl.Expr]:
    """Create cumulative count expressions for tracking user/device history."""
    return [pl.col(col).cum_count().over(col).alias(f"{col}_cumcount") for col in cols]


def bin_cumcount_features(cols: list[str]) -> list[pl.Expr]:
    """Bin cumulative counts: first, 2-3, 4-10, 11-50, 51-100, 100+."""
    expressions = []
    for col in cols:
        cumcount_col = f"{col}_cumcount"
        binned_col = f"{col}_cumcount_bin"
        expr = (
            pl.when(pl.col(cumcount_col) == 1)
            .then(pl.lit("first"))
            .when(pl.col(cumcount_col) <= 3)
            .then(pl.lit("2-3"))
            .when(pl.col(cumcount_col) <= 10)
            .then(pl.lit("4-10"))
            .when(pl.col(cumcount_col) <= 50)
            .then(pl.lit("11-50"))
            .when(pl.col(cumcount_col) <= 100)
            .then(pl.lit("51-100"))
            .otherwise(pl.lit("100+"))
            .alias(binned_col)
        )
        expressions.append(expr)
    return expressions


# --- Time-Delta Features ---
def get_time_delta_expressions() -> list[pl.Expr]:
    """Parse hour column to datetime for time delta computation."""
    return [
        (pl.col("hour") + "00").str.to_datetime("%y%m%d%H%M").alias("_timestamp"),
    ]


def get_time_delta_window_expressions() -> list[pl.Expr]:
    """Compute hours since last click per user (requires _timestamp column)."""
    return [
        (pl.col("_timestamp") - pl.col("_timestamp").shift(1).over("user_proxy"))
        .dt.total_hours()
        .fill_null(0)
        .cast(pl.UInt32)
        .alias("hours_since_last_click"),
    ]


def bin_time_delta_features() -> pl.Expr:
    """Bin hours_since_last_click: first, 1-4h, 5-19h, 20-53h, >53h."""
    return (
        pl.when(pl.col("hours_since_last_click") == 0)
        .then(pl.lit("first"))
        .when(pl.col("hours_since_last_click") <= 4)
        .then(pl.lit("1-4h"))
        .when(pl.col("hours_since_last_click") <= 19)
        .then(pl.lit("5-19h"))
        .when(pl.col("hours_since_last_click") <= 53)
        .then(pl.lit("20-53h"))
        .otherwise(pl.lit(">53h"))
        .alias("hours_since_last_click_bin")
    )


# --- Previous Click Count Features ---
def get_prev_clicks_expression(group_col: str = "user_proxy") -> pl.Expr:
    """Get previous click count (cumcount - 1, clipped to 0)."""
    return (
        (pl.col(group_col).cum_count().over(group_col) - 1)
        .clip(lower_bound=0)
        .cast(pl.UInt32)
        .alias(f"{group_col}_prev_clicks")
    )


def bin_prev_clicks(group_col: str) -> pl.Expr:
    """Bin previous clicks: new, returning, regular, heavy, power."""
    col_name = f"{group_col}_prev_clicks"
    return (
        pl.when(pl.col(col_name) == 0)
        .then(pl.lit("new"))
        .when(pl.col(col_name) <= 7)
        .then(pl.lit("returning"))
        .when(pl.col(col_name) <= 32)
        .then(pl.lit("regular"))
        .when(pl.col(col_name) <= 224)
        .then(pl.lit("heavy"))
        .otherwise(pl.lit("power"))
        .alias(f"{col_name}_bin")
    )


def get_string_cast_expressions(columns: list[str]) -> list[pl.Expr]:
    """Cast columns to String type."""
    return [pl.col(c).cast(pl.String) for c in columns]


def bin_count_features(count_cols: list[str]) -> list[pl.Expr]:
    """Bin count features into categorical buckets."""
    expressions = []
    for col in count_cols:
        count_col = f"{col}_count"
        binned_col = f"{col}_count_bin"
        expr = (
            pl.when(pl.col(count_col) == 0)
            .then(pl.lit("0"))
            .when(pl.col(count_col) == 1)
            .then(pl.lit("1"))
            .when(pl.col(count_col) <= 5)
            .then(pl.lit("2-5"))
            .when(pl.col(count_col) <= 10)
            .then(pl.lit("6-10"))
            .when(pl.col(count_col) <= 50)
            .then(pl.lit("11-50"))
            .when(pl.col(count_col) <= 100)
            .then(pl.lit("51-100"))
            .when(pl.col(count_col) <= 500)
            .then(pl.lit("101-500"))
            .when(pl.col(count_col) <= 1000)
            .then(pl.lit("501-1000"))
            .otherwise(pl.lit("1000+"))
            .alias(binned_col)
        )
        expressions.append(expr)
    return expressions


def bin_hourly_impressions() -> pl.Expr:
    """Bin hourly impressions: single, 2, 3-4, 5+."""
    return (
        pl.when(pl.col("user_hourly_impressions") == 1)
        .then(pl.lit("single"))
        .when(pl.col("user_hourly_impressions") == 2)
        .then(pl.lit("2"))
        .when(pl.col("user_hourly_impressions") <= 4)
        .then(pl.lit("3-4"))
        .otherwise(pl.lit("5+"))
        .alias("user_hourly_impressions_bin")
    )


# --- Counts Unique (Nunique) Features ---
def collect_nunique_stats(
    lf: pl.LazyFrame, specs: list[tuple[str, str, str]]
) -> dict[str, tuple[pl.LazyFrame, str]]:
    """
    Collect nunique statistics for each (group_col, value_col) pair.

    Computes how many unique values of value_col exist for each group_col.
    E.g., how many unique apps has each device_ip visited?

    Returns dict mapping output_name -> (LazyFrame with [group_col, output_name], group_col).
    """
    nunique_lfs: dict[str, tuple[pl.LazyFrame, str]] = {}
    for group_col, value_col, output_name in specs:
        nunique_df = (
            lf.group_by(group_col)
            .agg(pl.col(value_col).n_unique().alias(output_name))
            .with_columns(pl.col(output_name).cast(pl.UInt32))
            .collect()
        )
        nunique_lfs[output_name] = (nunique_df.lazy(), group_col)
    return nunique_lfs


def bin_nunique_features(specs: list[tuple[str, str, str]]) -> list[pl.Expr]:
    """Bin nunique features: 1, 2, 3-5, 6-10, 11-50, 50+."""
    expressions = []
    for _, _, output_name in specs:
        binned_col = f"{output_name}_bin"
        expr = (
            pl.when(pl.col(output_name) == 1)
            .then(pl.lit("1"))
            .when(pl.col(output_name) == 2)
            .then(pl.lit("2"))
            .when(pl.col(output_name) <= 5)
            .then(pl.lit("3-5"))
            .when(pl.col(output_name) <= 10)
            .then(pl.lit("6-10"))
            .when(pl.col(output_name) <= 50)
            .then(pl.lit("11-50"))
            .otherwise(pl.lit("50+"))
            .alias(binned_col)
        )
        expressions.append(expr)
    return expressions


# --- Likelihood Features (Target Encoding) ---
def collect_likelihood_stats(
    lf_train: pl.LazyFrame,
    cols: list[str],
    smoothing_weight: int = 20,
) -> tuple[float, dict[str, pl.LazyFrame]]:
    """
    Compute target encoding (click probability) per category from TRAINING data only.

    Uses smoothed mean: (sum_clicks + global_mean * smoothing) / (count + smoothing)
    This regularizes toward the global mean for categories with few samples.

    Args:
        lf_train: LazyFrame of training data (must have 'click' column)
        cols: List of columns to compute likelihood for
        smoothing_weight: Higher = more regularization toward global mean

    Returns:
        global_mean: Global click rate (for unknown categories)
        likelihood_lfs: Dict mapping col -> LazyFrame with [col, {col}_likelihood]
    """
    # Compute global mean from training data
    global_stats = lf_train.select(
        [
            pl.col("click").sum().alias("total_clicks"),
            pl.len().alias("total_count"),
        ]
    ).collect()

    total_clicks = global_stats["total_clicks"][0]
    total_count = global_stats["total_count"][0]
    global_mean = float(total_clicks / total_count) if total_count > 0 else 0.5

    likelihood_lfs: dict[str, pl.LazyFrame] = {}
    for col in cols:
        # Compute per-category click stats with smoothing
        likelihood_df = (
            lf_train.group_by(col)
            .agg(
                [
                    pl.col("click").sum().alias("_click_sum"),
                    pl.len().alias("_count"),
                ]
            )
            .with_columns(
                [
                    # Smoothed click probability
                    (
                        (pl.col("_click_sum") + pl.lit(global_mean) * smoothing_weight)
                        / (pl.col("_count") + smoothing_weight)
                    ).alias(f"{col}_likelihood")
                ]
            )
            .select([col, f"{col}_likelihood"])
            .collect()
        )
        likelihood_lfs[col] = likelihood_df.lazy()

    return global_mean, likelihood_lfs


def collect_kfold_likelihood_stats(
    lf_train: pl.LazyFrame,
    cols: list[str],
    k: int = 5,
    smoothing_weight: int = 20,
    temp_dir: Path | None = None,
) -> tuple[float, dict[str, pl.LazyFrame], Path]:
    """
    Compute K-Fold target encoding to prevent target leakage.

    Uses disk-based streaming to avoid OOM on large datasets:
    1. Write training data with fold IDs to temp parquet
    2. For each fold, compute stats from other folds lazily
    3. Stream-join and sink each fold to temp parquet files
    4. Return path to combined result

    Args:
        lf_train: Training data LazyFrame (must have 'click' column)
        cols: Columns to compute likelihood for
        k: Number of folds
        smoothing_weight: Regularization weight
        temp_dir: Directory for temp files (defaults to CONFIG temp path)

    Returns:
        global_mean: Global click rate (for test set)
        global_likelihood_lfs: Dict of global likelihoods (for test set)
        train_with_likelihoods_path: Path to parquet with fold-specific likelihoods
    """
    print(f"      Computing K-Fold target encoding (K={k}, disk-based)...")

    if temp_dir is None:
        temp_dir = Path(CONFIG["processed_path"]) / "_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Compute global mean and likelihoods (for test set)
    global_mean, global_likelihood_lfs = collect_likelihood_stats(
        lf_train, cols, smoothing_weight
    )

    # Step 2: Add fold IDs lazily using row_index modulo k
    # Write to temp parquet with fold IDs
    train_with_folds_path = temp_dir / "_train_with_folds.parquet"
    lf_train.with_row_index("_row_idx").with_columns(
        (pl.col("_row_idx") % k).cast(pl.UInt8).alias("_fold_id")
    ).drop("_row_idx").sink_parquet(train_with_folds_path)
    print(f"        Written train with fold IDs to {train_with_folds_path}")

    # Step 3: For each fold, compute likelihoods from OTHER folds and sink
    fold_paths: list[Path] = []
    for fold in range(k):
        fold_path = temp_dir / f"_fold_{fold}_with_likelihoods.parquet"
        fold_paths.append(fold_path)

        # Scan train data for stats computation (exclude current fold)
        lf_other_folds = pl.scan_parquet(train_with_folds_path).filter(
            pl.col("_fold_id") != fold
        )

        # Compute likelihoods from other folds
        _, fold_likelihood_lfs = collect_likelihood_stats(
            lf_other_folds, cols, smoothing_weight
        )

        # Scan current fold data
        lf_current = pl.scan_parquet(train_with_folds_path).filter(
            pl.col("_fold_id") == fold
        )

        # Apply fold-specific likelihoods via lazy joins
        for col in cols:
            likelihood_lf = fold_likelihood_lfs[col]
            lf_current = lf_current.join(likelihood_lf, on=col, how="left")
            lf_current = lf_current.with_columns(
                pl.col(f"{col}_likelihood").fill_null(global_mean)
            )

        # Sink this fold's result
        lf_current.drop("_fold_id").sink_parquet(fold_path)
        print(f"        Fold {fold + 1}/{k} done -> {fold_path.name}")

        # Cleanup fold likelihood dfs
        del fold_likelihood_lfs
        gc.collect()

    # Step 4: Combine all fold parquets into one (streaming)
    train_with_likelihoods_path = temp_dir / "_train_kfold_likelihoods.parquet"
    pl.scan_parquet(fold_paths).sink_parquet(train_with_likelihoods_path)
    print(f"        Combined folds -> {train_with_likelihoods_path}")

    # Cleanup temp fold files
    for fold_path in fold_paths:
        fold_path.unlink(missing_ok=True)
    train_with_folds_path.unlink(missing_ok=True)

    print("      K-Fold target encoding complete")
    return global_mean, global_likelihood_lfs, train_with_likelihoods_path


def bin_likelihood_features(cols: list[str]) -> list[pl.Expr]:
    """Bin likelihood (click probability) into buckets based on CTR ranges."""
    expressions = []
    for col in cols:
        likelihood_col = f"{col}_likelihood"
        binned_col = f"{col}_likelihood_bin"
        expr = (
            pl.when(pl.col(likelihood_col) < 0.05)
            .then(pl.lit("very_low"))
            .when(pl.col(likelihood_col) < 0.15)
            .then(pl.lit("low"))
            .when(pl.col(likelihood_col) < 0.25)
            .then(pl.lit("medium"))
            .when(pl.col(likelihood_col) < 0.40)
            .then(pl.lit("high"))
            .otherwise(pl.lit("very_high"))
            .alias(binned_col)
        )
        expressions.append(expr)
    return expressions


# --- Lazy Vocabulary Mapping ---
def get_lazy_vocab_map(lf: pl.LazyFrame, col: str, min_freq: int) -> pl.LazyFrame:
    """Create vocabulary mapping LazyFrame with IDs starting at 1 (0 = UNK)."""
    return (
        lf.group_by(col)
        .len()
        .filter(pl.col("len") >= min_freq)
        .select(pl.col(col))
        .sort(col)
        .with_row_index(name=f"{col}_id", offset=1)
        .with_columns(pl.col(f"{col}_id").cast(pl.Int32))
    )


def apply_lazy_vocab(
    lf_data: pl.LazyFrame, lf_vocab: pl.LazyFrame, col: str
) -> pl.LazyFrame:
    """Apply vocabulary mapping via lazy join, filling unknown with 0."""
    return (
        lf_data.join(lf_vocab, on=col, how="left")
        .drop(col)
        .rename({f"{col}_id": col})
        .with_columns(pl.col(col).fill_null(0))
    )


# --- Statistics Collection (Pipeline Stage 2) ---
def collect_stats_pass(
    lf: pl.LazyFrame, count_cols: list[str], cat_cols: list[str], min_freq: int
) -> tuple[pl.LazyFrame, dict[str, pl.LazyFrame], dict[str, pl.LazyFrame]]:
    """
    Collect statistics from data: hourly impressions, counts, and vocabularies.

    Returns materialized stats as LazyFrames backed by in-memory DataFrames.
    """
    print("  [Pass 1] Collecting global statistics (Counts, Hourly, Vocabs)...")

    # Hourly impressions
    print("    Computing hourly stats...")
    hourly_df = (
        lf.group_by(["user_proxy", "hour"])
        .len()
        .rename({"len": "user_hourly_impressions"})
        .with_columns(pl.col("user_hourly_impressions").cast(pl.UInt32))
        .collect()
    )

    # Count features
    print(f"    Computing counts for {len(count_cols)} features...")
    count_lfs = {}
    for col in count_cols:
        count_df = (
            lf.group_by(col)
            .len()
            .rename({"len": f"{col}_count"})
            .with_columns(pl.col(f"{col}_count").cast(pl.UInt32))
            .collect()
        )
        count_lfs[col] = count_df.lazy()

    # Vocabularies
    vocab_lfs = {}
    if cat_cols:
        print(f"    Building vocabularies for {len(cat_cols)} columns...")
        for i, col in enumerate(cat_cols):
            vocab_df = (
                lf.group_by(col)
                .len()
                .filter(pl.col("len") >= min_freq)
                .select(pl.col(col))
                .sort(col)
                .with_row_index(name=f"{col}_id", offset=1)
                .with_columns(pl.col(f"{col}_id").cast(pl.Int32))
                .collect()
            )
            vocab_lfs[col] = vocab_df.lazy()

            if (i + 1) % 5 == 0 or i == len(cat_cols) - 1:
                print(f"      Built {i + 1}/{len(cat_cols)} vocabularies")

    return hourly_df.lazy(), count_lfs, vocab_lfs


def apply_transforms_lazy(
    lf: pl.LazyFrame,
    hourly_lf: pl.LazyFrame,
    count_lfs: dict[str, pl.LazyFrame],
    vocab_lfs: dict[str, pl.LazyFrame],
    count_cols: list[str],
    cat_cols: list[str],
    nunique_lfs: dict[str, tuple[pl.LazyFrame, str]] | None = None,
    likelihood_lfs: dict[str, pl.LazyFrame] | None = None,
    global_mean: float = 0.17,
) -> pl.LazyFrame:
    """
    Apply pre-computed stats via lazy joins (fully streamable).

    Args:
        lf: Base LazyFrame to transform
        hourly_lf: Hourly impressions stats
        count_lfs: Count stats per column
        vocab_lfs: Vocabulary mappings per column
        count_cols: List of columns with count stats
        cat_cols: List of categorical columns for vocab mapping
        nunique_lfs: Dict of nunique stats (output_name -> (LazyFrame, group_col))
        likelihood_lfs: Dict of likelihood stats (col -> LazyFrame with likelihood).
                        If None but columns already exist (K-Fold case), binning is still applied.
        global_mean: Global click rate for filling unknown likelihood values
    """
    # Join hourly
    lf = lf.join(hourly_lf, on=["user_proxy", "hour"], how="left")

    # Join counts
    for col in count_cols:
        lf = lf.join(count_lfs[col], on=col, how="left")

    # Join nunique stats
    if nunique_lfs:
        for output_name, (nunique_lf, group_col) in nunique_lfs.items():
            lf = lf.join(nunique_lf, on=group_col, how="left")

    # Join likelihood stats (skip if None - columns may already exist from K-Fold)
    if likelihood_lfs:
        for col, likelihood_lf in likelihood_lfs.items():
            lf = lf.join(likelihood_lf, on=col, how="left")

    # Fill nulls for count/hourly features
    fill_exprs = [pl.col("user_hourly_impressions").fill_null(1)] + [
        pl.col(f"{c}_count").fill_null(0) for c in count_cols
    ]

    # Fill nulls for nunique features
    if nunique_lfs:
        fill_exprs.extend(
            [pl.col(output_name).fill_null(1) for output_name in nunique_lfs.keys()]
        )

    # Fill nulls for likelihood features (use global mean for unknown categories)
    if likelihood_lfs:
        fill_exprs.extend(
            [
                pl.col(f"{col}_likelihood").fill_null(global_mean)
                for col in likelihood_lfs.keys()
            ]
        )

    lf = lf.with_columns(fill_exprs)

    # Binning expressions
    all_bin_exprs = [
        *bin_count_features(count_cols),
        *bin_cumcount_features(CUMCOUNT_COLS),
        bin_hourly_impressions(),
        bin_time_delta_features(),
        bin_prev_clicks("user_proxy"),
    ]

    # Add nunique binning
    if nunique_lfs:
        all_bin_exprs.extend(bin_nunique_features(NUNIQUE_SPECS))

    # Add likelihood binning (always apply since columns exist from K-Fold or join)
    # Use LIKELIHOOD_COLS directly since binning is independent of how columns were created
    all_bin_exprs.extend(bin_likelihood_features(LIKELIHOOD_COLS))

    lf = lf.with_columns(all_bin_exprs)

    # Apply vocabularies
    for col in cat_cols:
        vocab_lf = vocab_lfs[col]
        lf = (
            lf.join(vocab_lf, on=col, how="left")
            .drop(col)
            .rename({f"{col}_id": col})
            .with_columns(pl.col(col).fill_null(0))
        )

    return lf


# --- Main Processing Pipeline (3-Stage) ---
def process_data_polars() -> tuple[dict, list, int, int]:
    """
    Main data processing pipeline using 3-stage materialization:
    1. Materialize foundation (sort + windows) to temp parquet
    2. Collect statistics from foundation
    3. Stream joins and sink to final parquet
    """
    print("Loading data with Polars (3-Stage Pipeline)...")
    print("NOTE: Statistics computed from combined train+test for maximum coverage")

    output_path = Path(CONFIG["processed_path"])
    output_path.mkdir(parents=True, exist_ok=True)
    temp_base_path = output_path / "base_sorted_windowed.parquet"

    # Define final categorical columns
    count_bin_cols = [f"{col}_count_bin" for col in COUNT_FEATURE_COLS]
    cumcount_bin_cols = [f"{col}_cumcount_bin" for col in CUMCOUNT_COLS]
    nunique_bin_cols = [f"{name}_bin" for _, _, name in NUNIQUE_SPECS]
    likelihood_bin_cols = [f"{col}_likelihood_bin" for col in LIKELIHOOD_COLS]
    final_cat_cols = (
        CATEGORICAL_COLS
        + ["month", "day_of_month", "hour_of_day", "day_of_week"]
        + count_bin_cols
        + cumcount_bin_cols
        + ["user_hourly_impressions_bin"]
        + ["hours_since_last_click_bin"]
        + ["user_proxy_prev_clicks_bin"]
        + nunique_bin_cols
        + likelihood_bin_cols
    )

    print(f"Total categorical features: {len(final_cat_cols)}")

    # --- STAGE 1: Materialize Foundation ---
    if not temp_base_path.exists():
        print("\n--- STAGE 1: Materializing Foundation (Sort + Windows) ---")

        lf_train = pl.scan_csv(
            CONFIG["train_path"], schema_overrides=SCHEMA
        ).with_columns(pl.lit("train").alias("_source"))
        lf_test = pl.scan_csv(
            CONFIG["test_path"], schema_overrides=SCHEMA
        ).with_columns(pl.lit("test").alias("_source"))

        lf_combined = pl.concat([lf_train, lf_test], how="diagonal")

        lf_foundation = (
            lf_combined.with_columns(
                [
                    get_user_proxy_expression(),
                    *get_interaction_feature_expressions(),
                    *get_time_feature_expressions(),
                ]
            )
            .sort(["hour"])
            .with_columns(get_time_delta_expressions())
            .with_columns(
                [
                    *get_cumulative_count_expressions(CUMCOUNT_COLS),
                    *get_time_delta_window_expressions(),
                    get_prev_clicks_expression("user_proxy"),
                ]
            )
            .drop("_timestamp")
            .sort(CONFIG["data_processor_sort_keys"])
        )

        print("  Collecting and writing foundation to parquet...")
        lf_foundation.collect().write_parquet(temp_base_path)
        print(f"  Foundation materialized to {temp_base_path}")

        del lf_foundation, lf_combined, lf_train, lf_test
        gc.collect()
    else:
        print(f"\n--- STAGE 1: Foundation already exists at {temp_base_path} ---")

    # --- STAGE 2: Collect Statistics ---
    print("\n--- STAGE 2: Statistics Collection ---")

    lf_foundation_scan = pl.scan_parquet(temp_base_path)

    # Collect counts & hourly from combined data (no vocab yet)
    print("  [2a] Collecting counts & hourly stats...")
    hourly_lf, count_lfs, _ = collect_stats_pass(
        lf_foundation_scan,
        COUNT_FEATURE_COLS,
        [],
        CONFIG["min_freq"],
    )

    # Collect nunique stats from combined data
    print(f"  [2b] Collecting nunique stats for {len(NUNIQUE_SPECS)} specs...")
    nunique_lfs = collect_nunique_stats(lf_foundation_scan, NUNIQUE_SPECS)

    # Collect likelihood stats using K-Fold for training data (prevents leakage)
    print(
        f"  [2c] Collecting K-Fold likelihood stats for {len(LIKELIHOOD_COLS)} columns..."
    )
    lf_train_only = lf_foundation_scan.filter(pl.col("_source") == "train")
    global_mean, global_likelihood_lfs, train_with_likelihoods_path = (
        collect_kfold_likelihood_stats(
            lf_train_only,
            LIKELIHOOD_COLS,
            LIKELIHOOD_KFOLD,
            LIKELIHOOD_SMOOTHING_WEIGHT,
        )
    )
    print(f"      Global click rate: {global_mean:.4f}")

    # Create temp LF for vocab collection (with all new features)
    # Use global likelihoods here since we just need category coverage
    print("  [2d] Building vocabularies...")
    lf_temp_vocab = apply_transforms_lazy(
        pl.scan_parquet(temp_base_path),
        hourly_lf,
        count_lfs,
        {},
        COUNT_FEATURE_COLS,
        [],
        nunique_lfs=nunique_lfs,
        likelihood_lfs=global_likelihood_lfs,
        global_mean=global_mean,
    )

    # Collect vocabularies from binned features
    _, _, vocab_lfs = collect_stats_pass(
        lf_temp_vocab,
        [],
        final_cat_cols,
        CONFIG["min_freq"],
    )

    del lf_temp_vocab
    gc.collect()

    print(
        f"  Collected stats: {len(count_lfs)} counts, {len(nunique_lfs)} nuniques, "
        f"{len(global_likelihood_lfs)} likelihoods, {len(vocab_lfs)} vocabularies"
    )

    # --- STAGE 3: Streaming Sink ---
    print("\n--- STAGE 3: Streaming Sink ---")

    # For TRAINING: Use pre-computed K-Fold likelihoods (from parquet file)
    # Apply other transforms (counts, nuniques, binning, vocab) to training data
    print("  [3a] Processing training data with K-Fold likelihoods...")
    lf_train_final = apply_transforms_lazy(
        pl.scan_parquet(train_with_likelihoods_path),
        hourly_lf,
        count_lfs,
        vocab_lfs,
        COUNT_FEATURE_COLS,
        final_cat_cols,
        nunique_lfs=nunique_lfs,
        likelihood_lfs=None,  # Skip likelihood joins (already have them)
        global_mean=global_mean,
    )

    # For TEST: Apply global likelihoods via join
    print("  [3b] Processing test data with global likelihoods...")
    lf_test_final = apply_transforms_lazy(
        pl.scan_parquet(temp_base_path).filter(pl.col("_source") == "test"),
        hourly_lf,
        count_lfs,
        vocab_lfs,
        COUNT_FEATURE_COLS,
        final_cat_cols,
        nunique_lfs=nunique_lfs,
        likelihood_lfs=global_likelihood_lfs,
        global_mean=global_mean,
    )

    train_cols = ["click", "hour"] + final_cat_cols + list(NUMERICAL_FEATURE_COLS)
    test_cols = ["id"] + final_cat_cols + list(NUMERICAL_FEATURE_COLS)

    train_parquet = output_path / "train.parquet"
    test_parquet = output_path / "test.parquet"

    print(f"  Sinking Train to {train_parquet}...")
    lf_train_final.select(train_cols).sink_parquet(train_parquet)

    print(f"  Sinking Test to {test_parquet}...")
    lf_test_final.select(test_cols).sink_parquet(test_parquet)

    # --- Metadata Recovery ---
    print("\n--- Metadata Recovery ---")

    vocab_sizes = {}
    for col, lf in vocab_lfs.items():
        max_id = lf.select(pl.col(f"{col}_id").max()).collect().item()
        vocab_sizes[col] = (max_id if max_id is not None else 0) + 1

    print("  Saving metadata...")
    with open(output_path / "vocab_sizes.pkl", "wb") as f:
        pickle.dump(vocab_sizes, f)
    with open(output_path / "feature_names.pkl", "wb") as f:
        pickle.dump(final_cat_cols, f)
    with open(output_path / "numerical_feature_names.pkl", "wb") as f:
        pickle.dump(list(NUMERICAL_FEATURE_COLS), f)

    train_rows = pl.scan_parquet(train_parquet).select(pl.len()).collect().item()
    test_rows = pl.scan_parquet(test_parquet).select(pl.len()).collect().item()

    print(f"\nProcessing complete!")
    print(f"  Train: {train_rows:,} rows -> {train_parquet}")
    print(f"  Test:  {test_rows:,} rows -> {test_parquet}")
    print(f"  Categorical features: {len(final_cat_cols)}")
    print(f"  Numerical features: {len(NUMERICAL_FEATURE_COLS)}")

    return vocab_sizes, final_cat_cols, train_rows, test_rows


# --- Data Loading (Metadata Only) ---
def load_metadata() -> tuple[dict, list, list]:
    """
    Load vocab sizes, categorical feature names, and numerical feature names from disk.

    Returns:
        vocab_sizes: Dict mapping feature name to vocabulary size
        feature_names: List of categorical feature names
        numerical_feature_names: List of numerical feature names
    """
    path = Path(CONFIG["processed_path"])

    try:
        with open(path / "vocab_sizes.pkl", "rb") as f:
            vocab_sizes = pickle.load(f)
        with open(path / "feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)

        # Load numerical features (with backward compatibility)
        numerical_path = path / "numerical_feature_names.pkl"
        if numerical_path.exists():
            with open(numerical_path, "rb") as f:
                numerical_feature_names = pickle.load(f)
        else:
            numerical_feature_names = []

        return vocab_sizes, feature_names, numerical_feature_names

    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Processed data not found in {path}. Run 'python data_processor.py' first."
        ) from e


def get_parquet_path(split: str = "train") -> Path:
    """Get path to train or test parquet file."""
    path = Path(CONFIG["processed_path"])
    return path / f"{split}.parquet"


def get_parquet_row_count(split: str = "train") -> int:
    """Get row count of a parquet file."""
    parquet_path = get_parquet_path(split)
    return pl.scan_parquet(parquet_path).select(pl.len()).collect().item()


if __name__ == "__main__":
    vocab_sizes, cat_cols, train_rows, test_rows = process_data_polars()
    print(f"\nVocabulary sizes: {len(vocab_sizes)} features")
    print(f"Feature names: {cat_cols[:5]}... ({len(cat_cols)} total)")
