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
) -> pl.LazyFrame:
    """Apply pre-computed stats via lazy joins (fully streamable)."""
    # Join hourly
    lf = lf.join(hourly_lf, on=["user_proxy", "hour"], how="left")

    # Join counts
    for col in count_cols:
        lf = lf.join(count_lfs[col], on=col, how="left")

    # Fill nulls
    fill_exprs = [pl.col("user_hourly_impressions").fill_null(1)] + [
        pl.col(f"{c}_count").fill_null(0) for c in count_cols
    ]
    lf = lf.with_columns(fill_exprs)

    # Binning
    all_bin_exprs = [
        *bin_count_features(count_cols),
        *bin_cumcount_features(CUMCOUNT_COLS),
        bin_hourly_impressions(),
        bin_time_delta_features(),
        bin_prev_clicks("user_proxy"),
    ]
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
    final_cat_cols = (
        CATEGORICAL_COLS
        + ["month", "day_of_month", "hour_of_day", "day_of_week"]
        + count_bin_cols
        + cumcount_bin_cols
        + ["user_hourly_impressions_bin"]
        + ["hours_since_last_click_bin"]
        + ["user_proxy_prev_clicks_bin"]
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

    # Collect counts & hourly (no vocab yet)
    hourly_lf, count_lfs, _ = collect_stats_pass(
        lf_foundation_scan,
        COUNT_FEATURE_COLS,
        [],
        CONFIG["min_freq"],
    )

    # Create temp LF for vocab collection
    lf_temp_vocab = apply_transforms_lazy(
        pl.scan_parquet(temp_base_path),
        hourly_lf,
        count_lfs,
        {},
        COUNT_FEATURE_COLS,
        [],
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
        f"  Collected stats for {len(count_lfs)} count features and {len(vocab_lfs)} vocabularies"
    )

    # --- STAGE 3: Streaming Sink ---
    print("\n--- STAGE 3: Streaming Sink ---")

    lf_final = apply_transforms_lazy(
        pl.scan_parquet(temp_base_path),
        hourly_lf,
        count_lfs,
        vocab_lfs,
        COUNT_FEATURE_COLS,
        final_cat_cols,
    )

    train_cols = ["click", "hour"] + final_cat_cols
    test_cols = ["id"] + final_cat_cols

    train_parquet = output_path / "train.parquet"
    test_parquet = output_path / "test.parquet"

    print(f"  Sinking Train to {train_parquet}...")
    lf_final.filter(pl.col("_source") == "train").select(train_cols).sink_parquet(
        train_parquet
    )

    print(f"  Sinking Test to {test_parquet}...")
    lf_final.filter(pl.col("_source") == "test").select(test_cols).sink_parquet(
        test_parquet
    )

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

    train_rows = pl.scan_parquet(train_parquet).select(pl.len()).collect().item()
    test_rows = pl.scan_parquet(test_parquet).select(pl.len()).collect().item()

    print(f"\nProcessing complete!")
    print(f"  Train: {train_rows:,} rows -> {train_parquet}")
    print(f"  Test:  {test_rows:,} rows -> {test_parquet}")

    return vocab_sizes, final_cat_cols, train_rows, test_rows


# --- Data Loading (Metadata Only) ---
def load_metadata() -> tuple[dict, list]:
    """Load vocab sizes and feature names from disk."""
    path = Path(CONFIG["processed_path"])

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
