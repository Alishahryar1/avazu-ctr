"""
Avazu CTR EDA - Raw Data Analysis

Analyzes raw training CSV to suggest optimal parameters for data_processor.py:
- Count feature binning boundaries
- Cumulative count binning boundaries
- Time delta binning boundaries  
- Previous clicks binning boundaries
- Hourly impressions binning boundaries

Uses Polars lazy evaluation for memory efficiency.

Run from project root: python -m misc.eda_raw
"""

import polars as pl
import numpy as np
from pathlib import Path
import gc

from src.config.config import CONFIG
from src.processing.data_processor import (
    SCHEMA, BASE_CATEGORICAL_COLS, CATEGORICAL_COLS,
    get_user_proxy_expression, get_interaction_feature_expressions,
    get_time_feature_expressions, COUNT_FEATURE_COLS, CUMCOUNT_COLS,
)


# =============================================================================
# Utility Functions
# =============================================================================

def round_to_nice(x: float) -> int:
    """Round a number to a 'nice' boundary for binning."""
    if x < 5:
        return max(1, int(x))
    elif x < 10:
        return int(round(x))
    elif x < 100:
        return int(round(x / 5) * 5)
    elif x < 1000:
        return int(round(x / 10) * 10)
    elif x < 10000:
        return int(round(x / 100) * 100)
    else:
        return int(round(x / 1000) * 1000)


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(title)
    print('='*80)


def print_subsection(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n{'-'*60}")
    print(title)
    print('-'*60)


# =============================================================================
# Analysis Functions - All use lazy evaluation with batched collection
# =============================================================================

def analyze_count_features_batched(lf: pl.LazyFrame, count_cols: list[str]) -> dict[str, dict]:
    """
    Analyze count feature distributions in a single batched pass.
    
    Computes count statistics for all columns by:
    1. Computing counts per column (lazy)
    2. Joining all counts back (lazy) 
    3. Collecting all statistics in one pass
    """
    print_section("COUNT FEATURE ANALYSIS")
    print("Analyzing: " + ", ".join(count_cols))
    
    # Build count LazyFrames for each column
    count_lfs = {}
    for col in count_cols:
        count_lf = (
            lf
            .select(pl.col(col).cast(pl.String))
            .group_by(col)
            .len()
            .rename({"len": f"{col}_count"})
        )
        count_lfs[col] = count_lf
    
    # Collect count statistics for each in sequence (to avoid memory spike)
    all_stats = {}
    
    for col in count_cols:
        print(f"\n  Processing {col}_count...")
        
        # Join and collect for this column
        lf_with_count = lf.join(count_lfs[col], on=col, how="left")
        
        # Compute percentiles in a single collection
        count_col_name = f"{col}_count"
        stats_df = (
            lf_with_count
            .select([
                pl.col(count_col_name).min().alias("min"),
                pl.col(count_col_name).max().alias("max"),
                pl.col(count_col_name).mean().alias("mean"),
                pl.col(count_col_name).median().alias("median"),
                pl.col(count_col_name).quantile(0.25).alias("p25"),
                pl.col(count_col_name).quantile(0.50).alias("p50"),
                pl.col(count_col_name).quantile(0.75).alias("p75"),
                pl.col(count_col_name).quantile(0.90).alias("p90"),
                pl.col(count_col_name).quantile(0.95).alias("p95"),
                pl.col(count_col_name).quantile(0.99).alias("p99"),
            ])
            .collect()
        )
        
        all_stats[col] = {
            "min": int(stats_df["min"][0]),
            "max": int(stats_df["max"][0]),
            "mean": float(stats_df["mean"][0]),
            "median": float(stats_df["median"][0]),
            "p25": float(stats_df["p25"][0]),
            "p50": float(stats_df["p50"][0]),
            "p75": float(stats_df["p75"][0]),
            "p90": float(stats_df["p90"][0]),
            "p95": float(stats_df["p95"][0]),
            "p99": float(stats_df["p99"][0]),
        }
        
        # Print stats
        s = all_stats[col]
        print(f"    Range: {s['min']:,} - {s['max']:,}")
        print(f"    Mean: {s['mean']:,.1f}, Median: {s['median']:,.0f}")
        print(f"    P25: {s['p25']:,.0f}, P50: {s['p50']:,.0f}, P75: {s['p75']:,.0f}, P90: {s['p90']:,.0f}")
        
        gc.collect()
    
    # Suggest binning based on aggregated percentiles
    print_subsection("RECOMMENDED bin_count_features() boundaries")
    
    avg_p25 = np.mean([s["p25"] for s in all_stats.values()])
    avg_p50 = np.mean([s["p50"] for s in all_stats.values()])
    avg_p75 = np.mean([s["p75"] for s in all_stats.values()])
    avg_p90 = np.mean([s["p90"] for s in all_stats.values()])
    avg_p99 = np.mean([s["p99"] for s in all_stats.values()])
    
    p25 = round_to_nice(avg_p25) if avg_p25 > 2 else 5
    p50 = round_to_nice(avg_p50) if avg_p50 > 5 else 10
    p75 = round_to_nice(avg_p75)
    p90 = round_to_nice(avg_p90)
    p99 = round_to_nice(avg_p99)
    
    print(f"""
Aggregated percentiles across {len(count_cols)} count features:
  P25: {avg_p25:.0f} -> bin boundary: {p25}
  P50: {avg_p50:.0f} -> bin boundary: {p50}
  P75: {avg_p75:.0f} -> bin boundary: {p75}
  P90: {avg_p90:.0f} -> bin boundary: {p90}
  P99: {avg_p99:.0f} -> bin boundary: {p99}

Suggested bin_count_features() code:
    pl.when(pl.col(count_col) == 0).then(pl.lit("0"))
    .when(pl.col(count_col) == 1).then(pl.lit("1"))
    .when(pl.col(count_col) <= {p25}).then(pl.lit("2-{p25}"))
    .when(pl.col(count_col) <= {p50}).then(pl.lit("{p25+1}-{p50}"))
    .when(pl.col(count_col) <= {p75}).then(pl.lit("{p50+1}-{p75}"))
    .when(pl.col(count_col) <= {p90}).then(pl.lit("{p75+1}-{p90}"))
    .otherwise(pl.lit("{p90}+"))
""")
    
    return all_stats


def analyze_cumcount_features(lf: pl.LazyFrame, cumcount_cols: list[str]) -> dict[str, dict]:
    """
    Analyze cumulative count feature distributions.
    
    Must sort by hour first, then compute cumcount per group.
    """
    print_section("CUMULATIVE COUNT FEATURE ANALYSIS")
    print("Analyzing: " + ", ".join(cumcount_cols))
    
    # Sort by hour for proper temporal ordering
    lf_sorted = lf.sort("hour")
    
    all_stats = {}
    
    for col in cumcount_cols:
        print(f"\n  Processing {col}_cumcount...")
        
        cumcount_col = f"{col}_cumcount"
        
        # Compute cumcount and statistics in one pass
        stats_df = (
            lf_sorted
            .with_columns([
                pl.col(col).cum_count().over(col).alias(cumcount_col)
            ])
            .select([
                pl.col(cumcount_col).min().alias("min"),
                pl.col(cumcount_col).max().alias("max"),
                pl.col(cumcount_col).mean().alias("mean"),
                pl.col(cumcount_col).quantile(0.25).alias("p25"),
                pl.col(cumcount_col).quantile(0.50).alias("p50"),
                pl.col(cumcount_col).quantile(0.75).alias("p75"),
                pl.col(cumcount_col).quantile(0.90).alias("p90"),
                pl.col(cumcount_col).quantile(0.95).alias("p95"),
            ])
            .collect()
        )
        
        all_stats[col] = {
            "min": int(stats_df["min"][0]),
            "max": int(stats_df["max"][0]),
            "mean": float(stats_df["mean"][0]),
            "p25": float(stats_df["p25"][0]),
            "p50": float(stats_df["p50"][0]),
            "p75": float(stats_df["p75"][0]),
            "p90": float(stats_df["p90"][0]),
            "p95": float(stats_df["p95"][0]),
        }
        
        s = all_stats[col]
        print(f"    Range: {s['min']:,} - {s['max']:,}, Mean: {s['mean']:,.1f}")
        print(f"    P25: {s['p25']:,.0f}, P50: {s['p50']:,.0f}, P75: {s['p75']:,.0f}, P90: {s['p90']:,.0f}")
        
        gc.collect()
    
    # Suggest bins
    print_subsection("RECOMMENDED bin_cumcount_features() boundaries")
    
    avg_p50 = np.mean([s["p50"] for s in all_stats.values()])
    avg_p75 = np.mean([s["p75"] for s in all_stats.values()])
    avg_p90 = np.mean([s["p90"] for s in all_stats.values()])
    
    p50 = round_to_nice(avg_p50)
    p75 = round_to_nice(avg_p75)
    p90 = round_to_nice(avg_p90)
    
    print(f"""
Aggregated percentiles:
  P50: {avg_p50:.0f}, P75: {avg_p75:.0f}, P90: {avg_p90:.0f}

Suggested bin_cumcount_features() boundaries:
  - "first" (== 1): First occurrence
  - "2-3" (<= 3): Very early
  - "4-{p50}" (<= P50): Early
  - "{p50+1}-{p75}" (<= P75): Medium
  - "{p75+1}-{p90}" (<= P90): Frequent
  - "{p90}+": Very frequent
""")
    
    return all_stats


def analyze_time_delta(lf: pl.LazyFrame) -> dict:
    """
    Analyze hours_since_last_click distribution.
    """
    print_section("TIME DELTA ANALYSIS (hours_since_last_click)")
    
    # Add user_proxy and compute time delta
    lf_with_proxy = lf.with_columns(get_user_proxy_expression())
    lf_sorted = lf_with_proxy.sort("hour")
    
    # Compute time delta (hours since last click per user)
    # hour format: YYMMDDHH -> convert to hours since epoch
    lf_with_delta = lf_sorted.with_columns([
        (
            pl.col("hour") // 100  # YYMMDD
        ).alias("_date_part"),
        (
            pl.col("hour") % 100  # HH
        ).alias("_hour_part"),
    ]).with_columns([
        (
            (pl.col("_date_part") % 100) * 24 +  # Day * 24
            ((pl.col("_date_part") // 100) % 100) * 24 * 31 +  # Month * days * 24
            pl.col("_hour_part")
        ).alias("_abs_hour")
    ]).with_columns([
        (
            pl.col("_abs_hour") - 
            pl.col("_abs_hour").shift(1).over("user_proxy")
        ).fill_null(0).clip(0, None).alias("hours_since_last_click")
    ])
    
    # Collect statistics
    stats_df = (
        lf_with_delta
        .select([
            (pl.col("hours_since_last_click") == 0).sum().alias("zero_count"),
            pl.len().alias("total"),
            pl.col("hours_since_last_click").filter(pl.col("hours_since_last_click") > 0).min().alias("nz_min"),
            pl.col("hours_since_last_click").filter(pl.col("hours_since_last_click") > 0).max().alias("nz_max"),
            pl.col("hours_since_last_click").filter(pl.col("hours_since_last_click") > 0).mean().alias("nz_mean"),
            pl.col("hours_since_last_click").filter(pl.col("hours_since_last_click") > 0).quantile(0.50).alias("nz_p50"),
            pl.col("hours_since_last_click").filter(pl.col("hours_since_last_click") > 0).quantile(0.75).alias("nz_p75"),
            pl.col("hours_since_last_click").filter(pl.col("hours_since_last_click") > 0).quantile(0.90).alias("nz_p90"),
        ])
        .collect()
    )
    
    zero_count = int(stats_df["zero_count"][0])
    total = int(stats_df["total"][0])
    zero_pct = zero_count / total * 100
    
    stats = {
        "zero_count": zero_count,
        "zero_pct": zero_pct,
        "nz_min": int(stats_df["nz_min"][0]) if stats_df["nz_min"][0] else 0,
        "nz_max": int(stats_df["nz_max"][0]) if stats_df["nz_max"][0] else 0,
        "nz_mean": float(stats_df["nz_mean"][0]) if stats_df["nz_mean"][0] else 0,
        "nz_p50": float(stats_df["nz_p50"][0]) if stats_df["nz_p50"][0] else 0,
        "nz_p75": float(stats_df["nz_p75"][0]) if stats_df["nz_p75"][0] else 0,
        "nz_p90": float(stats_df["nz_p90"][0]) if stats_df["nz_p90"][0] else 0,
    }
    
    print(f"""
Distribution:
  First clicks (0 hours): {stats['zero_count']:,} ({stats['zero_pct']:.1f}%)
  Non-zero range: {stats['nz_min']} - {stats['nz_max']} hours
  Non-zero mean: {stats['nz_mean']:.1f} hours
  Non-zero P50: {stats['nz_p50']:.0f}h, P75: {stats['nz_p75']:.0f}h, P90: {stats['nz_p90']:.0f}h
""")
    
    p50 = round_to_nice(stats["nz_p50"])
    p75 = round_to_nice(stats["nz_p75"])
    p90 = round_to_nice(stats["nz_p90"])
    
    print_subsection("RECOMMENDED bin_time_delta_features() boundaries")
    print(f"""
Suggested boundaries:
  - "first" (== 0): First click
  - "1-{p50}h" (<= P50): Short interval
  - "{p50+1}-{p75}h" (<= P75): Medium interval
  - "{p75+1}-{p90}h" (<= P90): Long interval
  - "{p90}h+": Re-engagement
""")
    
    return stats


def analyze_prev_clicks(lf: pl.LazyFrame) -> dict:
    """
    Analyze user_proxy_prev_clicks distribution.
    """
    print_section("PREVIOUS CLICKS ANALYSIS")
    
    lf_with_proxy = lf.with_columns(get_user_proxy_expression())
    lf_sorted = lf_with_proxy.sort("hour")
    
    # Compute previous clicks (cumcount - 1, but use row_number - 1)
    lf_with_prev = lf_sorted.with_columns([
        (pl.col("user_proxy").cum_count().over("user_proxy") - 1).alias("prev_clicks")
    ])
    
    # Collect statistics
    stats_df = (
        lf_with_prev
        .select([
            (pl.col("prev_clicks") == 0).sum().alias("zero_count"),
            pl.len().alias("total"),
            pl.col("prev_clicks").filter(pl.col("prev_clicks") > 0).max().alias("nz_max"),
            pl.col("prev_clicks").filter(pl.col("prev_clicks") > 0).mean().alias("nz_mean"),
            pl.col("prev_clicks").filter(pl.col("prev_clicks") > 0).quantile(0.50).alias("nz_p50"),
            pl.col("prev_clicks").filter(pl.col("prev_clicks") > 0).quantile(0.75).alias("nz_p75"),
            pl.col("prev_clicks").filter(pl.col("prev_clicks") > 0).quantile(0.90).alias("nz_p90"),
            pl.col("prev_clicks").filter(pl.col("prev_clicks") > 0).quantile(0.99).alias("nz_p99"),
        ])
        .collect()
    )
    
    zero_count = int(stats_df["zero_count"][0])
    total = int(stats_df["total"][0])
    
    stats = {
        "zero_count": zero_count,
        "zero_pct": zero_count / total * 100,
        "nz_max": int(stats_df["nz_max"][0]) if stats_df["nz_max"][0] else 0,
        "nz_mean": float(stats_df["nz_mean"][0]) if stats_df["nz_mean"][0] else 0,
        "nz_p50": float(stats_df["nz_p50"][0]) if stats_df["nz_p50"][0] else 0,
        "nz_p75": float(stats_df["nz_p75"][0]) if stats_df["nz_p75"][0] else 0,
        "nz_p90": float(stats_df["nz_p90"][0]) if stats_df["nz_p90"][0] else 0,
        "nz_p99": float(stats_df["nz_p99"][0]) if stats_df["nz_p99"][0] else 0,
    }
    
    print(f"""
Distribution:
  New users (0 prev clicks): {stats['zero_count']:,} ({stats['zero_pct']:.1f}%)
  Returning users max: {stats['nz_max']:,}
  Returning users mean: {stats['nz_mean']:.1f}
  P50: {stats['nz_p50']:.0f}, P75: {stats['nz_p75']:.0f}, P90: {stats['nz_p90']:.0f}, P99: {stats['nz_p99']:.0f}
""")
    
    p50 = round_to_nice(stats["nz_p50"])
    p75 = round_to_nice(stats["nz_p75"])
    p90 = round_to_nice(stats["nz_p90"])
    
    print_subsection("RECOMMENDED bin_prev_clicks() boundaries")
    print(f"""
Suggested boundaries:
  - "new" (== 0): First-time user
  - "returning" (1-{p50}): Up to median
  - "regular" ({p50+1}-{p75}): P50 to P75
  - "heavy" ({p75+1}-{p90}): P75 to P90
  - "power" ({p90}+): Top 10%
""")
    
    return stats


def analyze_hourly_impressions(lf: pl.LazyFrame) -> dict:
    """
    Analyze user_hourly_impressions distribution.
    """
    print_section("HOURLY IMPRESSIONS ANALYSIS")
    
    lf_with_proxy = lf.with_columns(get_user_proxy_expression())
    
    # Compute hourly counts per user
    hourly_counts = (
        lf_with_proxy
        .group_by(["user_proxy", "hour"])
        .len()
        .rename({"len": "hourly_impressions"})
    )
    
    # Collect statistics
    stats_df = (
        hourly_counts
        .select([
            pl.col("hourly_impressions").min().alias("min"),
            pl.col("hourly_impressions").max().alias("max"),
            pl.col("hourly_impressions").mean().alias("mean"),
            pl.col("hourly_impressions").median().alias("median"),
            (pl.col("hourly_impressions") == 1).sum().alias("single_count"),
            pl.len().alias("total"),
            pl.col("hourly_impressions").quantile(0.50).alias("p50"),
            pl.col("hourly_impressions").quantile(0.75).alias("p75"),
            pl.col("hourly_impressions").quantile(0.90).alias("p90"),
            pl.col("hourly_impressions").quantile(0.95).alias("p95"),
        ])
        .collect()
    )
    
    single_count = int(stats_df["single_count"][0])
    total = int(stats_df["total"][0])
    
    stats = {
        "min": int(stats_df["min"][0]),
        "max": int(stats_df["max"][0]),
        "mean": float(stats_df["mean"][0]),
        "median": float(stats_df["median"][0]),
        "single_pct": single_count / total * 100,
        "p50": float(stats_df["p50"][0]),
        "p75": float(stats_df["p75"][0]),
        "p90": float(stats_df["p90"][0]),
        "p95": float(stats_df["p95"][0]),
    }
    
    print(f"""
Distribution:
  Range: {stats['min']} - {stats['max']}
  Mean: {stats['mean']:.2f}, Median: {stats['median']:.0f}
  Single impression: {stats['single_pct']:.1f}%
  P50: {stats['p50']:.0f}, P75: {stats['p75']:.0f}, P90: {stats['p90']:.0f}
""")
    
    p50 = round_to_nice(stats["p50"])
    p75 = round_to_nice(stats["p75"])
    p90 = round_to_nice(stats["p90"])
    
    print_subsection("RECOMMENDED bin_hourly_impressions() boundaries")
    print(f"""
Suggested boundaries:
  - "single" (== 1)
  - "2-{p50}" (<= P50)
  - "{p50+1}-{p75}" (<= P75)
  - "{p75+1}-{p90}" (<= P90)
  - "{p90}+"
""")
    
    return stats


def analyze_cardinalities(lf: pl.LazyFrame, cat_cols: list[str], min_freq: int) -> dict[str, dict]:
    """
    Analyze cardinality (unique value counts) for each categorical feature.
    """
    print_section("CARDINALITY ANALYSIS")
    print(f"Analyzing {len(cat_cols)} categorical columns with min_freq={min_freq}")
    
    results = {}
    
    for col in cat_cols:
        # Count unique values meeting min_freq threshold
        counts = (
            lf
            .select(pl.col(col).cast(pl.String))
            .group_by(col)
            .len()
            .filter(pl.col("len") >= min_freq)
            .select(pl.len())
            .collect()
        )
        
        cardinality = counts.item()
        
        # Total unique (no filter)
        total_unique = lf.select(pl.col(col).n_unique()).collect().item()
        
        results[col] = {
            "cardinality": cardinality,
            "total_unique": total_unique,
            "filtered_out": total_unique - cardinality
        }
    
    # Sort by cardinality
    sorted_results = dict(sorted(results.items(), key=lambda x: x[1]["cardinality"]))
    
    print(f"\n{'Feature':<30} {'Cardinality':>12} {'Total Unique':>15} {'Filtered':>10}")
    print("-" * 70)
    for col, info in sorted_results.items():
        print(f"{col:<30} {info['cardinality']:>12,} {info['total_unique']:>15,} {info['filtered_out']:>10,}")
    
    return sorted_results


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main EDA entry point for raw data analysis."""
    print_section("AVAZU CTR - RAW DATA EDA")
    print("Suggesting optimal parameters for data_processor.py")
    
    # Load data lazily
    train_path = CONFIG['train_path']
    print(f"\nLoading: {train_path}")
    
    lf = pl.scan_csv(train_path, schema_overrides=SCHEMA)
    
    # Get row count
    row_count = lf.select(pl.len()).collect().item()
    print(f"Total rows: {row_count:,}")
    
    # Add engineered features for full analysis
    lf_with_features = (
        lf
        .with_columns(get_user_proxy_expression())
        .with_columns(get_interaction_feature_expressions())
        .with_columns(get_time_feature_expressions())
    )
    
    all_cat_cols = CATEGORICAL_COLS + ['month', 'day_of_month', 'hour_of_day', 'day_of_week']
    
    # Run analyses
    results = {}
    
    # 1. Cardinality (for reference)
    results["cardinalities"] = analyze_cardinalities(lf_with_features, all_cat_cols, CONFIG['min_freq'])
    gc.collect()
    
    # 2. Count features
    results["count_stats"] = analyze_count_features_batched(lf_with_features, COUNT_FEATURE_COLS)
    gc.collect()
    
    # 3. Cumulative count features
    results["cumcount_stats"] = analyze_cumcount_features(lf_with_features, CUMCOUNT_COLS)
    gc.collect()
    
    # 4. Time delta
    results["time_delta_stats"] = analyze_time_delta(lf)
    gc.collect()
    
    # 5. Previous clicks
    results["prev_clicks_stats"] = analyze_prev_clicks(lf)
    gc.collect()
    
    # 6. Hourly impressions
    results["hourly_stats"] = analyze_hourly_impressions(lf)
    gc.collect()
    
    print_section("EDA COMPLETE")
    print("\nUse the suggested boundaries above to update data_processor.py binning functions.")
    
    return results


if __name__ == "__main__":
    results = main()
