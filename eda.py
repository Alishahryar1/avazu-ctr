"""
Avazu CTR EDA Script - Optimal Binning & Embedding Rules

This script analyzes the training data to determine:
1. Optimal embedding dimension rules based on feature cardinalities
2. Optimal binning boundaries for count/cumcount/time features

Run: python eda.py
"""

import polars as pl
import numpy as np
from pathlib import Path
from config import CONFIG
from data_processor import (
    SCHEMA, BASE_CATEGORICAL_COLS, CATEGORICAL_COLS,
    get_user_proxy_expression, get_interaction_feature_expressions,
    get_time_feature_expressions, COUNT_FEATURE_COLS, CUMCOUNT_COLS,
    get_cumulative_count_expressions, compute_time_delta_features,
    compute_previous_click_count
)


def analyze_cardinalities(lf: pl.LazyFrame, cat_cols: list[str], min_freq: int) -> dict:
    """
    Analyze cardinality (unique value counts) for each categorical feature.
    
    Returns:
        Dictionary with column name -> cardinality info
    """
    print("\n" + "="*80)
    print("CARDINALITY ANALYSIS (for embedding dimension rules)")
    print("="*80)
    
    results = {}
    
    for col in cat_cols:
        # Get unique values that appear >= min_freq times
        counts = (
            lf
            .select(pl.col(col).cast(pl.String))
            .group_by(col)
            .len()
            .filter(pl.col("len") >= min_freq)
            .collect()
        )
        
        cardinality = len(counts)
        total_unique = lf.select(pl.col(col).n_unique()).collect().item()
        
        results[col] = {
            "cardinality": cardinality,  # After min_freq filtering
            "total_unique": total_unique,  # Before filtering
            "filtered_out": total_unique - cardinality
        }
    
    # Sort by cardinality for display
    sorted_cols = sorted(results.items(), key=lambda x: x[1]["cardinality"])
    
    print(f"\nFeature cardinalities (min_freq={min_freq}):\n")
    print(f"{'Feature':<30} {'Cardinality':>12} {'Total Unique':>15} {'Filtered':>10}")
    print("-" * 70)
    
    for col, info in sorted_cols:
        print(f"{col:<30} {info['cardinality']:>12,} {info['total_unique']:>15,} {info['filtered_out']:>10,}")
    
    return dict(sorted_cols)


def suggest_embedding_rules(cardinalities: dict) -> list[tuple[int, int]]:
    """
    Suggest optimal embedding dimension rules based on cardinality distribution.
    
    Uses the rule of thumb: embedding_dim ≈ min(50, (cardinality + 1) // 2) or similar heuristics
    Also considers practical groupings based on actual cardinalities.
    """
    print("\n" + "="*80)
    print("SUGGESTED EMBEDDING DIMENSION RULES")
    print("="*80)
    
    # Get all cardinalities
    cards = [info["cardinality"] for info in cardinalities.values()]
    
    # Calculate percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    pct_values = np.percentile(cards, percentiles)
    
    print("\nCardinality percentiles:")
    for p, v in zip(percentiles, pct_values):
        print(f"  {p}th percentile: {v:,.0f}")
    
    # Suggest rules based on natural breakpoints
    # Rule of thumb: sqrt(cardinality) or log(cardinality) correlates with embedding dim
    
    # Find natural breakpoints
    breakpoints = []
    
    # Very low cardinality (e.g., device_type = 5, banner_pos = 8)
    very_low = [c for c in cards if c <= 10]
    if very_low:
        breakpoints.append((10, 8))  # Cardinality <= 10 -> dim 8
    
    # Low cardinality (< 100)
    low = [c for c in cards if 10 < c <= 100]
    if low:
        breakpoints.append((100, 16))
    
    # Medium-low (< 500)
    med_low = [c for c in cards if 100 < c <= 500]
    if med_low:
        breakpoints.append((500, 24))
    
    # Medium (< 5000)
    medium = [c for c in cards if 500 < c <= 5000]
    if medium:
        breakpoints.append((5000, 32))
    
    # Medium-high (< 20000)
    med_high = [c for c in cards if 5000 < c <= 20000]
    if med_high:
        breakpoints.append((20000, 48))
    
    # High cardinality (< 100000)
    high = [c for c in cards if 20000 < c <= 100000]
    if high:
        breakpoints.append((100000, 64))
    
    # Very high (beyond) use default embedding_dim
    
    print("\n" + "-"*60)
    print("RECOMMENDED embedding_dim_rules for config.py:")
    print("-"*60)
    print('"embedding_dim_rules": [')
    for max_card, dim in breakpoints:
        # Show which features fall into this bucket
        features_in_bucket = [
            col for col, info in cardinalities.items()
            if info["cardinality"] <= max_card and 
            (not breakpoints or info["cardinality"] > (breakpoints[breakpoints.index((max_card, dim)) - 1][0] if breakpoints.index((max_card, dim)) > 0 else 0))
        ]
        print(f'    ({max_card:>6}, {dim:>2}),  # {len(features_in_bucket)} features')
    print('],')
    print(f'\n"embedding_dim": 128,  # Default for cardinality > {breakpoints[-1][0] if breakpoints else 10000}')
    
    return breakpoints


def analyze_count_distribution(lf: pl.LazyFrame, col: str) -> dict:
    """
    Analyze distribution of a count-type column to find optimal bin boundaries.
    """
    # Compute count feature
    count_df = (
        lf
        .select(pl.col(col).cast(pl.String))
        .group_by(col)
        .len()
        .rename({"len": f"{col}_count"})
    )
    
    # Join back
    lf_with_count = lf.join(count_df, on=col, how="left")
    
    # Get distribution statistics
    stats = (
        lf_with_count
        .select(pl.col(f"{col}_count"))
        .collect()
    )
    
    counts = stats[f"{col}_count"].to_numpy()
    
    percentiles = [25, 50, 75, 90, 95, 99, 99.9]
    pct_values = np.percentile(counts, percentiles)
    
    return {
        "min": int(counts.min()),
        "max": int(counts.max()),
        "mean": float(counts.mean()),
        "median": float(np.median(counts)),
        "percentiles": dict(zip(percentiles, [float(v) for v in pct_values]))
    }


def analyze_count_features(lf: pl.LazyFrame, count_cols: list[str]) -> dict:
    """
    Analyze count feature distributions to suggest optimal binning.
    """
    print("\n" + "="*80)
    print("COUNT FEATURE ANALYSIS (for bin_count_features)")
    print("="*80)
    
    all_stats = {}
    
    for col in count_cols:
        print(f"\n{col}_count distribution:")
        stats = analyze_count_distribution(lf, col)
        all_stats[col] = stats
        
        print(f"  Min: {stats['min']:,}, Max: {stats['max']:,}")
        print(f"  Mean: {stats['mean']:,.1f}, Median: {stats['median']:,.1f}")
        print(f"  Percentiles:")
        for p, v in stats["percentiles"].items():
            print(f"    {p}th: {v:,.0f}")
    
    # Suggest bins based on median of all percentiles
    print("\n" + "-"*60)
    print("RECOMMENDED bin_count_features() boundaries:")
    print("-"*60)
    
    # Aggregate percentiles across all count features
    all_p25 = np.mean([s["percentiles"][25] for s in all_stats.values()])
    all_p50 = np.mean([s["percentiles"][50] for s in all_stats.values()])
    all_p75 = np.mean([s["percentiles"][75] for s in all_stats.values()])
    all_p90 = np.mean([s["percentiles"][90] for s in all_stats.values()])
    all_p95 = np.mean([s["percentiles"][95] for s in all_stats.values()])
    all_p99 = np.mean([s["percentiles"][99] for s in all_stats.values()])
    
    # Round to nice numbers
    def round_nice(x):
        if x < 10:
            return int(x)
        elif x < 100:
            return int(round(x / 5) * 5)
        elif x < 1000:
            return int(round(x / 10) * 10)
        else:
            return int(round(x / 100) * 100)
    
    suggested_bins = [
        0,
        1,
        round_nice(all_p25) if all_p25 > 2 else 5,
        round_nice(all_p50) if all_p50 > 5 else 10,
        round_nice(all_p75),
        round_nice(all_p90),
        round_nice(all_p99),
        "+"
    ]
    
    print(f"Based on aggregated percentiles across all count features:")
    print(f"  Suggested boundaries: {suggested_bins}")
    p25 = round_nice(all_p25) if all_p25 > 2 else 5
    p50 = round_nice(all_p50) if all_p50 > 5 else 10
    p75 = round_nice(all_p75)
    p90 = round_nice(all_p90)
    
    print(f"""
    pl.when(pl.col(count_col) == 0).then(pl.lit("0"))
    .when(pl.col(count_col) == 1).then(pl.lit("1"))
    .when(pl.col(count_col) <= {p25}).then(pl.lit("2-{p25}"))
    .when(pl.col(count_col) <= {p50}).then(pl.lit("{p25+1}-{p50}"))
    .when(pl.col(count_col) <= {p75}).then(pl.lit("{p50+1}-{p75}"))
    .when(pl.col(count_col) <= {p90}).then(pl.lit("{p75+1}-{p90}"))
    .otherwise(pl.lit("{p90}+"))
    """)
    
    return all_stats


def analyze_cumcount_distribution(lf: pl.LazyFrame, cumcount_cols: list[str]) -> dict:
    """
    Analyze cumulative count distributions.
    """
    print("\n" + "="*80)
    print("CUMULATIVE COUNT ANALYSIS (for bin_cumcount_features)")
    print("="*80)
    
    # Need to sort by hour first for proper cumcount
    lf = lf.sort("hour")
    
    cumcount_exprs = get_cumulative_count_expressions(cumcount_cols)
    lf = lf.with_columns(cumcount_exprs)
    
    all_stats = {}
    
    for col in cumcount_cols:
        cumcount_col = f"{col}_cumcount"
        
        stats_df = lf.select(pl.col(cumcount_col)).collect()
        values = stats_df[cumcount_col].to_numpy()
        
        percentiles = [25, 50, 75, 90, 95, 99]
        pct_values = np.percentile(values, percentiles)
        
        stats = {
            "min": int(values.min()),
            "max": int(values.max()),
            "mean": float(values.mean()),
            "percentiles": dict(zip(percentiles, [float(v) for v in pct_values]))
        }
        all_stats[col] = stats
        
        print(f"\n{cumcount_col} distribution:")
        print(f"  Min: {stats['min']:,}, Max: {stats['max']:,}")
        print(f"  Mean: {stats['mean']:,.1f}")
        print(f"  Percentiles:")
        for p, v in stats["percentiles"].items():
            print(f"    {p}th: {v:,.0f}")
    
    # Suggest bins
    print("\n" + "-"*60)
    print("RECOMMENDED bin_cumcount_features() boundaries:")
    print("-"*60)
    
    all_p50 = np.mean([s["percentiles"][50] for s in all_stats.values()])
    all_p75 = np.mean([s["percentiles"][75] for s in all_stats.values()])
    all_p90 = np.mean([s["percentiles"][90] for s in all_stats.values()])
    all_p95 = np.mean([s["percentiles"][95] for s in all_stats.values()])
    
    print(f"  Aggregated P50: {all_p50:.0f}, P75: {all_p75:.0f}, P90: {all_p90:.0f}, P95: {all_p95:.0f}")
    print("""
Suggested bins: 
  - "first" (==1)
  - "2-3" (<=3)
  - "4-{p50}" (<=p50)
  - "{p50+1}-{p75}" (<=p75)
  - "{p75+1}-{p90}" (<=p90)
  - "{p90}+"
""".format(p50=int(all_p50), p75=int(all_p75), p90=int(all_p90)))
    
    return all_stats


def analyze_time_delta(lf: pl.LazyFrame) -> dict:
    """
    Analyze hours_since_last_click distribution.
    """
    print("\n" + "="*80)
    print("TIME DELTA ANALYSIS (for bin_time_delta_features)")
    print("="*80)
    
    # Need user_proxy first
    lf = lf.with_columns(get_user_proxy_expression())
    lf = lf.sort("hour")
    lf = compute_time_delta_features(lf, group_col="user_proxy")
    
    stats_df = lf.select(pl.col("hours_since_last_click")).collect()
    values = stats_df["hours_since_last_click"].to_numpy()
    
    # Exclude 0 (first clicks) for non-zero analysis
    non_zero = values[values > 0]
    
    percentiles = [25, 50, 75, 90, 95, 99]
    
    if len(non_zero) > 0:
        pct_values = np.percentile(non_zero, percentiles)
    else:
        pct_values = [0] * len(percentiles)
    
    stats = {
        "zero_count": int(np.sum(values == 0)),
        "zero_pct": float(np.mean(values == 0) * 100),
        "non_zero_min": int(non_zero.min()) if len(non_zero) > 0 else 0,
        "non_zero_max": int(non_zero.max()) if len(non_zero) > 0 else 0,
        "non_zero_mean": float(non_zero.mean()) if len(non_zero) > 0 else 0,
        "percentiles": dict(zip(percentiles, [float(v) for v in pct_values]))
    }
    
    print(f"\nhours_since_last_click distribution:")
    print(f"  Zero (first clicks): {stats['zero_count']:,} ({stats['zero_pct']:.1f}%)")
    print(f"  Non-zero range: {stats['non_zero_min']} - {stats['non_zero_max']} hours")
    print(f"  Non-zero mean: {stats['non_zero_mean']:.1f} hours")
    print(f"  Non-zero percentiles:")
    for p, v in stats["percentiles"].items():
        print(f"    {p}th: {v:.1f} hours")
    
    print("\n" + "-"*60)
    print("RECOMMENDED bin_time_delta_features() boundaries:")
    print("-"*60)
    
    p50 = stats["percentiles"][50]
    p75 = stats["percentiles"][75]
    p90 = stats["percentiles"][90]
    
    print(f"""
Based on percentiles:
  - "first" (==0): First click
  - "same_hour" (<1h): Very quick return
  - "1-{int(p50)}h" (<=P50): Short interval
  - "{int(p50)+1}-{int(p75)}h" (<=P75): Medium interval
  - "{int(p75)+1}-{int(p90)}h" (<=P90): Long interval
  - "{int(p90)}h+": Very long / re-engagement
""")
    
    return stats


def analyze_prev_clicks(lf: pl.LazyFrame) -> dict:
    """
    Analyze user_proxy_prev_clicks distribution.
    """
    print("\n" + "="*80)
    print("PREVIOUS CLICKS ANALYSIS (for bin_prev_clicks)")
    print("="*80)
    
    # Need user_proxy first
    lf = lf.with_columns(get_user_proxy_expression())
    lf = lf.sort("hour")
    lf = compute_previous_click_count(lf, group_col="user_proxy")
    
    stats_df = lf.select(pl.col("user_proxy_prev_clicks")).collect()
    values = stats_df["user_proxy_prev_clicks"].to_numpy()
    
    # Analyze zeros (new users) vs returning
    zero_count = np.sum(values == 0)
    non_zero = values[values > 0]
    
    percentiles = [25, 50, 75, 90, 95, 99]
    if len(non_zero) > 0:
        pct_values = np.percentile(non_zero, percentiles)
    else:
        pct_values = [0] * len(percentiles)
    
    stats = {
        "zero_count": int(zero_count),
        "zero_pct": float(zero_count / len(values) * 100),
        "non_zero_max": int(non_zero.max()) if len(non_zero) > 0 else 0,
        "non_zero_mean": float(non_zero.mean()) if len(non_zero) > 0 else 0,
        "non_zero_median": float(np.median(non_zero)) if len(non_zero) > 0 else 0,
        "percentiles": dict(zip(percentiles, [float(v) for v in pct_values]))
    }
    
    print(f"\nuser_proxy_prev_clicks distribution:")
    print(f"  New users (0 prev clicks): {stats['zero_count']:,} ({stats['zero_pct']:.1f}%)")
    print(f"  Returning users max: {stats['non_zero_max']:,}")
    print(f"  Returning users mean: {stats['non_zero_mean']:.1f}, median: {stats['non_zero_median']:.1f}")
    print(f"  Returning users percentiles:")
    for p, v in stats["percentiles"].items():
        print(f"    {p}th: {v:.0f}")
    
    print("\n" + "-"*60)
    print("RECOMMENDED bin_prev_clicks() boundaries:")
    print("-"*60)
    
    p50 = int(stats["percentiles"][50])
    p75 = int(stats["percentiles"][75])
    p90 = int(stats["percentiles"][90])
    p99 = int(stats["percentiles"][99])
    
    print(f"""
Based on percentiles:
  - "new" (==0): First-time user
  - "returning" (1-{p50}): Up to median
  - "regular" ({p50+1}-{p75}): Above median to P75
  - "heavy" ({p75+1}-{p90}): P75 to P90
  - "power" ({p90}+): Top 10% most active users
""")
    
    return stats


def analyze_hourly_impressions(lf: pl.LazyFrame) -> dict:
    """
    Analyze user_hourly_impressions distribution.
    """
    print("\n" + "="*80)
    print("HOURLY IMPRESSIONS ANALYSIS (for bin_hourly_impressions)")
    print("="*80)
    
    # Need user_proxy first
    lf = lf.with_columns(get_user_proxy_expression())
    
    # Compute hourly counts
    hourly_counts = (
        lf
        .select(["user_proxy", "hour"])
        .group_by(["user_proxy", "hour"])
        .len()
        .rename({"len": "user_hourly_impressions"})
        .collect()
    )
    
    values = hourly_counts["user_hourly_impressions"].to_numpy()
    
    percentiles = [25, 50, 75, 90, 95, 99]
    pct_values = np.percentile(values, percentiles)
    
    # Value counts for common values
    single = np.sum(values == 1)
    
    stats = {
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "single_pct": float(single / len(values) * 100),
        "percentiles": dict(zip(percentiles, [float(v) for v in pct_values]))
    }
    
    print(f"\nuser_hourly_impressions distribution:")
    print(f"  Range: {stats['min']} - {stats['max']}")
    print(f"  Mean: {stats['mean']:.2f}, Median: {stats['median']:.0f}")
    print(f"  Single impression: {stats['single_pct']:.1f}%")
    print(f"  Percentiles:")
    for p, v in stats["percentiles"].items():
        print(f"    {p}th: {v:.0f}")
    
    print("\n" + "-"*60)
    print("RECOMMENDED bin_hourly_impressions() boundaries:")
    print("-"*60)
    
    p50 = int(stats["percentiles"][50])
    p75 = int(stats["percentiles"][75])
    p90 = int(stats["percentiles"][90])
    
    print(f"""
Based on percentiles:
  - "single" (==1)
  - "2-{p50}" (<=P50)
  - "{p50+1}-{p75}" (<=P75)
  - "{p75+1}-{p90}" (<=P90)
  - "{p90}+"
""")
    
    return stats


def generate_config_recommendations(
    cardinalities: dict,
    embedding_rules: list[tuple[int, int]],
    count_stats: dict,
    cumcount_stats: dict,
    time_delta_stats: dict,
    prev_clicks_stats: dict,
    hourly_stats: dict
):
    """
    Generate a summary of all recommendations for config.py.
    """
    print("\n" + "="*80)
    print("SUMMARY: RECOMMENDED CONFIG CHANGES")
    print("="*80)
    
    print("""
# In config.py, update embedding_dim_rules:
"embedding_dim_rules": [""")
    for max_card, dim in embedding_rules:
        print(f"    ({max_card}, {dim}),")
    print("""],
"embedding_dim": 128,  # Default for highest cardinality features
""")
    
    print("""
# The binning functions in data_processor.py can be updated based on
# the percentile-based boundaries shown above. The current bins are
# reasonable but could be fine-tuned based on the actual data distribution.
""")


def main():
    """Main EDA entry point."""
    print("="*80)
    print("AVAZU CTR - EXPLORATORY DATA ANALYSIS")
    print("Analyzing optimal binning and embedding rules")
    print("="*80)
    
    # Load data lazily
    print(f"\nLoading training data from: {CONFIG['train_path']}")
    lf = pl.scan_csv(CONFIG['train_path'], schema_overrides=SCHEMA)
    
    # Get row count
    row_count = lf.select(pl.len()).collect().item()
    print(f"Total training rows: {row_count:,}")
    
    # 1. Cardinality analysis (for embedding rules)
    # Need to add engineered features first
    lf_with_features = lf.with_columns(get_user_proxy_expression())
    lf_with_features = lf_with_features.with_columns(get_interaction_feature_expressions())
    lf_with_features = lf_with_features.with_columns(get_time_feature_expressions())
    
    # All categorical columns including engineered
    all_cat_cols = CATEGORICAL_COLS + ['month', 'day_of_month', 'hour_of_day', 'day_of_week']
    
    cardinalities = analyze_cardinalities(lf_with_features, all_cat_cols, CONFIG['min_freq'])
    embedding_rules = suggest_embedding_rules(cardinalities)
    
    # 2. Count feature analysis
    count_stats = analyze_count_features(lf_with_features, COUNT_FEATURE_COLS)
    
    # 3. Cumulative count analysis
    cumcount_stats = analyze_cumcount_distribution(lf_with_features, CUMCOUNT_COLS)
    
    # 4. Time delta analysis
    time_delta_stats = analyze_time_delta(lf)
    
    # 5. Previous clicks analysis
    prev_clicks_stats = analyze_prev_clicks(lf)
    
    # 6. Hourly impressions analysis
    hourly_stats = analyze_hourly_impressions(lf)
    
    # Generate summary recommendations
    generate_config_recommendations(
        cardinalities, embedding_rules,
        count_stats, cumcount_stats, time_delta_stats,
        prev_clicks_stats, hourly_stats
    )
    
    print("\n" + "="*80)
    print("EDA COMPLETE")
    print("="*80)
    
    return {
        "cardinalities": cardinalities,
        "embedding_rules": embedding_rules,
        "count_stats": count_stats,
        "cumcount_stats": cumcount_stats,
        "time_delta_stats": time_delta_stats,
        "prev_clicks_stats": prev_clicks_stats,
        "hourly_stats": hourly_stats
    }


if __name__ == "__main__":
    results = main()
