"""
Avazu CTR EDA - Preprocessed Data Analysis

Analyzes processed parquet data to suggest optimal feature_embeddings config:
- Vocabulary sizes per feature
- Embedding type recommendations (standard vs hash)
- Dimension recommendations
- Hash bucket size recommendations


Run from project root: python -m misc.eda_preprocessed
"""


import pickle
from pathlib import Path
import math

from config import CONFIG


# =============================================================================
# Utility Functions
# =============================================================================

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


def compute_embedding_dim(vocab_size: int) -> int:
    """
    Compute recommended embedding dimension based on vocabulary size.
    
    Uses empirical heuristics:
    - Very small (<10): dim 8
    - Small (<100): dim 16
    - Medium (<1000): dim 24
    - Large (<10000): dim 32
    - Very large (>=10000): dim 32 (use hash embeddings)
    """
    if vocab_size < 10:
        return 8
    elif vocab_size < 100:
        return 16
    elif vocab_size < 1000:
        return 24
    elif vocab_size < 10000:
        return 32
    else:
        return 32  # For hash embeddings


def compute_num_buckets(vocab_size: int) -> int:
    """
    Compute recommended number of buckets for hash embeddings.
    
    Heuristic: sqrt(vocab_size) * factor, with reasonable bounds.
    """
    # sqrt(vocab_size) as base, with minimum factor
    base = int(math.sqrt(vocab_size))
    
    # Scale up for better coverage
    buckets = max(1000, base * 2)
    
    # Cap at reasonable maximum
    buckets = min(buckets, 50000)
    
    # Round to nice number
    if buckets < 5000:
        buckets = int(round(buckets / 500) * 500)
    else:
        buckets = int(round(buckets / 1000) * 1000)
    
    return max(1000, buckets)


def recommend_embedding_type(feature_name: str, vocab_size: int) -> str:
    """
    Recommend embedding type based on vocabulary size.
    
    - High cardinality (>10k): hash (memory efficient)
    - Cross/interaction features: hash (very high cardinality expected)
    - Everything else: standard
    """
    # Interaction features always use hash
    if "_x_" in feature_name:
        return "hash"
    
    # User proxy and high-cardinality device features
    if feature_name in ["user_proxy", "device_id", "device_ip"]:
        return "hash"
    
    # Threshold-based
    if vocab_size > 10000:
        return "hash"
    
    return "standard"


# =============================================================================
# Analysis Functions
# =============================================================================

def load_vocab_sizes(processed_path: Path) -> dict[str, int]:
    """Load vocabulary sizes from processed data directory."""
    vocab_path = processed_path / "vocab_sizes.pkl"
    
    if vocab_path.exists():
        with open(vocab_path, "rb") as f:
            return pickle.load(f)
    
    return {}


def compute_vocab_sizes_from_parquet(processed_path: Path) -> dict[str, int]:
    """
    Compute vocabulary sizes by scanning the processed parquet file.
    
    Falls back to this if vocab_sizes.pkl doesn't exist.
    """
    train_path = processed_path / "train.parquet"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training parquet not found: {train_path}")
    
    print(f"Computing vocab sizes from: {train_path}")
    
    # Scan parquet lazily
    lf = pl.scan_parquet(train_path)
    
    # Get column names (excluding 'click' label)
    schema = lf.collect_schema()
    feature_cols = [col for col in schema.names() if col != "click"]
    
    # Compute max value for each column (vocab size = max + 1 for 0-indexed)
    vocab_sizes = {}
    
    for col in feature_cols:
        max_val = lf.select(pl.col(col).max()).collect().item()
        vocab_sizes[col] = int(max_val) + 1  # +1 because 0-indexed
    
    return vocab_sizes


def analyze_vocab_sizes(vocab_sizes: dict[str, int]) -> dict[str, dict]:
    """
    Analyze vocabulary sizes and generate recommendations.
    """
    print_section("VOCABULARY SIZE ANALYSIS")
    
    # Sort by vocab size
    sorted_features = sorted(vocab_sizes.items(), key=lambda x: x[1])
    
    print(f"\n{'Feature':<30} {'Vocab Size':>15} {'Recommended Type':>18} {'Dim':>6}")
    print("-" * 75)
    
    recommendations = {}
    
    for feature, vocab_size in sorted_features:
        emb_type = recommend_embedding_type(feature, vocab_size)
        dim = compute_embedding_dim(vocab_size)
        
        recommendations[feature] = {
            "vocab_size": vocab_size,
            "type": emb_type,
            "dim": dim,
        }
        
        if emb_type == "hash":
            num_buckets = compute_num_buckets(vocab_size)
            recommendations[feature]["num_buckets"] = num_buckets
            recommendations[feature]["num_hashes"] = 2
        
        type_str = f"{emb_type}" + (f" ({recommendations[feature].get('num_buckets', '')})" if emb_type == "hash" else "")
        print(f"{feature:<30} {vocab_size:>15,} {type_str:>18} {dim:>6}")
    
    return recommendations


def categorize_features(recommendations: dict[str, dict]) -> dict[str, list[str]]:
    """Categorize features by cardinality tier."""
    categories = {
        "very_low": [],   # <10
        "low": [],        # 10-99
        "medium": [],     # 100-999
        "high": [],       # 1000-9999
        "very_high": [],  # 10000+
    }
    
    for feature, rec in recommendations.items():
        vocab_size = rec["vocab_size"]
        if vocab_size < 10:
            categories["very_low"].append(feature)
        elif vocab_size < 100:
            categories["low"].append(feature)
        elif vocab_size < 1000:
            categories["medium"].append(feature)
        elif vocab_size < 10000:
            categories["high"].append(feature)
        else:
            categories["very_high"].append(feature)
    
    return categories


def generate_feature_embeddings_config(recommendations: dict[str, dict]) -> str:
    """Generate feature_embeddings config dict as a string."""
    lines = ['"feature_embeddings": {']
    
    # Group by type for cleaner output
    standard_features = [(f, r) for f, r in recommendations.items() if r["type"] == "standard"]
    hash_features = [(f, r) for f, r in recommendations.items() if r["type"] == "hash"]
    
    # Standard embeddings first
    if standard_features:
        lines.append("    # --- Standard embeddings (low-medium cardinality) ---")
        for feature, rec in sorted(standard_features, key=lambda x: x[1]["vocab_size"]):
            lines.append(f'    "{feature}": {{"type": "standard", "dim": {rec["dim"]}}},')
    
    # Hash embeddings
    if hash_features:
        lines.append("    # --- Hash embeddings (high cardinality) ---")
        for feature, rec in sorted(hash_features, key=lambda x: x[1]["vocab_size"]):
            num_buckets = rec.get("num_buckets", 5000)
            num_hashes = rec.get("num_hashes", 2)
            lines.append(
                f'    "{feature}": {{"type": "hash", "dim": {rec["dim"]}, '
                f'"num_buckets": {num_buckets}, "num_hashes": {num_hashes}}},'
            )
    
    lines.append("},")
    
    return "\n".join(lines)


def estimate_parameter_count(recommendations: dict[str, dict]) -> dict:
    """Estimate total embedding parameters for standard vs hash comparison."""
    standard_params = 0
    hash_params = 0
    
    for feature, rec in recommendations.items():
        vocab_size = rec["vocab_size"]
        dim = rec["dim"]
        
        # Standard embedding params = vocab_size * dim
        standard_only = vocab_size * dim
        
        if rec["type"] == "hash":
            # Hash embedding params = num_buckets * dim + vocab_size * num_hashes (importance weights)
            num_buckets = rec.get("num_buckets", 5000)
            num_hashes = rec.get("num_hashes", 2)
            hash_layer = num_buckets * dim + vocab_size * num_hashes
            hash_params += hash_layer
            standard_params += standard_only  # For comparison
        else:
            standard_params += standard_only
            hash_params += standard_only  # Same for standard
    
    return {
        "total_with_hash": hash_params,
        "total_all_standard": standard_params,
        "savings": standard_params - hash_params,
        "savings_pct": (standard_params - hash_params) / standard_params * 100 if standard_params > 0 else 0,
    }


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main EDA entry point for preprocessed data analysis."""
    print_section("AVAZU CTR - PREPROCESSED DATA EDA")
    print("Suggesting optimal feature_embeddings config for config.py")
    
    processed_path = Path(CONFIG['processed_path'])
    print(f"\nProcessed data path: {processed_path}")
    
    # Try to load vocab sizes from pickle first
    vocab_sizes = load_vocab_sizes(processed_path)
    
    if vocab_sizes:
        print(f"Loaded vocab_sizes.pkl with {len(vocab_sizes)} features")
    else:
        print("vocab_sizes.pkl not found, computing from parquet...")
        vocab_sizes = compute_vocab_sizes_from_parquet(processed_path)
    
    print(f"Total features: {len(vocab_sizes)}")
    
    # Analyze and generate recommendations
    recommendations = analyze_vocab_sizes(vocab_sizes)
    
    # Categorize features
    print_subsection("FEATURE CATEGORIZATION")
    categories = categorize_features(recommendations)
    
    for category, features in categories.items():
        if features:
            print(f"\n{category.upper()} cardinality ({len(features)} features):")
            for f in features:
                print(f"  - {f}: {recommendations[f]['vocab_size']:,}")
    
    # Parameter estimation
    print_subsection("PARAMETER COUNT ESTIMATION")
    param_stats = estimate_parameter_count(recommendations)
    
    print(f"""
Embedding parameter estimates:
  With hash embeddings:  {param_stats['total_with_hash']:>15,} params
  All standard:          {param_stats['total_all_standard']:>15,} params
  Savings:               {param_stats['savings']:>15,} params ({param_stats['savings_pct']:.1f}%)
""")
    
    # Generate config
    print_section("RECOMMENDED feature_embeddings CONFIG")
    config_str = generate_feature_embeddings_config(recommendations)
    print(config_str)
    
    print_section("EDA COMPLETE")
    print("\nCopy the feature_embeddings dict above into config.py")
    
    return {
        "vocab_sizes": vocab_sizes,
        "recommendations": recommendations,
        "categories": categories,
        "param_stats": param_stats,
    }


if __name__ == "__main__":
    results = main()
