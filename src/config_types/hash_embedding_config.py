from typing import TypedDict, Literal


class HashEmbeddingConfig(TypedDict, total=False):
    """Config for HashEmbedding layer with multiple hash functions."""

    type: Literal["hash"]
    dim: int
    num_buckets: int  # Number of shared embedding buckets
    num_hashes: int  # Number of hash functions (default: 2)
    aggregation_mode: Literal["sum", "concatenate", "median"]  # How to combine hashes
