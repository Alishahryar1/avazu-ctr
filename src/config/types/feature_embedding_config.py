from typing import TypedDict, Literal

class FeatureEmbeddingConfig(TypedDict, total=False):
    type: Literal['standard', 'hash']
    dim: int
    num_buckets: int  # Only for type='hash'
    num_hashes: int  # Only for type='hash', default: 2
    aggregation_mode: Literal['sum', 'concatenate', 'median']  # Only for type='hash', default: 'sum'
