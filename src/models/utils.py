"""Utility functions for model components."""
import torch.nn as nn
from typing import Any, Mapping


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "leaky_relu": nn.LeakyReLU(0.1),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "softmax": nn.Softmax(dim=-1),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
    return activations[name]


def get_embedding(
    feature_name: str,
    vocab_size: int,
    config: Mapping[str, Any],
) -> tuple[nn.Module, int]:
    """
    Create an embedding layer for a feature based on config.

    Looks up the feature in config['feature_embeddings'] and creates either
    a standard nn.Embedding or a HashEmbedding depending on the 'type' field.
    Falls back to config['embedding_dim'] with type='standard' if feature not found.

    Args:
        feature_name: Name of the feature
        vocab_size: Number of unique values (vocabulary size) for this feature
        config: Configuration dictionary containing 'feature_embeddings' and 'embedding_dim'

    Returns:
        Tuple of (embedding_module, output_dim) where output_dim is the effective
        embedding dimension (may differ from input dim for concatenate aggregation)
    """
    from src.models.layers.hash_embedding import HashEmbedding

    feature_embeddings = config.get('feature_embeddings', {})
    default_dim = config.get('embedding_dim', 16)

    # Get feature-specific config or use defaults
    feat_config = feature_embeddings.get(feature_name, {})
    embed_type = feat_config.get('type', 'standard')
    embed_dim = feat_config.get('dim', default_dim)

    if embed_type == 'hash':
        # HashEmbedding with configurable parameters
        num_buckets = feat_config.get('num_buckets', max(1, vocab_size // 10))
        num_hashes = feat_config.get('num_hashes', 2)
        aggregation_mode = feat_config.get('aggregation_mode', 'sum')

        embedding = HashEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            num_buckets=num_buckets,
            num_hashes=num_hashes,
            aggregation_mode=aggregation_mode,
        )
        output_dim = embedding.output_dim
    else:
        # Standard nn.Embedding
        embedding = nn.Embedding(vocab_size, embed_dim)
        nn.init.xavier_uniform_(embedding.weight)
        output_dim = embed_dim

    return embedding, output_dim
