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


def compute_embedding_dim(vocab_size: int, config: Mapping[str, Any]) -> int:
    """
    Compute optimal embedding dimension based on vocabulary size.

    Uses cardinality-based rules from config:
    - Smaller vocabularies get smaller embedding dimensions
    - Larger vocabularies get larger dimensions (more capacity needed)

    Args:
        vocab_size: Number of unique values for this feature
        config: Configuration dict containing embedding_dim_rules

    Returns:
        Optimal embedding dimension for this vocabulary size
    """
    if not config.get('use_variable_embeddings', False):
        return config['embedding_dim']

    # Check cardinality rules (sorted ascending by max_vocab_size)
    for max_vocab, embed_dim in config['embedding_dim_rules']:
        if vocab_size <= max_vocab:
            return embed_dim

    # Default to base embedding_dim for very large vocabularies
    return config['embedding_dim']
