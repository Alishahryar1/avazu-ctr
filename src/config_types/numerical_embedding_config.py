from typing import TypedDict, Literal


class NumericalEmbeddingConfig(TypedDict, total=False):
    """Config for NumericalEmbedding layer for continuous features."""

    type: Literal["numerical"]
    dim: int
    use_log_transform: bool  # Apply log1p(x) before projection (default: False)
    use_batch_norm: bool  # Apply batch normalization (default: True)
    numerical_dropout: float  # Dropout probability (default: 0.0)
