from typing import TypedDict, Literal


class NumericalEmbeddingConfig(TypedDict, total=False):
    """Config for NumericalEmbedding layer for continuous features."""

    type: Literal["numerical"]
    use_log_transform: bool  # Apply log1p(x) before output (default: False)
