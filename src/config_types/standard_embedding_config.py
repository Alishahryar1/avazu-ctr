from typing import TypedDict, Literal


class StandardEmbeddingConfig(TypedDict, total=False):
    """Config for standard nn.Embedding layer."""

    type: Literal["standard"]
    dim: int
