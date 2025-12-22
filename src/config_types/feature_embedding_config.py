from typing import Union

from .standard_embedding_config import StandardEmbeddingConfig
from .hash_embedding_config import HashEmbeddingConfig
from .numerical_embedding_config import NumericalEmbeddingConfig


# Union type alias for any embedding config
FeatureEmbeddingConfig = Union[
    StandardEmbeddingConfig, HashEmbeddingConfig, NumericalEmbeddingConfig
]
