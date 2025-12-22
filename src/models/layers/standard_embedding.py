"""Standard Embedding wrapper with automatic type conversion."""

import torch
import torch.nn as nn


class StandardEmbedding(nn.Module):
    """
    Wrapper around nn.Embedding that handles float-to-long conversion.

    This allows the dataset to load all features as float32 (to support
    numerical likelihood columns) while still working with standard embeddings
    that require integer indices.

    Args:
        num_embeddings: Size of the embedding dictionary
        embedding_dim: Size of each embedding vector
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_dim = embedding_dim
        self.output_dim = embedding_dim
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with automatic type conversion.

        Args:
            x: Tensor of any numeric type containing indices

        Returns:
            Embedded tensor of shape [batch_size, embedding_dim]
        """
        return self.embedding(x.long())
