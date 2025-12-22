"""Numerical Embedding for continuous features."""

import torch
import torch.nn as nn


class NumericalEmbedding(nn.Module):
    """
    Embedding layer for continuous numerical features.

    Projects a scalar value into a dense embedding space via a linear transformation,
    optionally with log transform, batch normalization, and dropout.

    Args:
        embedding_dim: Output embedding dimension
        use_log_transform: Apply log1p(x) before projection (good for count features)
        use_batch_norm: Whether to apply batch normalization
        dropout: Dropout probability (0 = no dropout)
    """

    def __init__(
        self,
        embedding_dim: int,
        use_log_transform: bool = False,
        use_batch_norm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = embedding_dim
        self.use_log_transform = use_log_transform

        # Linear projection: scalar -> embedding_dim
        self.projection = nn.Linear(1, embedding_dim)

        # Optional batch norm
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(embedding_dim)

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Initialize
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for numerical embedding.

        Args:
            x: Tensor of shape [batch_size] containing continuous values

        Returns:
            Tensor of shape [batch_size, embedding_dim]
        """
        # Reshape to [batch_size, 1]
        x = x.unsqueeze(-1).float()

        # Apply log transform if enabled (log1p for numerical stability with 0s)
        if self.use_log_transform:
            x = torch.log1p(x.clamp(min=0))  # log(1 + x), clamp to handle negatives

        # Project to embedding space
        x = self.projection(x)

        # Apply batch norm
        if self.use_batch_norm:
            x = self.batch_norm(x)

        # Apply dropout
        if self.dropout is not None:
            x = self.dropout(x)

        return x
