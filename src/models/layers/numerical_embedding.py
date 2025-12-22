"""Numerical Embedding for continuous features."""

import torch
import torch.nn as nn


class NumericalEmbedding(nn.Module):
    """
    Passthrough embedding layer for continuous numerical features.

    Simply reshapes the scalar value to [batch_size, 1] with optional log transform.
    No projection is applied - the numerical value is used directly as a 1D embedding.

    Args:
        use_log_transform: Apply log1p(x) before output (good for count features)
    """

    def __init__(self, use_log_transform: bool = False):
        super().__init__()
        self.embedding_dim = 1
        self.output_dim = 1
        self.use_log_transform = use_log_transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for numerical embedding.

        Args:
            x: Tensor of shape [batch_size] containing continuous values

        Returns:
            Tensor of shape [batch_size, 1]
        """
        # Reshape to [batch_size, 1]
        x = x.unsqueeze(-1).float()

        # Apply log transform if enabled (log1p for numerical stability with 0s)
        if self.use_log_transform:
            x = torch.log1p(x.clamp(min=0))  # log(1 + x), clamp to handle negatives

        return x
