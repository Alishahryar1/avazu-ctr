"""STEC Block: See-Through Transformer-based Encoder Block.

This module implements the core STEC block which extracts bilinear interactions
from scaled dot-product attention calculations, as described in the STEC paper.

Key insight: The attention matrix P = K^T * Q = x^T * W_k * W_q^T * x contains
bilinear interactions P_ij = x_i^T * W * x_j that can be extracted at no additional cost.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class STECBlock(nn.Module):
    """
    Single-head STEC Block that outputs both attention and bilinear interaction.

    The key innovation is extracting the bilinear interaction matrix from the
    intermediate attention computation, which corresponds to:
        P_ij = x_i^T * W * x_j

    This is mathematically equivalent to the bilinear interaction used in FiBiNet,
    but extracted at no additional computational cost from attention.

    Args:
        embed_dim: Embedding dimension (d)
        dropout: Dropout rate for attention weights
    """

    def __init__(self, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = math.sqrt(embed_dim)

        # Combined projection for Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for module in [self.W_q, self.W_k, self.W_v]:
            nn.init.xavier_uniform_(module.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both attention output and bilinear interaction.

        Args:
            x: Input tensor of shape [B, F, D] where F is number of fields

        Returns:
            attention_output: [B, F, D] - standard self-attention output
            bilinear_interaction: [B, F, F] - pooled bilinear interaction tensor
        """
        # x: [B, F, D]
        Q = self.W_q(x)  # [B, F, D]
        K = self.W_k(x)  # [B, F, D]
        V = self.W_v(x)  # [B, F, D]

        # Compute attention scores: P = K^T @ Q / sqrt(d)
        # Shape: [B, F, F]
        attn_scores = torch.bmm(K, Q.transpose(1, 2)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Standard attention output: [B, F, D]
        attention_output = torch.bmm(attn_weights, V)

        # Extract bilinear interaction: P_ij = x_i ⊙ (W @ x_j)
        # This is the Hadamard product formulation from the paper
        # We compute: x_i[:, None, :] * (W @ x_j)  -> [B, F, F, D]
        # Using K as W @ x (since K = W_k @ x)
        # bilinear: [B, F, 1, D] * [B, 1, F, D] = [B, F, F, D]
        bilinear_interaction = x.unsqueeze(2) * K.unsqueeze(1)

        # Apply AvgPool (pooling over embedding dimension): [B, F, F]
        bilinear_interaction = bilinear_interaction.mean(dim=-1)

        return attention_output, bilinear_interaction
