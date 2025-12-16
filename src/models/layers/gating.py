"""Feature gating layer for element-wise gating."""
import torch
import torch.nn as nn

from src.models.utils import get_activation


class FeatureGatingLayer(nn.Module):
    """
    The 'Fast' implementation of the Gated Attention paper.
    Instead of O(N^2) Self-Attention, we use O(N) Element-wise Gating.
    It learns to suppress noise (sparsity) and adds non-linearity.

    Supports low-rank decomposition: W = U @ V where U is (input_dim, rank)
    and V is (rank, input_dim). This reduces parameters from O(d^2) to O(2*d*r).
    """
    def __init__(self, input_dim, gating_activation: str = "sigmoid", low_rank: int | None = None):
        super().__init__()
        self.input_dim = input_dim
        self.low_rank = low_rank
        self.activation = get_activation(gating_activation)

        if low_rank is not None:
            # Low-rank decomposition: W = U @ V
            self.U = nn.Parameter(torch.randn(input_dim, low_rank))
            self.V = nn.Parameter(torch.randn(low_rank, input_dim))
            self.bias = nn.Parameter(torch.zeros(input_dim))
            # Xavier initialization
            nn.init.xavier_uniform_(self.U)
            nn.init.xavier_uniform_(self.V)
        else:
            # Full-rank linear layer
            self.gate_linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x shape: [Batch, Num_Features * Embed_Dim]

        # Calculate Gate Score
        if self.low_rank is not None:
            # Low-rank: x @ U @ V + bias
            gate_logits = torch.matmul(torch.matmul(x, self.U), self.V) + self.bias
        else:
            # Full-rank linear
            gate_logits = self.gate_linear(x)

        gate_score = self.activation(gate_logits)

        # Apply Gate
        return x * gate_score
