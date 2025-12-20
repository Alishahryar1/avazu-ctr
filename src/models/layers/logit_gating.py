import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitGatingLayer(nn.Module):
    """
    A gating layer that learns to weight multiple logits using softmax attention.

    Takes K logits as input and outputs a weighted combination based on learned
    attention weights. Supports optional hidden layer for more expressive gating.

    Args:
        num_inputs: Number of input logits (e.g., number of heads).
        hidden_dim: Optional hidden dimension. If None, uses a single linear layer.
                    If provided, uses Linear -> ReLU -> Linear architecture.
    """

    def __init__(self, num_inputs: int, hidden_dim: int | None = None):
        super().__init__()
        self.num_inputs = num_inputs
        self.hidden_dim = hidden_dim

        if hidden_dim is not None:
            self.gate: nn.Module = nn.Sequential(
                nn.Linear(num_inputs, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_inputs),
            )
        else:
            self.gate: nn.Module = nn.Linear(num_inputs, num_inputs)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Tensor of shape [Batch, K] where K is num_inputs.

        Returns:
            Weighted aggregated logit of shape [Batch, 1].
        """
        gate_weights = F.softmax(self.gate(logits), dim=-1)  # [Batch, K]
        aggregated = (logits * gate_weights).sum(dim=-1, keepdim=True)  # [Batch, 1]
        return aggregated
