"""Squeeze-and-Excitation Network layer."""
import torch
import torch.nn as nn

from src.models.utils import get_activation


class SENetLayer(nn.Module):
    """
    Squeeze-and-Excitation Network (SENET) from FiBiNET paper.

    Modified to support multiple squeeze functions (mean, max, etc.) that can be
    used together. The squeeze outputs are concatenated before the excitation network.

    Reference: FiBiNET: Combining Feature Importance and Bilinear feature Interaction
    for Click-Through Rate Prediction (RecSys 2019)

    Args:
        num_fields: Number of feature fields
        embedding_dim: Dimension of each field's embedding
        squeeze_funcs: List of squeeze functions to use. Options: 'mean', 'max', 'min'
        reduction_ratio: Reduction ratio for the excitation network bottleneck
        excitation_activation: Activation function for the excitation output
    """
    # Hashmap of squeeze operations - single source of truth
    SQUEEZE_OPS = {
        "mean": lambda t: t.mean(dim=-1),
        "max": lambda t: t.max(dim=-1).values,
        "min": lambda t: t.min(dim=-1).values,
    }

    def __init__(
        self,
        num_fields: int,
        embedding_dim: int,
        squeeze_funcs: list[str] = ["mean"],
        reduction_ratio: int = 3,
        excitation_activation: str = "sigmoid"
    ):
        super().__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.squeeze_funcs = squeeze_funcs

        # Validate squeeze functions using the class-level hashmap
        for func in squeeze_funcs:
            if func not in self.SQUEEZE_OPS:
                raise ValueError(f"Unknown squeeze function: {func}. Choose from {list(self.SQUEEZE_OPS.keys())}")

        # Number of squeeze outputs determines input to excitation network
        num_squeeze_outputs = len(squeeze_funcs)
        squeeze_output_dim = num_fields * num_squeeze_outputs

        # Excitation network (2-layer MLP with bottleneck)
        reduced_dim = max(1, num_fields // reduction_ratio)
        self.excitation = nn.Sequential(
            nn.Linear(squeeze_output_dim, reduced_dim, bias=False),
            nn.ReLU(),
            nn.Linear(reduced_dim, num_fields, bias=False),
            get_activation(excitation_activation)
        )

    def forward(self, x):
        # x shape: [Batch, Num_Fields * Embed_Dim]
        batch_size = x.size(0)

        # Reshape to [Batch, Num_Fields, Embed_Dim]
        x_3d = x.view(batch_size, self.num_fields, self.embedding_dim)

        # Squeeze: Pool each field's embedding to a scalar using class-level hashmap
        squeeze_outputs = []
        for func in self.squeeze_funcs:
            squeezed = self.SQUEEZE_OPS[func](x_3d)
            squeeze_outputs.append(squeezed)

        # Concatenate squeeze outputs: [Batch, Num_Fields * Num_Squeeze_Funcs]
        squeeze_concat = torch.cat(squeeze_outputs, dim=-1)

        # Excitation: Learn field importance weights [Batch, Num_Fields]
        field_weights = self.excitation(squeeze_concat)

        # Expand weights to match embedding dimension: [Batch, Num_Fields, 1]
        field_weights = field_weights.unsqueeze(-1)

        # Re-weight: Scale each field's embedding by its importance
        x_reweighted = x_3d * field_weights

        # Flatten back to [Batch, Num_Fields * Embed_Dim]
        return x_reweighted.view(batch_size, -1)
