"""Squeeze-and-Excitation Network layer."""
from collections import defaultdict
import torch
import torch.nn as nn

from src.models.utils import get_activation


class SENetLayer(nn.Module):
    """
    Squeeze-and-Excitation Network (SENET) from FiBiNET paper.

    Modified to support:
    1. Multiple squeeze functions (mean, max, etc.) that can be used together
    2. Variable embedding dimensions per field (no projection required)

    The squeeze outputs are concatenated before the excitation network.
    For efficiency, embeddings with the same dimension are grouped and processed
    together in batched operations.

    Reference: FiBiNET: Combining Feature Importance and Bilinear feature Interaction
    for Click-Through Rate Prediction (RecSys 2019)

    Args:
        num_fields: Number of feature fields
        feature_dims: List of embedding dimensions per field (or single int for uniform dims)
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
        feature_dims: list[int] | int,
        squeeze_funcs: list[str] = ["mean"],
        reduction_ratio: int = 3,
        hidden_activation: str = "relu",
        excitation_activation: str = "sigmoid"
    ):
        super().__init__()
        self.num_fields = num_fields
        self.squeeze_funcs = squeeze_funcs

        # Handle both uniform (int) and variable (list) dimensions
        if isinstance(feature_dims, int):
            self.feature_dims = [feature_dims] * num_fields
        else:
            self.feature_dims = list(feature_dims)
            assert len(self.feature_dims) == num_fields, \
                f"feature_dims length ({len(self.feature_dims)}) must match num_fields ({num_fields})"

        # Precompute dimension groups for efficient batched squeeze
        # Maps dim -> list of field indices with that dimension
        self._dim_to_indices: dict[int, list[int]] = defaultdict(list)
        for i, dim in enumerate(self.feature_dims):
            self._dim_to_indices[dim].append(i)

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
            get_activation(hidden_activation),
            nn.Linear(reduced_dim, num_fields, bias=False),
            get_activation(excitation_activation)
        )

    def forward(self, embeddings: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Forward pass with variable-dimension embeddings.

        Args:
            embeddings: List of tensors, each [Batch, Dim_i] for field i

        Returns:
            List of reweighted tensors, each [Batch, Dim_i] for field i
        """
        batch_size = embeddings[0].size(0)
        num_squeeze_funcs = len(self.squeeze_funcs)

        # Squeeze: Pool each field's embedding to scalars
        # squeeze_results[func_idx][field_idx] = [Batch] tensor
        squeeze_results: list[list[torch.Tensor | None]] = [
            [None] * self.num_fields for _ in range(num_squeeze_funcs)
        ]

        # Process each dimension group efficiently
        for dim, indices in self._dim_to_indices.items():
            # Stack embeddings with same dim: [Batch, NumInGroup, Dim]
            group_tensor = torch.stack([embeddings[i] for i in indices], dim=1)

            # Apply each squeeze function
            for func_idx, func_name in enumerate(self.squeeze_funcs):
                squeezed = self.SQUEEZE_OPS[func_name](group_tensor)  # [Batch, NumInGroup]
                # Scatter results back to original field positions
                for group_idx, field_idx in enumerate(indices):
                    squeeze_results[func_idx][field_idx] = squeezed[:, group_idx]

        # Concatenate squeeze outputs: [Batch, Num_Fields * Num_Squeeze_Funcs]
        # Order: [field0_func0, field1_func0, ..., fieldN_func0, field0_func1, ...]
        squeeze_tensors = []
        for func_results in squeeze_results:
            squeeze_tensors.extend(func_results)  # type: ignore
        squeeze_concat = torch.stack(squeeze_tensors, dim=1)  # [Batch, Num_Fields * Num_Squeeze_Funcs]

        # Excitation: Learn field importance weights [Batch, Num_Fields]
        field_weights = self.excitation(squeeze_concat)

        # Re-weight: Scale each field's embedding by its importance weight
        # Vectorized approach: expand weights to match embedding dims, then multiply
        # field_weights: [Batch, Num_Fields] -> expand to [Batch, Total_Embed_Dim]
        dims_tensor = torch.tensor(self.feature_dims, device=field_weights.device)
        expanded_weights = field_weights.repeat_interleave(dims_tensor, dim=1)  # [Batch, Total_Dim]
        
        # Concatenate embeddings and multiply in one operation
        concat_emb = torch.cat(embeddings, dim=1)  # [Batch, Total_Dim]
        reweighted_concat = concat_emb * expanded_weights
        
        # Split back to list for return
        return list(reweighted_concat.split(self.feature_dims, dim=1))
