"""Squeeze-and-Excitation Network layer (SENet+)."""
from collections import defaultdict
from typing import Literal
import torch
import torch.nn as nn

from src.models.utils import get_activation


class SENetLayer(nn.Module):
    """
    Squeeze-and-Excitation Network (SENet+) from FiBiNET paper with extensions.

    Supports:
    1. Multiple squeeze functions (mean, max, etc.) that can be used together
    2. Variable embedding dimensions per field (no projection required)
    3. Grouped squeeze: Split embeddings into groups before squeezing (SENet+)
    4. Reweight modes: Feature-level or element-level reweighting
    5. Optional fuse: Add original embeddings to reweighted (residual)
    6. Optional layer norm: Apply LayerNorm after fuse

    Reference: FiBiNET: Combining Feature Importance and Bilinear feature Interaction
    for Click-Through Rate Prediction (RecSys 2019)

    Args:
        num_fields: Number of feature fields
        feature_dims: List of embedding dimensions per field (or single int for uniform dims)
        squeeze_funcs: List of squeeze functions to use. Options: 'mean', 'max', 'min'
        reduction_ratio: Reduction ratio for the excitation network bottleneck
        hidden_activation: Activation function for excitation hidden layer
        excitation_activation: Activation function for the excitation output
        num_groups: Number of groups to split each embedding into for grouped squeeze.
                    Must evenly divide all feature_dims. Default 1 = no grouping.
        reweight_mode: 'feature' = one weight per field, 'element' = weight per element
        use_fuse: If True, add original embeddings to reweighted (residual connection)
        use_layer_norm: If True, apply LayerNorm after fuse step
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
        excitation_activation: str = "sigmoid",
        num_groups: int = 1,
        reweight_mode: Literal["feature", "element"] = "feature",
        use_fuse: bool = False,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.num_fields = num_fields
        self.squeeze_funcs = squeeze_funcs
        self.num_groups = num_groups
        self.reweight_mode = reweight_mode
        self.use_fuse = use_fuse
        self.use_layer_norm = use_layer_norm

        # Handle both uniform (int) and variable (list) dimensions
        if isinstance(feature_dims, int):
            self.feature_dims = [feature_dims] * num_fields
        else:
            self.feature_dims = list(feature_dims)
            assert len(self.feature_dims) == num_fields, \
                f"feature_dims length ({len(self.feature_dims)}) must match num_fields ({num_fields})"

        # Validate num_groups divides all feature dims
        if num_groups > 1:
            for i, dim in enumerate(self.feature_dims):
                assert dim % num_groups == 0, \
                    f"num_groups ({num_groups}) must evenly divide feature_dims[{i}] ({dim})"

        # Precompute dimension groups for efficient batched squeeze
        # Maps dim -> list of field indices with that dimension
        self._dim_to_indices: dict[int, list[int]] = defaultdict(list)
        for i, dim in enumerate(self.feature_dims):
            self._dim_to_indices[dim].append(i)

        # Validate squeeze functions using the class-level hashmap
        for func in squeeze_funcs:
            if func not in self.SQUEEZE_OPS:
                raise ValueError(f"Unknown squeeze function: {func}. Choose from {list(self.SQUEEZE_OPS.keys())}")

        # Validate reweight_mode
        if reweight_mode not in ("feature", "element"):
            raise ValueError(f"reweight_mode must be 'feature' or 'element', got '{reweight_mode}'")

        # Calculate squeeze output dimension
        # With groups: each field produces num_groups squeeze values per squeeze function
        num_squeeze_outputs = len(squeeze_funcs)
        squeeze_output_dim = num_fields * num_groups * num_squeeze_outputs

        # Calculate excitation output dimension based on reweight mode
        if reweight_mode == "feature":
            excitation_output_dim = num_fields
        else:  # element
            excitation_output_dim = sum(self.feature_dims)

        # Excitation network (2-layer MLP with bottleneck)
        reduced_dim = max(1, excitation_output_dim // reduction_ratio)
        self.excitation = nn.Sequential(
            nn.Linear(squeeze_output_dim, reduced_dim, bias=False),
            get_activation(hidden_activation),
            nn.Linear(reduced_dim, excitation_output_dim, bias=False),
            get_activation(excitation_activation)
        )

        # Optional LayerNorm for each field
        if use_layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(dim) for dim in self.feature_dims
            ])
        else:
            self.layer_norms = None

    def forward(self, embeddings: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Forward pass with variable-dimension embeddings and SENet+ features.

        Args:
            embeddings: List of tensors, each [Batch, Dim_i] for field i

        Returns:
            List of (optionally fused and normalized) reweighted tensors
        """
        batch_size = embeddings[0].size(0)
        num_squeeze_funcs = len(self.squeeze_funcs)
        device = embeddings[0].device

        # Flatten all embeddings upfront: [Batch, Total_Dim]
        flat_emb = torch.cat(embeddings, dim=1)
        total_dim = flat_emb.size(1)

        # === SQUEEZE PHASE ===
        # With groups: each embedding [B, D] -> [B, G, D/G] -> squeeze each group -> [B, G]
        # squeeze_results[func_idx][field_idx] = [Batch, num_groups] tensor
        squeeze_results: list[list[torch.Tensor | None]] = [
            [None] * self.num_fields for _ in range(num_squeeze_funcs)
        ]

        # Process each dimension group efficiently
        for dim, indices in self._dim_to_indices.items():
            # Stack embeddings with same dim: [Batch, NumInGroup, Dim]
            group_tensor = torch.stack([embeddings[i] for i in indices], dim=1)

            if self.num_groups > 1:
                # Reshape to groups: [Batch, NumInGroup, G, Dim/G]
                group_dim = dim // self.num_groups
                group_tensor = group_tensor.view(batch_size, len(indices), self.num_groups, group_dim)

            # Apply each squeeze function
            for func_idx, func_name in enumerate(self.squeeze_funcs):
                # Squeeze over last dim: [Batch, NumInGroup] or [Batch, NumInGroup, G]
                squeezed = self.SQUEEZE_OPS[func_name](group_tensor)

                # Scatter results back to original field positions
                for group_idx, field_idx in enumerate(indices):
                    if self.num_groups > 1:
                        squeeze_results[func_idx][field_idx] = squeezed[:, group_idx, :]  # [B, G]
                    else:
                        squeeze_results[func_idx][field_idx] = squeezed[:, group_idx].unsqueeze(-1)  # [B, 1]

        # Concatenate squeeze outputs: [Batch, Num_Fields * Num_Groups * Num_Squeeze_Funcs]
        squeeze_tensors = []
        for func_results in squeeze_results:
            for field_result in func_results:
                squeeze_tensors.append(field_result)  # Each is [B, G]
        squeeze_concat = torch.cat(squeeze_tensors, dim=1)

        # === EXCITATION PHASE ===
        # Learn importance weights: [Batch, Num_Fields] or [Batch, Total_Dim]
        weights = self.excitation(squeeze_concat)

        # === REWEIGHT PHASE (on flattened tensor) ===
        if self.reweight_mode == "feature":
            # Expand field weights to element weights
            dims_tensor = torch.tensor(self.feature_dims, device=device)
            expanded_weights = weights.repeat_interleave(dims_tensor, dim=1)  # [Batch, Total_Dim]
            output = flat_emb * expanded_weights
        else:  # element
            # Weights are already per-element
            output = flat_emb * weights

        # === FUSE PHASE (on flattened tensor) ===
        if self.use_fuse:
            output = flat_emb + output  # residual: original + reweighted

        # === LAYER NORM PHASE (on flattened tensor) ===
        if self.use_layer_norm and self.layer_norms is not None:
            # Split, apply per-field LayerNorm, and concat back
            split_output = output.split(self.feature_dims, dim=1)
            output = torch.cat([self.layer_norms[i](emb) for i, emb in enumerate(split_output)], dim=1)

        # Split to list only at the very end
        return list(output.split(self.feature_dims, dim=1))
