"""Squeeze-and-Excitation Network layer (SENet+) - Optimized."""

import PIL.ImageSequence
from typing import Literal
import torch
from torch import Tensor
import torch.nn as nn
from src.models.utils import get_activation
from typing import List, Union


class SENetLayer(nn.Module):
    """
    Optimized SENet+ Layer.

    Accepts a list of embeddings (potentially different sizes), applies Squeeze-and-Excitation,
    and returns a single flattened tensor.
    """

    def __init__(
        self,
        num_fields: int,
        feature_dims: Union[List[int], int],
        squeeze_funcs: List[str] = ["mean"],
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

        # --- 1. Dimension Setup ---
        if isinstance(feature_dims, int):
            self.feature_dims = [feature_dims] * num_fields
        else:
            self.feature_dims = list(feature_dims)

        assert len(self.feature_dims) == num_fields, (
            "feature_dims length must match num_fields"
        )

        # Validation
        if num_groups > 1:
            for i, dim in enumerate(self.feature_dims):
                assert dim % num_groups == 0, (
                    f"Dim {dim} (field {i}) not divisible by groups {num_groups}"
                )

        # Register dimensions as a buffer for fast repeat_interleave later
        # This allows us to expand [Batch, Num_Fields] -> [Batch, Total_Dim] efficiently
        self.register_buffer(
            "field_dims_tensor", torch.tensor(self.feature_dims, dtype=torch.long)
        )

        # --- 2. Network Architecture ---
        num_squeeze_funcs = len(squeeze_funcs)

        # Input to Excitation: (Num_Fields * Num_Groups * Num_Funcs)
        squeeze_output_dim = num_fields * num_groups * num_squeeze_funcs

        if reweight_mode == "feature":
            # Output: One weight per field
            excitation_output_dim = num_fields
        else:
            # Output: One weight per element (total embedding size)
            excitation_output_dim = sum(self.feature_dims)

        reduced_dim = max(1, squeeze_output_dim // reduction_ratio)

        self.excitation = nn.Sequential(
            nn.Linear(squeeze_output_dim, reduced_dim, bias=False),
            get_activation(hidden_activation),
            nn.Linear(reduced_dim, excitation_output_dim, bias=False),
            get_activation(excitation_activation),
        )

        # Single LayerNorm for the entire flattened embedding
        self.total_dim = sum(self.feature_dims)
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.total_dim)

    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            embeddings: List of tensors, where each is [Batch, Dim_i]
        Returns:
            Fused and reweighted tensor: [Batch, Sum(Dim_i)]
        """
        batch_size = embeddings[0].shape[0]

        # --- 1. Group embeddings by dimension (Optimization) ---
        # We group to vectorize operations, but we must track original indices
        # to restore order later.
        # grouped: dim -> (list_of_original_indices, list_of_tensors)
        grouped = {}
        for i, emb in enumerate(embeddings):
            dim = self.feature_dims[i]
            if dim not in grouped:
                grouped[dim] = ([], [])
            grouped[dim][0].append(i)
            grouped[dim][1].append(emb)

        # --- 2. Squeeze Operation ---
        # We need to store results in a way we can re-order them to [0..num_fields]
        # Placeholder list to put results back in correct order
        ordered_squeeze_outputs = [None] * self.num_fields

        for dim, (indices, embs) in grouped.items():
            # Stack: [num_fields_in_group, batch_size, dim]
            stacked = torch.stack(embs, dim=0)

            # Reshape for groups: [num_fields_in_group, batch_size, num_groups, dim // num_groups]
            group_dim = dim // self.num_groups
            stacked = stacked.view(len(indices), batch_size, self.num_groups, group_dim)

            # Calculate stats
            stats_list = []
            for func_name in self.squeeze_funcs:
                if func_name == "mean":
                    stat = torch.mean(stacked, dim=-1)
                elif func_name == "max":
                    stat = torch.max(stacked, dim=-1).values
                elif func_name == "min":
                    stat = torch.min(stacked, dim=-1).values
                elif func_name == "std":
                    stat = torch.std(stacked, dim=-1)
                else:
                    raise ValueError(f"Unknown squeeze function: {func_name}")
                stats_list.append(stat)

            # Concatenate functions: [num_fields_in_group, batch_size, num_groups * num_funcs]
            group_stats = torch.cat(stats_list, dim=-1)

            # Permute to [batch_size, num_fields_in_group, num_groups * num_funcs]
            group_stats = group_stats.permute(1, 0, 2)

            # Scatter back to ordered list
            for i, original_idx in enumerate(indices):
                ordered_squeeze_outputs[original_idx] = group_stats[:, i, :]

        # Concatenate all fields: [batch_size, num_fields * num_groups * num_funcs]
        # This ensures the input to the Linear layer is always in field-order (0, 1, 2...)
        squeeze_concat = torch.cat(ordered_squeeze_outputs, dim=1)

        # --- 3. Excitation ---
        # attn_scores: [batch_size, num_fields] OR [batch_size, total_dim]
        attn_scores = self.excitation(squeeze_concat)

        # --- 4. Flatten Inputs ---
        # [batch_size, total_dim]
        flat_embeddings = torch.cat(embeddings, dim=1)

        # --- 5. Re-weighting (Fixing Shape Mismatch) ---
        if self.reweight_mode == "feature":
            # attn_scores is [B, num_fields]. flat_embeddings is [B, total_dim].
            # We must repeat the score for field_i by dim_i times.
            # repeat_interleave is highly optimized for this.
            expanded_scores = torch.repeat_interleave(  # type: ignore
                attn_scores, self.field_dims_tensor, dim=1
            )
            weighted = flat_embeddings * expanded_scores
        else:
            # Element mode: shapes already match [B, total_dim]
            weighted = flat_embeddings * attn_scores

        # --- 6. Fuse (Residual Connection) ---
        if self.use_fuse:
            weighted = flat_embeddings + weighted

        # --- 7. LayerNorm ---
        if self.use_layer_norm:
            weighted = self.layer_norm(weighted)

        return weighted
