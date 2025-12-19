"""Squeeze-and-Excitation Network layer (SENet+) - Optimized."""
from typing import Literal
import torch
from torch import Tensor
import torch.nn as nn
from src.models.utils import get_activation

class SENetLayer(nn.Module):
    """
    Optimized SENet+ Layer.
    
    Optimizations:
    1. Flattening: Operates on concatenated tensors to avoid Python loops.
    2. Vectorization: Uses view/reshape for uniform dimensions (Fast Path).
    3. Scatter/Gather: Uses index-based reduction for variable dimensions (No loops).
    4. Fused LayerNorm: Uses a single LayerNorm operation for uniform dimensions.
    """
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

        # --- 1. Dimension Setup ---
        if isinstance(feature_dims, int):
            self.feature_dims = [feature_dims] * num_fields
        else:
            self.feature_dims = list(feature_dims)

        # Check if we can use the Uniform Fast Path
        # (All dims are equal AND LayerNorm is either off or compatible)
        unique_dims = set(self.feature_dims)
        self.is_uniform = (len(unique_dims) == 1)
        
        # Validation
        if num_groups > 1:
            for i, dim in enumerate(self.feature_dims):
                assert dim % num_groups == 0, f"Dim {dim} not divisible by groups {num_groups}"

        # --- 2. Precompute Indices for Variable Dims (Flattening Strategy) ---
        # We map every element in the flattened embedding vector to a specific group ID.
        # Total segments = num_fields * num_groups
        self.num_segments = num_fields * num_groups
        
        if not self.is_uniform:
            # Create segment indices for scatter_reduce
            # e.g. Field0(dim=2), Field1(dim=1) -> [0, 0, 1]
            segment_indices = []
            field_indices = [] # Maps element to field ID (for reweighting)
            
            for field_idx, dim in enumerate(self.feature_dims):
                group_size = dim // num_groups
                for g in range(num_groups):
                    # The global ID for this specific group
                    global_group_id = field_idx * num_groups + g
                    segment_indices.extend([global_group_id] * group_size)
                field_indices.extend([field_idx] * dim)

            # Register as buffers so they move to GPU automatically
            self.register_buffer('segment_indices', torch.tensor(segment_indices, dtype=torch.long))
            self.register_buffer('field_indices', torch.tensor(field_indices, dtype=torch.long))
            
            # For mean calculation, we need group sizes
            # We count occurrences of each group ID
            ones = torch.ones(len(segment_indices), dtype=torch.float32)
            group_sizes = torch.zeros(self.num_segments, dtype=torch.float32)
            segment_idx_tensor = torch.tensor(segment_indices, dtype=torch.long)
            group_sizes.scatter_add_(0, segment_idx_tensor, ones)
            self.register_buffer('group_sizes', group_sizes)

        # --- 3. Network Architecture ---
        num_squeeze_funcs = len(squeeze_funcs)
        squeeze_output_dim = self.num_segments * num_squeeze_funcs

        if reweight_mode == "feature":
            excitation_output_dim = num_fields
        else:
            excitation_output_dim = sum(self.feature_dims)

        reduced_dim = max(1, excitation_output_dim // reduction_ratio)
        
        self.excitation = nn.Sequential(
            nn.Linear(squeeze_output_dim, reduced_dim, bias=False),
            get_activation(hidden_activation),
            nn.Linear(reduced_dim, excitation_output_dim, bias=False),
            get_activation(excitation_activation)
        )

        # Optimization: For uniform dims, we can use ONE LayerNorm layer 
        # that broadcasts, instead of a ModuleList loop.
        if use_layer_norm:
            if self.is_uniform:
                # Apply over the last dimension (embedding dim)
                self.layer_norm = nn.LayerNorm(self.feature_dims[0])
            else:
                self.layer_norms = nn.ModuleList([
                    nn.LayerNorm(dim) for dim in self.feature_dims
                ])

    def forward(self, embeddings: list[torch.Tensor]) -> list[torch.Tensor]:
        # 1. Flatten Inputs: [Batch, Total_Dim]
        # This is the only unavoidable copy, but it enables all subsequent optimizations
        flat_emb = torch.cat(embeddings, dim=1)
        batch_size = flat_emb.size(0)

        # === SQUEEZE PHASE ===
        if self.is_uniform:
            # FAST PATH: Vectorized Reshape
            # Reshape to [Batch, Num_Fields, Num_Groups, Dim_Per_Group]
            dim_per_group = self.feature_dims[0] // self.num_groups
            reshaped = flat_emb.view(batch_size, self.num_fields, self.num_groups, dim_per_group)
            
            # Apply squeezes. Result: [Batch, Num_Fields, Num_Groups]
            # Flatten to [Batch, Num_Fields * Num_Groups]
            squeezed_list = []
            for func in self.squeeze_funcs:
                if func == 'mean':
                    val = reshaped.mean(dim=-1)
                elif func == 'max':
                    val = reshaped.amax(dim=-1)
                elif func == 'min':
                    val = reshaped.amin(dim=-1)
                elif func == 'std':
                    val = reshaped.std(dim=-1)
                elif func == 'norm':
                    val = reshaped.norm(dim=-1)
                else:
                    raise NotImplementedError(f"Opt func {func} not implemented")
                squeezed_list.append(val.flatten(1))
            
            squeeze_concat = torch.cat(squeezed_list, dim=1)

        else:
            # VARIABLE PATH: Segmented Scatter
            # Input: [Batch, Total_Dim] -> Reduce to [Batch, Num_Segments]
            # We treat the batch as one large index operation
            
            # Expand indices for batch: [Batch, Total_Dim]
            segment_indices: Tensor = self.segment_indices  # type: ignore[assignment]
            idx_expanded = segment_indices.unsqueeze(0).expand(batch_size, -1)
            
            squeezed_list = []
            for func in self.squeeze_funcs:
                # Initialize output buffer
                out = torch.zeros(batch_size, self.num_segments, 
                                device=flat_emb.device, dtype=flat_emb.dtype)
                
                if func == 'mean':
                    # scatter_add_ sums elements into the group buckets
                    out.scatter_add_(1, idx_expanded, flat_emb)
                    # Divide by precomputed group sizes
                    group_sizes: Tensor = self.group_sizes  # type: ignore[assignment]
                    out = out / group_sizes.unsqueeze(0)
                elif func == 'max':
                    # scatter_reduce_ with 'amax'
                    # Note: init with very small number or handle carefully. 
                    # If inputs are embeddings (usually small), -1e9 is safe init or copy first element.
                    # PyTorch scatter_reduce_ 'amax' handles initialization implicitly if include_self=False (v1.12+)
                    out.fill_(-1e9) 
                    out.scatter_reduce_(1, idx_expanded, flat_emb, reduce="amax", include_self=True)
                elif func == 'min':
                    out.fill_(1e9)
                    out.scatter_reduce_(1, idx_expanded, flat_emb, reduce="amin", include_self=True)
                
                squeezed_list.append(out)
            
            squeeze_concat = torch.cat(squeezed_list, dim=1)

        # === EXCITATION PHASE ===
        # [Batch, Segments * Funcs] -> [Batch, Weights]
        weights = self.excitation(squeeze_concat)

        # === REWEIGHT PHASE ===
        if self.reweight_mode == "feature":
            if self.is_uniform:
                # Weights: [Batch, Fields] -> [Batch, Fields, 1] -> Broadcast to [Batch, Fields, Dim]
                # Then flatten back to match flat_emb
                weights_expanded = weights.unsqueeze(2).expand(-1, -1, self.feature_dims[0]).reshape(batch_size, -1)
            else:
                # Use index_select to map field weights to element positions
                # self.field_indices maps every element to its field ID
                # weights: [Batch, Fields] -> index_select -> [Batch, Total_Dim]
                field_indices: Tensor = self.field_indices  # type: ignore[assignment]
                weights_expanded = torch.index_select(weights, 1, field_indices)
            
            output = flat_emb * weights_expanded
        else:
            # Element mode: Weights are already [Batch, Total_Dim]
            output = flat_emb * weights

        # === FUSE PHASE ===
        if self.use_fuse:
            output = output + flat_emb

        # === LAYER NORM PHASE ===
        if self.use_layer_norm:
            if self.is_uniform:
                # Optimization: Apply LayerNorm on the structured tensor
                # Reshape [B, F*D] -> [B, F, D], apply LN(D), reshape back
                # This is mathematically equivalent to looping but fully vectorized
                temp = output.view(batch_size, self.num_fields, -1)
                output = self.layer_norm(temp).view(batch_size, -1)
            else:
                # Fallback for variable dims: must split to apply different LN modules
                # This is the only remaining loop, unavoidable for variable-dim LN
                split_out = output.split(self.feature_dims, dim=1)
                output = torch.cat([ln(t) for ln, t in zip(self.layer_norms, split_out)], dim=1)

        # Return list (as per interface)
        return list(output.split(self.feature_dims, dim=1))