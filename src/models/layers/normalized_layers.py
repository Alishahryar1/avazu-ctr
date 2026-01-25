"""
Normalized layers implementing nGPT (Normalized Transformer) concepts.

Key principles from "nGPT: Normalized Transformer with Representation
Learning on the Hypersphere" (Loshchilov et al., ICLR 2025):

1. All vectors (embeddings, weights) are unit norm normalized
2. Hidden state updates use LERP: h ← Norm(h + α(h_block - h))
3. Learnable eigen learning rates (α) control block contributions
4. Scaling factors restore magnitude after normalization
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """L2 normalize tensor along specified dimension."""
    return F.normalize(x, p=2, dim=dim, eps=eps)


class NormalizedEmbedding(nn.Module):
    """
    Embedding layer with unit norm normalized vectors.

    Each embedding vector is normalized to lie on the unit hypersphere.
    This ensures dot products represent cosine similarities bounded in [-1, 1].

    Args:
        num_embeddings: Size of the vocabulary
        embedding_dim: Dimension of embedding vectors
        padding_idx: Optional padding index
        normalize_grad: Whether to also normalize during backward pass
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        normalize_grad: bool = True,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.normalize_grad = normalize_grad

        # Initialize embedding weights
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self._init_weights()

    def _init_weights(self):
        # Initialize with normal distribution, then normalize
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(self.embedding_dim))
        with torch.no_grad():
            self.weight.data = l2_normalize(self.weight.data, dim=-1)
            if self.padding_idx is not None:
                self.weight.data[self.padding_idx].zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure indices are long type (required for embedding lookup)
        if x.dtype != torch.long:
            x = x.long()

        # For torch.compile compatibility, we normalize weights in-place before forward
        # and use the stored weights directly (avoids dynamic computation in embedding)
        # The normalize_weights_() should be called after optimizer step
        return F.embedding(x, self.weight, padding_idx=self.padding_idx)

    def normalize_weights_(self):
        """In-place normalize weights. Call after optimizer step."""
        with torch.no_grad():
            self.weight.data = l2_normalize(self.weight.data, dim=-1)
            if self.padding_idx is not None:
                self.weight.data[self.padding_idx].zero_()


class NormalizedLinear(nn.Module):
    """
    Linear layer with normalized weight matrix.

    Weights are normalized along the input dimension, making the output
    represent cosine similarities between input and weight vectors.

    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: Whether to include bias term
        scale_init: Initial value for learnable output scale
        scale_factor: Factor to control effective learning rate for scale
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        scale_init: float = 1.0,
        scale_factor: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight matrix (normalized along input dimension)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Learnable scaling factor to restore magnitude information
        # Using the init/scale pattern from nGPT for effective LR control
        self._scale_init = scale_init
        self._scale_factor = scale_factor
        self.scale = nn.Parameter(torch.full((out_features,), scale_factor))

        self._init_weights()

    def _init_weights(self):
        # Initialize and normalize
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(self.in_features))
        with torch.no_grad():
            self.weight.data = l2_normalize(self.weight.data, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For torch.compile compatibility, use stored weights directly
        # (normalize_weights_() should be called after optimizer step)
        out = F.linear(x, self.weight, self.bias)

        # Apply scaling (restore actual magnitude using init/scale pattern)
        actual_scale = self.scale * (self._scale_init / self._scale_factor)
        out = out * actual_scale

        return out

    def normalize_weights_(self):
        """In-place normalize weights. Call after optimizer step."""
        with torch.no_grad():
            self.weight.data = l2_normalize(self.weight.data, dim=-1)


class NormalizedMLP(nn.Module):
    """
    MLP block with nGPT-style normalized updates.

    Implements the SwiGLU-style gated MLP with normalized weights and
    scaling factors as described in nGPT:

        u = x @ W_u * s_u
        v = x @ W_v * s_v * sqrt(d_model)
        h_M = SwiGLU(u, v) @ W_o
        h = Norm(h + α_M(Norm(h_M) - h))  # LERP update with eigen LR

    Args:
        input_dim: Input/output dimension
        hidden_dim: Hidden dimension (typically 4 * input_dim)
        su_init: Initial scaling for u path
        sv_init: Initial scaling for v path (multiplied by sqrt(input_dim))
        alpha_init: Initial eigen learning rate
        alpha_scale: Scale for effective LR in Adam
        use_glu: Whether to use SwiGLU activation
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        su_init: float = 1.0,
        sv_init: float = 1.0,
        alpha_init: float = 0.05,
        alpha_scale: Optional[float] = None,
        use_glu: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_glu = use_glu

        # Default alpha_scale to 1/sqrt(input_dim) as in nGPT
        if alpha_scale is None:
            alpha_scale = 1.0 / math.sqrt(input_dim)

        self._alpha_init = alpha_init
        self._alpha_scale = alpha_scale

        # Input projections (normalized)
        self.w_u = NormalizedLinear(
            input_dim, hidden_dim, bias=False, scale_init=su_init, scale_factor=1.0
        )

        if use_glu:
            self.w_v = NormalizedLinear(
                input_dim,
                hidden_dim,
                bias=False,
                scale_init=sv_init * math.sqrt(input_dim),  # Scale by sqrt(d) for SiLU
                scale_factor=1.0,
            )

        # Output projection (normalized)
        self.w_out = NormalizedLinear(
            hidden_dim, input_dim, bias=False, scale_init=1.0, scale_factor=1.0
        )

        # Eigen learning rate (learnable, per-dimension)
        # Using init/scale pattern for effective LR control
        self.alpha = nn.Parameter(torch.full((input_dim,), alpha_scale))

    def _get_alpha(self) -> torch.Tensor:
        """Get actual alpha values using init/scale pattern."""
        return torch.abs(self.alpha) * (self._alpha_init / self._alpha_scale)

    def forward(
        self, h: torch.Tensor, use_lerp: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional LERP update.

        Args:
            h: Input hidden state [B, D]
            use_lerp: Whether to apply LERP update (else just returns block output)

        Returns:
            Tuple of (updated_h, h_M) where h_M is the raw block output
        """
        # Normalize input (ensure on hypersphere)
        h_norm = l2_normalize(h, dim=-1)

        # GLU-style gating
        u = self.w_u(h_norm)

        if self.use_glu:
            v = self.w_v(h_norm)
            # SwiGLU: u * SiLU(v)
            hidden = u * F.silu(v)
        else:
            hidden = F.silu(u)

        # Output projection
        h_M = self.w_out(hidden)

        # Normalize block output
        h_M_norm = l2_normalize(h_M, dim=-1)

        if use_lerp:
            # LERP update: h ← Norm(h + α(h_M - h))
            alpha = self._get_alpha()
            h_updated = h_norm + alpha * (h_M_norm - h_norm)
            h_updated = l2_normalize(h_updated, dim=-1)
            return h_updated, h_M_norm
        else:
            return h_M_norm, h_M_norm

    def normalize_weights_(self):
        """In-place normalize all weights. Call after optimizer step."""
        self.w_u.normalize_weights_()
        if self.use_glu:
            self.w_v.normalize_weights_()
        self.w_out.normalize_weights_()


class NormalizedResidualMLP(nn.Module):
    """
    Multi-layer MLP with nGPT-style normalized residual connections.

    Each layer applies a LERP update with its own eigen learning rate,
    keeping the hidden state on the hypersphere throughout.

    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden dimensions for each layer
        alpha_init: Initial eigen learning rate
        alpha_scale: Scale for effective LR
        su_init: Initial u scaling
        sv_init: Initial v scaling
        dropout: Dropout rate
        use_glu: Whether to use SwiGLU
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        alpha_init: float = 0.05,
        alpha_scale: Optional[float] = None,
        su_init: float = 1.0,
        sv_init: float = 1.0,
        dropout: float = 0.0,
        use_glu: bool = True,
    ):
        super().__init__()

        if alpha_scale is None:
            alpha_scale = 1.0 / math.sqrt(input_dim)

        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList() if dropout > 0 else None

        # Build layers - each layer maintains input_dim (residual style)
        for hidden_dim in hidden_dims:
            self.layers.append(
                NormalizedMLP(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    su_init=su_init,
                    sv_init=sv_init,
                    alpha_init=alpha_init,
                    alpha_scale=alpha_scale,
                    use_glu=use_glu,
                )
            )
            if self.dropouts is not None:
                self.dropouts.append(nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, use_lerp: bool = True) -> torch.Tensor:
        """
        Forward pass through all layers.

        Args:
            x: Input tensor [B, D]
            use_lerp: Whether to use LERP updates

        Returns:
            Output tensor [B, D] (same dimension due to residual structure)
        """
        # Initial normalization
        h = l2_normalize(x, dim=-1)

        for i, layer in enumerate(self.layers):
            h, _ = layer(h, use_lerp=use_lerp)

            if self.dropouts is not None:
                h = self.dropouts[i](h)
                # Re-normalize after dropout (maintains hypersphere)
                h = l2_normalize(h, dim=-1)

        return h

    def normalize_weights_(self):
        """In-place normalize all weights. Call after optimizer step."""
        for layer in self.layers:
            layer.normalize_weights_()


class WeightNormalizationCallback:
    """
    Callback to normalize weights after optimizer step.

    Usage:
        callback = WeightNormalizationCallback(model)
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
            callback()  # Normalize weights
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def __call__(self):
        """Normalize all normalized layers in the model."""
        for module in self.model.modules():
            if hasattr(module, "normalize_weights_"):
                module.normalize_weights_()
