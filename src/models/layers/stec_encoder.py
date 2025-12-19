import torch
import torch.nn as nn
from .multi_head_stec import MultiHeadSTEC
from .position_wise_ffn import PositionWiseFFN


class STECEncoderLayer(nn.Module):
    """
    Single STEC Encoder Layer combining MultiHeadSTEC with Add & Norm + FFN.

    Architecture:
        x -> MultiHeadSTEC -> Add & Norm -> FFN -> Add & Norm -> output
             (also outputs bilinear interaction)

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        ffn_hidden_dim: FFN hidden dimension (default: 4x embed_dim)
        dropout: Dropout rate
        use_ffn: Whether to include FFN (ablation showed FFN helps on large datasets)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        ffn_hidden_dim: int | None = None,
        dropout: float = 0.0,
        use_ffn: bool = True,
    ):
        super().__init__()
        self.use_ffn = use_ffn

        # Multi-head STEC attention
        self.stec = MultiHeadSTEC(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # Optional FFN
        if use_ffn:
            ffn_dim = ffn_hidden_dim if ffn_hidden_dim is not None else embed_dim * 4
            self.ffn = PositionWiseFFN(embed_dim, ffn_dim, dropout)
            self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder layer.

        Args:
            x: Input [B, F, D]

        Returns:
            output: [B, F, D] - for next layer
            bilinear: [B, H*F*F] - pooled bilinear interaction for fusion
        """
        # Self-attention with residual
        attn_out, bilinear = self.stec(x)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN with residual (if enabled)
        if self.use_ffn:
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)

        return x, bilinear
