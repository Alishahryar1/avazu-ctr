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
            bilinear_interaction: [B, F, F, D] - bilinear interaction tensor
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
        
        return attention_output, bilinear_interaction


class MultiHeadSTEC(nn.Module):
    """
    Multi-Head STEC Block with Group Bilinear Interactions.
    
    Extends STECBlock to support multiple attention heads, each learning
    different interaction subspaces. The bilinear interactions from all heads
    are grouped and concatenated.
    
    Args:
        embed_dim: Total embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Linear projections
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-head forward pass.
        
        Args:
            x: Input tensor [B, F, D]
            
        Returns:
            attention_output: [B, F, D] - concatenated multi-head attention
            bilinear_interaction: [B, F*F*H, head_dim] - grouped bilinear interactions
                where H is num_heads, flattened for efficiency
        """
        B, num_fields, D = x.shape
        
        # Project to Q, K, V
        Q = self.W_q(x)  # [B, F, D]
        K = self.W_k(x)  # [B, F, D]
        V = self.W_v(x)  # [B, F, D]
        
        # Reshape for multi-head: [B, F, H, head_dim] -> [B, H, F, head_dim]
        Q = Q.view(B, num_fields, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, num_fields, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, num_fields, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention: [B, H, F, F]
        attn_scores = torch.matmul(K, Q.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Attention output: [B, H, F, head_dim]
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape back: [B, F, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, num_fields, D)
        attn_output = self.W_o(attn_output)
        
        # Extract bilinear interactions for each head
        # K: [B, H, F, head_dim], x reshaped: [B, H, F, head_dim]
        x_heads = x.view(B, num_fields, self.num_heads, self.head_dim).transpose(1, 2)
        # bilinear: [B, H, F, 1, head_dim] * [B, H, 1, F, head_dim] = [B, H, F, F, head_dim]
        bilinear = x_heads.unsqueeze(3) * K.unsqueeze(2)
        
        # Flatten to [B, H*F*F, head_dim] for concatenation layer
        bilinear = bilinear.reshape(B, self.num_heads * num_fields * num_fields, self.head_dim)
        
        return attn_output, bilinear


class PositionWiseFFN(nn.Module):
    """
    Position-wise Feed-Forward Network as used in Transformers.
    
    FFN(x) = max(0, x @ W1 + b1) @ W2 + b2
    
    Args:
        embed_dim: Input/output dimension
        hidden_dim: Hidden layer dimension (typically 4x embed_dim)
        dropout: Dropout rate
    """
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FFN: [B, F, D] -> [B, F, D]"""
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


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
        use_ffn: bool = True
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
            bilinear: [B, H*F*F, head_dim] - bilinear interaction for fusion
        """
        # Self-attention with residual
        attn_out, bilinear = self.stec(x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN with residual (if enabled)
        if self.use_ffn:
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)
        
        return x, bilinear
