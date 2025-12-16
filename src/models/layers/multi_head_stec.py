import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
