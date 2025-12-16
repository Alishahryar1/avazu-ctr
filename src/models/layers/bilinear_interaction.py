import torch
import torch.nn as nn

class BilinearInteractionLayer(nn.Module):
    """
    Standalone bilinear interaction layer for final STEC block output.
    
    Used to extract bilinear interaction from the final attention output,
    producing the (N+1)th interaction for concatenation.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
    """
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.W = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.xavier_uniform_(self.W.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract bilinear interaction from input.
        
        Args:
            x: Input [B, F, D]
            
        Returns:
            bilinear: [B, H*F*F, head_dim] - flattened bilinear interaction
        """
        B, F, D = x.shape
        
        # Project x
        Wx = self.W(x)  # [B, F, D]
        
        # Reshape for multi-head
        x_heads = x.view(B, F, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, F, head_dim]
        Wx_heads = Wx.view(B, F, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, F, head_dim]
        
        # Bilinear: [B, H, F, 1, head_dim] * [B, H, 1, F, head_dim] = [B, H, F, F, head_dim]
        bilinear = x_heads.unsqueeze(3) * Wx_heads.unsqueeze(2)
        
        # Flatten: [B, H*F*F, head_dim]
        bilinear = bilinear.reshape(B, self.num_heads * F * F, self.head_dim)
        
        return bilinear
