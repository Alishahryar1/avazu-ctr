"""Multi-Head Feature Embedding layer for FCNv2."""
import torch
import torch.nn as nn


class MultiHeadFeatureEmbedding(nn.Module):
    """
    Multi-Head Feature Embedding for FCNv2.
    
    Takes concatenated feature embeddings and splits them into multiple heads
    for parallel processing in the cross networks.
    
    Args:
        num_heads: Number of attention heads to split into.
    """
    def __init__(self, num_heads: int = 2):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, feature_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            feature_emb: Input embeddings [B, F, D] or [B, D]
            
        Returns:
            Multi-head embeddings [B, H, D/H] where H is num_heads
        """
        # If input is 2D [B, D], reshape to 3D for processing
        if feature_emb.dim() == 2:
            batch_size, total_dim = feature_emb.shape
            # Split into heads: [B, D] -> [B, H, D/H]
            head_dim = total_dim // self.num_heads
            multihead_emb = feature_emb.view(batch_size, self.num_heads, head_dim)
            return multihead_emb
        
        # Input is 3D [B, F, D]
        batch_size = feature_emb.shape[0]
        
        # Split embedding dimension into heads: B × F × D -> B × H × F × D/H
        multihead_feature_emb = torch.tensor_split(feature_emb, self.num_heads, dim=-1)
        multihead_feature_emb = torch.stack(multihead_feature_emb, dim=1)  # B × H × F × D/H
        
        # Split into two halves for different processing
        multihead_feature_emb1, multihead_feature_emb2 = torch.tensor_split(
            multihead_feature_emb, 2, dim=-1
        )  # B × H × F × D/2H
        
        # Flatten the field and embedding dimensions
        multihead_feature_emb1 = multihead_feature_emb1.flatten(start_dim=2)  # B × H × FD/2H
        multihead_feature_emb2 = multihead_feature_emb2.flatten(start_dim=2)  # B × H × FD/2H
        
        # Concatenate back
        multihead_feature_emb = torch.cat(
            [multihead_feature_emb1, multihead_feature_emb2], dim=-1
        )  # B × H × FD/H
        
        return multihead_feature_emb
