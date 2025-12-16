"""FCNv2 (Feature Cross Network v2) model for CTR prediction."""
import torch
import torch.nn as nn

from src.config.config import ConfigType
from src.models.utils import compute_embedding_dim
from src.models.layers.multihead_embedding import MultiHeadFeatureEmbedding
from src.models.layers.exp2lin_cross_network import Exponential2LinearCrossNetwork
from src.models.layers.lin2exp_cross_network import Linear2ExponentialCrossNetwork


class FCNv2Model(nn.Module):
    """
    FCNv2 (Feature Cross Network v2) for CTR prediction.
    
    Combines two cross network paths:
    - E2LCN: Exponential-to-Linear Cross Network
    - L2ECN: Linear-to-Exponential Cross Network
    
    The final prediction is the average of both paths.
    Uses TriBCELoss for weighted auxiliary loss training.
    
    Args:
        vocab_sizes: Dictionary mapping feature names to vocabulary sizes.
        feature_names: List of feature names in order.
        config: Configuration dictionary with model hyperparameters.
    """
    def __init__(
        self, 
        vocab_sizes: dict[str, int], 
        feature_names: list[str], 
        config: ConfigType
    ):
        super().__init__()
        self.feature_names = feature_names
        self.num_fields = len(feature_names)
        
        # Extract config values
        embedding_dim = config['embedding_dim']
        num_heads = config['fcnv2_num_heads']
        exp_num_layers = config['fcnv2_exp_num_layers']
        lin_num_layers = config['fcnv2_lin_num_layers']
        batch_norm = config['fcnv2_batch_norm']
        layer_norm = config['fcnv2_layer_norm']
        net_dropout = config['fcnv2_dropout']
        
        # Variable embeddings support
        use_variable_embeddings = config.get('use_variable_embeddings', False)
        feature_overrides = config.get('feature_embedding_overrides', {})
        
        # 1. Embedding Layer
        self.embeddings = nn.ModuleDict()
        self.feature_dims: dict[str, int] = {}
        total_embed_dim = 0
        
        # FCNv2 requires embeddings to be divisible by num_heads
        # We scale up the embedding dimension to ensure divisibility
        scaled_embedding_dim = embedding_dim * num_heads
        
        for feat in feature_names:
            # Check for manual override first
            if feat in feature_overrides and 'embedding_dim' in feature_overrides[feat]:
                feat_dim = feature_overrides[feat]['embedding_dim']
            elif use_variable_embeddings:
                feat_dim = compute_embedding_dim(vocab_sizes[feat], config)
            else:
                feat_dim = scaled_embedding_dim
            
            # Ensure divisibility by num_heads
            if feat_dim % num_heads != 0:
                feat_dim = ((feat_dim // num_heads) + 1) * num_heads
            
            self.feature_dims[feat] = feat_dim
            emb = nn.Embedding(vocab_sizes[feat], feat_dim)
            nn.init.xavier_uniform_(emb.weight)
            self.embeddings[feat] = emb
            total_embed_dim += feat_dim
        
        self.total_embed_dim = total_embed_dim
        self.num_heads = num_heads
        
        # Compute input dimension per head
        input_dim_per_head = total_embed_dim // num_heads
        
        # Ensure input_dim is even for the cross networks (they split in half)
        if input_dim_per_head % 2 != 0:
            input_dim_per_head += 1
            self.total_embed_dim = input_dim_per_head * num_heads
            # Add padding projection
            self.pad_projection = nn.Linear(total_embed_dim, self.total_embed_dim)
        else:
            self.pad_projection = None
        
        # 2. Multi-Head Feature Embedding
        self.multihead_embedding = MultiHeadFeatureEmbedding(num_heads=num_heads)
        
        # 3. Dual-Path Cross Networks
        self.E2LCN = Exponential2LinearCrossNetwork(
            input_dim=input_dim_per_head,
            exp_num_layers=exp_num_layers,
            lin_num_layers=lin_num_layers,
            batch_norm=batch_norm,
            layer_norm=layer_norm,
            net_dropout=net_dropout,
            num_heads=num_heads
        )
        
        self.L2ECN = Linear2ExponentialCrossNetwork(
            input_dim=input_dim_per_head,
            exp_num_layers=exp_num_layers,
            lin_num_layers=lin_num_layers,
            batch_norm=batch_norm,
            layer_norm=layer_norm,
            net_dropout=net_dropout,
            num_heads=num_heads
        )
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input features [B, Num_Features]
            
        Returns:
            Dictionary with:
            - y_pred: Combined prediction (averaged)
            - y_d: E2L path prediction
            - y_s: L2E path prediction
        """
        # 1. Get embeddings for each feature
        embeds = []
        for i, feat in enumerate(self.feature_names):
            embeds.append(self.embeddings[feat](x[:, i]))
        
        # Concatenate: [B, Total_Embed_Dim]
        feature_emb = torch.cat(embeds, dim=1)
        
        # Apply padding projection if needed
        if self.pad_projection is not None:
            feature_emb = self.pad_projection(feature_emb)
        
        # 2. Split into multi-head format: [B, H, D/H]
        batch_size = feature_emb.shape[0]
        head_dim = feature_emb.shape[1] // self.num_heads
        multihead_emb = feature_emb.view(batch_size, self.num_heads, head_dim)
        
        # 3. Dual-path cross networks
        E2L_logit = self.E2LCN(multihead_emb)  # [B, H, 1]
        L2E_logit = self.L2ECN(multihead_emb)  # [B, H, 1]
        
        # Average across heads: [B, H, 1] -> [B, 1]
        E2L_logit = E2L_logit.mean(dim=1)
        L2E_logit = L2E_logit.mean(dim=1)
        
        # 4. Combined prediction
        combined_logit = (E2L_logit + L2E_logit) * 0.5
        
        return {
            "y_pred": combined_logit,
            "y_d": E2L_logit,
            "y_s": L2E_logit
        }
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get only the combined logits (for inference compatibility).
        
        Args:
            x: Input features [B, Num_Features]
            
        Returns:
            Combined logits [B, 1]
        """
        return self.forward(x)["y_pred"]
