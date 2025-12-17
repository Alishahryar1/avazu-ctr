"""STEC: See-Through Transformer-based Encoder for CTR Prediction.

This module implements the full STEC architecture which:
1. Extracts bilinear interactions from attention at no additional cost
2. Fuses interactions from multiple levels with direct connections to output
3. Uses multi-head attention with group bilinear interactions

Reference: "STEC: See-Through Transformer-Based Encoder for CTR Prediction"
"""
import torch
import torch.nn as nn

from src.config.config import ConfigType
from src.models.utils import compute_embedding_dim
from src.models.types import ModelOutput
from src.models.architectures.base import BaseCTRModel
from src.models.losses import FocalLoss
from src.models.layers.stec_encoder import STECEncoderLayer
from src.models.layers.multi_head_stec import MultiHeadSTEC


from src.models.layers.bilinear_interaction import BilinearInteractionLayer



class STECModel(BaseCTRModel):
    """
    STEC: See-Through Transformer-based Encoder for CTR Prediction.
    
    Architecture:
        Input -> Embedding -> [STEC Block + Add&Norm + FFN + Add&Norm] x N
              -> Concat(BN(bilinear_0), ..., BN(bilinear_N)) -> MLP -> Prediction
    
    Key features:
    - Extracts bilinear interactions from attention at no extra cost
    - Direct connections from all interaction levels to output
    - Multi-head group bilinear interactions
    
    Args:
        vocab_sizes: Dictionary mapping feature names to vocabulary sizes
        feature_names: List of feature names in order
        config: Configuration dictionary
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
        
        # Extract config
        embedding_dim = config['embedding_dim']
        stec_num_layers = config.get('stec_num_layers', 2)
        stec_num_heads = config.get('stec_num_heads', 4)
        stec_hidden_dim = config.get('stec_hidden_dim', None)  # None = 4x embed_dim
        stec_dropout = config.get('stec_dropout', 0.1)
        stec_use_ffn = config.get('stec_use_ffn', True)
        mlp_hidden_dims = config.get('stec_mlp_hidden_dims', [256, 128])
        mlp_dropout = config.get('mlp_dropout', 0.1)
        
        self.stec_num_layers = stec_num_layers
        self.stec_num_heads = stec_num_heads
        
        # Variable embeddings support
        use_variable_embeddings = config.get('use_variable_embeddings', False)
        feature_overrides = config.get('feature_embedding_overrides', {})
        projection_dim = config.get('embedding_projection_dim', None)
        
        # 1. Embedding Layer
        self.embeddings = nn.ModuleDict()
        self.feature_dims: dict[str, int] = {}
        total_embed_dim = 0
        
        for feat in feature_names:
            if feat in feature_overrides and 'embedding_dim' in feature_overrides[feat]:
                feat_dim = feature_overrides[feat]['embedding_dim']
            elif use_variable_embeddings:
                feat_dim = compute_embedding_dim(vocab_sizes[feat], config)
            else:
                feat_dim = embedding_dim
            
            self.feature_dims[feat] = feat_dim
            emb = nn.Embedding(vocab_sizes[feat], feat_dim)
            nn.init.xavier_uniform_(emb.weight)
            self.embeddings[feat] = emb
            total_embed_dim += feat_dim
        
        self.total_embed_dim = total_embed_dim
        self.use_projection = projection_dim is not None
        self.projection = None
        
        # 2. Projection to uniform dimension (required for STEC attention)
        # STEC requires uniform embedding dims for attention
        if self.use_projection and projection_dim is not None:
            self.projection = nn.Linear(total_embed_dim, projection_dim)
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
            # working_dim per field after projection
            assert projection_dim % self.num_fields == 0, \
                f"projection_dim ({projection_dim}) must be divisible by num_fields ({self.num_fields})"
            self.embed_per_field = projection_dim // self.num_fields
        else:
            # Without projection, all embeddings must be same dim
            if use_variable_embeddings:
                # For variable embeddings without projection, use the base embedding_dim
                # and add a projection layer
                self.projection = nn.Linear(total_embed_dim, embedding_dim * self.num_fields)
                nn.init.xavier_uniform_(self.projection.weight)
                nn.init.zeros_(self.projection.bias)
                self.use_projection = True
                self.embed_per_field = embedding_dim
            else:
                self.embed_per_field = embedding_dim
        
        working_dim = self.embed_per_field * self.num_fields
        
        # Ensure embed_per_field is divisible by num_heads
        if self.embed_per_field % stec_num_heads != 0:
            # Adjust to nearest valid dimension
            head_dim = max(1, self.embed_per_field // stec_num_heads)
            self.embed_per_field = head_dim * stec_num_heads
            working_dim = self.embed_per_field * self.num_fields
            # Add a linear layer to adjust dimensions
            self.dim_adjust = nn.Linear(
                self.embed_per_field * self.num_fields // stec_num_heads * stec_num_heads,
                working_dim
            )
        else:
            self.dim_adjust = None
        
        # 4. STEC Encoder Layers
        self.stec_layers = nn.ModuleList([
            STECEncoderLayer(
                embed_dim=self.embed_per_field,
                num_heads=stec_num_heads,
                ffn_hidden_dim=stec_hidden_dim,
                dropout=stec_dropout,
                use_ffn=stec_use_ffn
            )
            for _ in range(stec_num_layers)
        ])
        
        # 5. Final Bilinear Layer (for final output)
        self.final_bilinear = BilinearInteractionLayer(self.embed_per_field, stec_num_heads)
        
        # 6. Batch Normalization for each level's bilinear interaction
        # N+1 levels: N from STEC layers + final
        bilinear_size = stec_num_heads * self.num_fields * self.num_fields
        num_bilinear_levels = stec_num_layers + 1  # N layers + final
        
        self.bilinear_bns = nn.ModuleList([
            nn.BatchNorm1d(bilinear_size)
            for _ in range(num_bilinear_levels)
        ])
        
        # 7. Final MLP
        total_bilinear_dim = bilinear_size * num_bilinear_levels
        
        self.mlp = nn.Sequential()
        prev_dim = total_bilinear_dim
        for i, hidden_dim in enumerate(mlp_hidden_dims):
            self.mlp.add_module(f'fc{i}', nn.Linear(prev_dim, hidden_dim))
            self.mlp.add_module(f'bn{i}', nn.BatchNorm1d(hidden_dim))
            self.mlp.add_module(f'relu{i}', nn.ReLU())
            self.mlp.add_module(f'drop{i}', nn.Dropout(mlp_dropout))
            prev_dim = hidden_dim
        self.mlp.add_module('output', nn.Linear(prev_dim, 1))
        
        # Initialize MLP
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # 8. Loss function
        focal_gamma = config.get('focal_loss_gamma', 0)
        if focal_gamma > 0:
            self._loss_fn: nn.Module = FocalLoss(gamma=focal_gamma)
        else:
            self._loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Forward pass through STEC model.
        
        Args:
            x: Input features [B, Num_Features]
            
        Returns:
            ModelOutput with 'logits' and 'aux_logits'
        """
        B = x.shape[0]
        
        # 1. Get embeddings for each feature
        embeds = []
        for i, feat in enumerate(self.feature_names):
            embeds.append(self.embeddings[feat](x[:, i]))  # [B, feat_dim]
        
        # 2. Concatenate and optionally project
        # [B, total_embed_dim]
        embed_concat = torch.cat(embeds, dim=1)
        
        if self.use_projection:
            embed_concat = self.projection(embed_concat)  # [B, num_fields * embed_per_field]
        
        # Reshape to [B, F, D] for attention
        h = embed_concat.view(B, self.num_fields, self.embed_per_field)
        
        # 3. Collect bilinear interactions from all levels
        bilinear_interactions = []
        
        # 4. Pass through STEC layers
        for i, layer in enumerate(self.stec_layers):
            h, bilinear = layer(h)  # h: [B, F, D], bilinear: [B, H*F*F]
            bilinear_interactions.append(self.bilinear_bns[i](bilinear))
        
        # 5. Final bilinear from last layer output
        final_bilinear = self.final_bilinear(h) # [B, H*F*F]
        bilinear_interactions.append(self.bilinear_bns[-1](final_bilinear))
        
        # 6. Concatenate all bilinear interactions
        fused = torch.cat(bilinear_interactions, dim=1)  # [B, total_bilinear_dim]
        
        # 7. Final prediction
        logits = self.mlp(fused)  # [B, 1]
        
        return {"logits": logits, "aux_logits": None}
    
    def compute_loss(
        self,
        output: ModelOutput,
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss using internal loss function."""
        return self._loss_fn(output["logits"], y_true)
    
    @classmethod
    def model_name(cls) -> str:
        """Return model name for registry."""
        return "stec"
