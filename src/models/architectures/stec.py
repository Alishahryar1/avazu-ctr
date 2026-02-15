"""STEC: See-Through Transformer-based Encoder for CTR Prediction.

This module implements the full STEC architecture which:
1. Extracts bilinear interactions from attention at no additional cost
2. Fuses interactions from multiple levels with direct connections to output
3. Uses multi-head attention with group bilinear interactions

Reference: "STEC: See-Through Transformer-Based Encoder for CTR Prediction"
"""

from typing import Any, Optional, cast
import torch
import torch.nn as nn

from src.config_types import ConfigType
from src.models.utils import get_embedding
from src.models.types import ModelOutput
from src.models.architectures.base import BaseCTRModel

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
        self, vocab_sizes: dict[str, int], feature_names: list[str], config: ConfigType
    ):
        super().__init__()
        self.feature_names = feature_names
        self.num_fields = len(feature_names)

        # Extract config (global at top level, model-specific in config['model'])
        model_config = cast(dict[str, Any], config.get("model", {}))
        embedding_dim: int = config["embedding_dim"]
        stec_num_layers: int = int(model_config.get("stec_num_layers", 2))
        stec_num_heads: int = int(model_config.get("stec_num_heads", 4))
        stec_hidden_dim: int | None = cast(
            int | None, model_config.get("stec_hidden_dim", None)
        )
        stec_dropout: float = float(model_config.get("stec_dropout", 0.1))
        stec_use_ffn: bool = bool(model_config.get("stec_use_ffn", True))
        mlp_hidden_dims: list[int] = cast(
            list[int], model_config.get("stec_mlp_hidden_dims", [256, 128])
        )
        mlp_dropout: float = float(model_config.get("mlp_dropout", 0.1))

        self.stec_num_layers = stec_num_layers
        self.stec_num_heads = stec_num_heads
        # 1. Embedding Layer using get_embedding utility
        self.embeddings = nn.ModuleDict()
        self.feature_dims: dict[str, int] = {}
        total_embed_dim = 0

        for feat in feature_names:
            emb, feat_dim = get_embedding(feat, vocab_sizes.get(feat, 1), config)
            self.embeddings[feat] = emb
            self.feature_dims[feat] = feat_dim
            total_embed_dim += feat_dim

        self.total_embed_dim = total_embed_dim
        self.projection = None
        self.use_projection = False

        # 2. Projection to uniform dimension (required for STEC attention)
        # Check if we need projection (non-uniform dims)
        unique_dims = set(self.feature_dims.values())
        needs_projection = len(unique_dims) > 1

        if needs_projection:
            # Non-uniform embeddings without explicit projection - project to uniform
            self.projection = nn.Linear(
                total_embed_dim, embedding_dim * self.num_fields
            )
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
            self.dim_adjust: Optional[nn.Linear] = nn.Linear(
                self.embed_per_field
                * self.num_fields
                // stec_num_heads
                * stec_num_heads,
                working_dim,
            )
        else:
            self.dim_adjust = None

        # 4. STEC Encoder Layers
        self.stec_layers = nn.ModuleList(
            [
                STECEncoderLayer(
                    embed_dim=self.embed_per_field,
                    num_heads=stec_num_heads,
                    ffn_hidden_dim=stec_hidden_dim,
                    dropout=stec_dropout,
                    use_ffn=stec_use_ffn,
                )
                for _ in range(stec_num_layers)
            ]
        )

        # 5. Final Bilinear Layer (for final output)
        self.final_bilinear = BilinearInteractionLayer(
            self.embed_per_field, stec_num_heads
        )

        # 6. Batch Normalization for each level's bilinear interaction
        # N+1 levels: N from STEC layers + final
        bilinear_size = stec_num_heads * self.num_fields * self.num_fields
        num_bilinear_levels = stec_num_layers + 1  # N layers + final

        self.bilinear_bns = nn.ModuleList(
            [nn.BatchNorm1d(bilinear_size) for _ in range(num_bilinear_levels)]
        )

        # 7. Final MLP
        total_bilinear_dim = bilinear_size * num_bilinear_levels

        self.mlp = nn.Sequential()
        prev_dim = total_bilinear_dim
        for i, hidden_dim in enumerate(mlp_hidden_dims):
            self.mlp.add_module(f"fc{i}", nn.Linear(prev_dim, hidden_dim))
            self.mlp.add_module(f"bn{i}", nn.BatchNorm1d(hidden_dim))
            self.mlp.add_module(f"relu{i}", nn.ReLU())
            self.mlp.add_module(f"drop{i}", nn.Dropout(mlp_dropout))
            prev_dim = hidden_dim
        self.mlp.add_module("output", nn.Linear(prev_dim, 1))

        # Initialize MLP
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 8. Loss function
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

        if self.use_projection and self.projection is not None:
            embed_concat = self.projection(
                embed_concat
            )  # [B, num_fields * embed_per_field]

        # Reshape to [B, F, D] for attention
        h = embed_concat.view(B, self.num_fields, self.embed_per_field)

        # 3. Collect bilinear interactions from all levels
        bilinear_interactions = []

        # 4. Pass through STEC layers
        for i, layer in enumerate(self.stec_layers):
            h, bilinear = layer(h)  # h: [B, F, D], bilinear: [B, H*F*F]
            bilinear_interactions.append(self.bilinear_bns[i](bilinear))

        # 5. Final bilinear from last layer output
        final_bilinear = self.final_bilinear(h)  # [B, H*F*F]
        bilinear_interactions.append(self.bilinear_bns[-1](final_bilinear))

        # 6. Concatenate all bilinear interactions
        fused = torch.cat(bilinear_interactions, dim=1)  # [B, total_bilinear_dim]

        # 7. Final prediction
        logits = self.mlp(fused)  # [B, 1]

        return {"logits": logits, "aux_logits": None}

    def compute_loss(self, output: ModelOutput, y_true: torch.Tensor) -> torch.Tensor:
        """Compute loss using internal loss function."""
        return self._loss_fn(output["logits"], y_true)

    @classmethod
    def model_name(cls) -> str:
        """Return model name for registry."""
        return "stec"
