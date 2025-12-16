"""Base Gated DCN Model for CTR prediction."""
import torch
import torch.nn as nn

from src.config.config import ConfigType
from src.models.utils import compute_embedding_dim
from src.models.layers.attention import SENetLayer
from src.models.layers.gating import FeatureGatingLayer
from src.models.layers.cross_network import DCNv2
from src.models.layers.mlp import ResidualMLP


class GatedDCNModel(nn.Module):
    """
    Gated DCN Model for CTR prediction.

    Args:
        vocab_sizes: Dictionary mapping feature names to vocabulary sizes.
        feature_names: List of feature names in order.
        config: Configuration dictionary with model hyperparameters.
    """
    def __init__(self, vocab_sizes: dict[str, int], feature_names: list[str], config: ConfigType):
        super().__init__()
        self.feature_names = feature_names

        # Extract config values
        embedding_dim = config['embedding_dim']
        use_dcn = config['use_dcn']
        dcn_num_layers = config['dcn_num_layers']
        dcn_use_layernorm = config['dcn_use_layernorm']
        dcn_low_rank = config['dcn_low_rank']
        use_senet = config['use_senet']
        senet_squeeze_funcs = config['senet_squeeze_funcs']
        senet_reduction_ratio = config['senet_reduction_ratio']
        senet_activation = config['senet_activation']
        use_feature_gating = config['use_feature_gating']
        feature_gating_activation = config['feature_gating_activation']
        mlp_hidden_dims = config['mlp_hidden_dims']
        mlp_dropout = config['mlp_dropout']
        use_layer_norm = config['use_layer_norm']
        mlp_activation = config['mlp_activation']

        # Validate mutual exclusivity
        if use_senet and use_feature_gating:
            raise ValueError(
                "Cannot enable both SENET and Feature Gating. "
                "Set either 'use_senet' or 'use_feature_gating' to False."
            )

        self.use_layer_norm = use_layer_norm
        self.use_dcn = use_dcn
        self.use_senet = use_senet
        self.use_feature_gating = use_feature_gating
        self.num_fields = len(feature_names)
        self.base_embedding_dim = embedding_dim  # Base/fallback dimension

        # Track per-feature embedding dimensions for variable embeddings
        use_variable_embeddings = config.get('use_variable_embeddings', False)
        feature_overrides = config.get('feature_embedding_overrides', {})
        projection_dim = config.get('embedding_projection_dim', None)

        # 1. Embedding Layer with variable dimensions per feature
        self.embeddings = nn.ModuleDict()
        self.feature_dims: dict[str, int] = {}  # Track dimension per feature
        total_embed_dim = 0

        for feat in feature_names:
            # Check for manual override first
            if feat in feature_overrides and 'embedding_dim' in feature_overrides[feat]:
                feat_dim = feature_overrides[feat]['embedding_dim']
            elif use_variable_embeddings:
                # Compute dimension based on cardinality
                feat_dim = compute_embedding_dim(vocab_sizes[feat], config)
            else:
                feat_dim = embedding_dim

            self.feature_dims[feat] = feat_dim
            emb = nn.Embedding(vocab_sizes[feat], feat_dim)
            # Xavier initialization for embeddings
            nn.init.xavier_uniform_(emb.weight)
            self.embeddings[feat] = emb
            total_embed_dim += feat_dim

        # Store dimensions for later use
        self.total_embed_dim = total_embed_dim
        self.use_projection = projection_dim is not None

        # 2. Optional Projection Layer to unify dimensions
        if self.use_projection and projection_dim is not None:
            self.projection = nn.Linear(total_embed_dim, projection_dim)
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
            working_dim = projection_dim
            # For SENET, we need uniform embedding dim after projection
            self.embedding_dim = projection_dim // self.num_fields
        else:
            working_dim = total_embed_dim
            # For SENET compatibility: only works with uniform embeddings
            # When variable embeddings without projection, SENET is disabled
            self.embedding_dim = embedding_dim if not use_variable_embeddings else embedding_dim

        # Layer norm after embeddings (before or after projection)
        if use_layer_norm:
            self.embed_ln = nn.LayerNorm(working_dim)

        # 3. SENET (Squeeze-and-Excitation) - Optional
        # Note: SENET requires uniform embedding dimensions per field
        if use_senet:
            if use_variable_embeddings and not self.use_projection:
                raise ValueError(
                    "SENET requires uniform embedding dimensions. "
                    "Either disable 'use_variable_embeddings', enable 'embedding_projection_dim', "
                    "or disable 'use_senet'."
                )
            senet_embed_dim = self.embedding_dim if self.embedding_dim else embedding_dim
            self.senet = SENetLayer(
                num_fields=self.num_fields,
                embedding_dim=senet_embed_dim,
                squeeze_funcs=senet_squeeze_funcs,
                reduction_ratio=senet_reduction_ratio,
                excitation_activation=senet_activation
            )

        # 2b. Feature Gating Layer - Optional (alternative to SENET)
        if use_feature_gating:
            feature_gating_low_rank = config['feature_gating_low_rank']
            self.feature_gating = FeatureGatingLayer(
                input_dim=working_dim,
                gating_activation=feature_gating_activation,
                low_rank=feature_gating_low_rank
            )

        # 4. DCNv2 - Optional (supports low-rank decomposition)
        if use_dcn:
            self.dcn = DCNv2(working_dim, num_layers=dcn_num_layers, use_layernorm=dcn_use_layernorm, low_rank=dcn_low_rank)


        # 5. Enhanced MLP with LayerNorm, configurable activation, and optional skip connections
        mlp_use_skip_connections = config['mlp_use_skip_connections']
        self.mlp = ResidualMLP(
            input_dim=working_dim,
            hidden_dims=mlp_hidden_dims,
            output_dim=1,
            activation=mlp_activation,
            dropout=mlp_dropout,
            use_layer_norm=use_layer_norm,
            use_skip_connections=mlp_use_skip_connections
        )

    def forward(self, x):
        # x shape: [Batch, Num_Features]

        # Flatten inputs into a single dense vector
        embeds = []
        for i, feat in enumerate(self.feature_names):
            embeds.append(self.embeddings[feat](x[:, i]))

        # Concatenate: [Batch, Total_Embed_Dim]
        dnn_input = torch.cat(embeds, dim=1)

        # Apply optional projection to unify dimensions
        if self.use_projection:
            dnn_input = self.projection(dnn_input)

        # Apply layer norm to embeddings
        if self.use_layer_norm:
            dnn_input = self.embed_ln(dnn_input)

        # Apply SENET (Feature Importance Reweighting) - Optional
        if self.use_senet:
            dnn_input = self.senet(dnn_input)

        # Apply Feature Gating - Optional (alternative to SENET)
        if self.use_feature_gating:
            dnn_input = self.feature_gating(dnn_input)

        # Apply Cross Network (Interactions) - Optional
        if self.use_dcn:
            dnn_input = self.dcn(dnn_input)

        # Final Prediction (no sigmoid here - we'll use BCEWithLogitsLoss)
        logits = self.mlp(dnn_input)
        return logits  # Return raw logits for numerical stability
