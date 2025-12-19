"""Base Gated DCN Model for CTR prediction."""

from typing import cast
import torch
import torch.nn as nn

from config import ConfigType
from src.config_types import GatedDCNConfig
from src.models.utils import get_embedding
from src.models.types import ModelOutput
from src.models.architectures.base import BaseCTRModel

from src.models.layers.senet import SENetLayer
from src.models.layers.gating import FeatureGatingLayer
from src.models.layers.cross_network import DCNv2
from src.models.layers.mlp import ResidualMLP


class GatedDCNModel(BaseCTRModel):
    """
    Gated DCN Model for CTR prediction.

    Args:
        vocab_sizes: Dictionary mapping feature names to vocabulary sizes.
        feature_names: List of feature names in order.
        config: Configuration dictionary with model hyperparameters.
    """

    def __init__(
        self, vocab_sizes: dict[str, int], feature_names: list[str], config: ConfigType
    ):
        super().__init__()
        self.feature_names = feature_names

        # Extract config values (global settings at top level, model-specific in config['model'])
        model_config = cast(GatedDCNConfig, config["model"])
        embedding_dim = config["embedding_dim"]
        use_dcn = model_config["use_dcn"]
        dcn_num_layers = model_config["dcn_num_layers"]
        dcn_use_layernorm = model_config["dcn_use_layernorm"]
        dcn_low_rank = model_config["dcn_low_rank"]
        use_senet = model_config["use_senet"]
        senet_squeeze_funcs = model_config["senet_squeeze_funcs"]
        senet_reduction_ratio = model_config["senet_reduction_ratio"]
        senet_hidden_activation = model_config["senet_hidden_activation"]
        senet_excitation_activation = model_config["senet_excitation_activation"]
        use_feature_gating = model_config["use_feature_gating"]
        feature_gating_activation = model_config["feature_gating_activation"]
        mlp_hidden_dims = model_config["mlp_hidden_dims"]
        mlp_dropout = model_config["mlp_dropout"]
        use_layer_norm = model_config["use_layer_norm"]
        mlp_activation = model_config["mlp_activation"]

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

        # 1. Embedding Layer using get_embedding utility
        self.embeddings = nn.ModuleDict()
        self.feature_dims: dict[str, int] = {}  # Track dimension per feature
        total_embed_dim = 0

        for feat in feature_names:
            emb, feat_dim = get_embedding(feat, vocab_sizes[feat], config)
            self.embeddings[feat] = emb
            self.feature_dims[feat] = feat_dim
            total_embed_dim += feat_dim

        # Store dimensions for later use
        self.total_embed_dim = total_embed_dim
        working_dim = total_embed_dim
        # For SENET compatibility
        self.embedding_dim = embedding_dim

        # Layer norm after embeddings (before or after projection)
        if use_layer_norm:
            self.embed_ln = nn.LayerNorm(working_dim)

        if use_senet:
            # SENet operates on variable-dimension embeddings
            senet_dims = [
                self.feature_dims[f] for f in self.feature_names
            ]  # maintain ordering
            self.senet = SENetLayer(
                num_fields=self.num_fields,
                feature_dims=senet_dims,
                squeeze_funcs=senet_squeeze_funcs,
                reduction_ratio=senet_reduction_ratio,
                hidden_activation=senet_hidden_activation,
                excitation_activation=senet_excitation_activation,
            )

        # 2b. Feature Gating Layer - Optional (alternative to SENET)
        if use_feature_gating:
            feature_gating_low_rank = model_config["feature_gating_low_rank"]
            self.feature_gating = FeatureGatingLayer(
                input_dim=working_dim,
                gating_activation=feature_gating_activation,
                low_rank=feature_gating_low_rank,
            )

        # 4. DCNv2 - Optional (supports low-rank decomposition)
        if use_dcn:
            self.dcn = DCNv2(
                working_dim,
                num_layers=dcn_num_layers,
                use_layernorm=dcn_use_layernorm,
                low_rank=dcn_low_rank,
            )

        # 5. Enhanced MLP with LayerNorm, configurable activation, and optional skip connections
        mlp_use_skip_connections = model_config["mlp_use_skip_connections"]
        self.mlp = ResidualMLP(
            input_dim=working_dim,
            hidden_dims=mlp_hidden_dims,
            output_dim=1,
            activation=mlp_activation,
            dropout=mlp_dropout,
            use_layer_norm=use_layer_norm,
            use_skip_connections=mlp_use_skip_connections,
        )

        # 6. Internal loss function
        self._loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> ModelOutput:
        # x shape: [Batch, Num_Features]

        # Flatten inputs into a single dense vector
        embeds = []
        for i, feat in enumerate(self.feature_names):
            embeds.append(self.embeddings[feat](x[:, i]))

        # Concatenate: [Batch, Total_Embed_Dim]
        dnn_input = torch.cat(embeds, dim=1)

        # Apply layer norm to embeddings
        if self.use_layer_norm:
            dnn_input = self.embed_ln(dnn_input)

        # Apply SENET (Feature Importance Reweighting) - Optional
        if self.use_senet:
            # Use variable-dimension embeddings directly
            dnn_input = self.senet(embeds)

        # Apply Feature Gating - Optional (alternative to SENET)
        if self.use_feature_gating:
            dnn_input = self.feature_gating(dnn_input)

        # Apply Cross Network (Interactions) - Optional
        if self.use_dcn:
            dnn_input = self.dcn(dnn_input)

        # Final Prediction (no sigmoid here - we'll use BCEWithLogitsLoss)
        logits = self.mlp(dnn_input)
        return {"logits": logits, "aux_logits": None}

    def compute_loss(self, output: ModelOutput, y_true: torch.Tensor) -> torch.Tensor:
        """Compute loss using internal loss function."""
        return self._loss_fn(output["logits"], y_true)

    @classmethod
    def model_name(cls) -> str:
        """Return model name for registry."""
        return "gated_dcn"
