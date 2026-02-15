"""Base Gated DCN Model for CTR prediction."""

from typing import cast

import torch
import torch.nn as nn

from src.config_types import ConfigType, GatedDCNConfig
from src.models.utils import get_embedding
from src.models.types import ModelOutput
from src.models.architectures.base import BaseCTRModel
from src.models.layers.backbone import build_backbone


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
        model_config = cast(GatedDCNConfig, config["model"])
        embedding_dim = config["embedding_dim"]

        # 1. Embeddings
        self.embeddings = nn.ModuleDict()
        self.feature_dims: dict[str, int] = {}
        total_embed_dim = 0
        for feat in feature_names:
            emb, feat_dim = get_embedding(feat, vocab_sizes.get(feat, 1), config)
            self.embeddings[feat] = emb
            self.feature_dims[feat] = feat_dim
            total_embed_dim += feat_dim

        self.total_embed_dim = total_embed_dim
        self.embedding_dim = embedding_dim
        self.base_embedding_dim = embedding_dim
        self.num_fields = len(feature_names)

        # 2. Shared backbone (LayerNorm -> SENet/FeatureGating -> DCN -> MLP)
        backbone_config_dict: dict[str, object] = {k: v for k, v in model_config.items()}
        self.backbone = build_backbone(
            backbone_config=backbone_config_dict,
            feature_names=feature_names,
            feature_dims=self.feature_dims,
            total_embed_dim=total_embed_dim,
            num_fields=self.num_fields,
        )

        # 3. Output layer
        self.output_layer = nn.Linear(cast(int, self.backbone.output_dim), 1)
        self._loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> ModelOutput:
        embeds = [
            self.embeddings[feat](x[:, i]) for i, feat in enumerate(self.feature_names)
        ]
        backbone_out = self.backbone(embeds)
        logits = self.output_layer(backbone_out)
        return {"logits": logits, "aux_logits": None}

    def compute_loss(self, output: ModelOutput, y_true: torch.Tensor) -> torch.Tensor:
        """Compute loss using internal loss function."""
        return self._loss_fn(output["logits"], y_true)

    @classmethod
    def model_name(cls) -> str:
        """Return model name for registry."""
        return "gated_dcn"
