from typing import cast

import torch
import torch.nn as nn

from src.models.architectures.base import BaseCTRModel
from src.models.types import ModelOutput
from src.config_types import ConfigType, MultiHeadDiversityConfig
from src.models.losses.diversity_loss import DiversityBCELoss
from src.models.layers.backbone import build_backbone
from src.models.layers.mlp import ResidualMLP
from src.models.layers.logit_gating import LogitGatingLayer
from src.models.utils import get_embedding


class MultiHeadDiversityModel(BaseCTRModel):
    """
    Single model that mimics an ensemble using a shared backbone and
    multiple diverse heads. Supports Feature Bagging.
    """

    def __init__(
        self, vocab_sizes: dict[str, int], feature_names: list[str], config: ConfigType
    ):
        super().__init__()

        model_config: MultiHeadDiversityConfig = cast(
            MultiHeadDiversityConfig, config["model"]
        )
        backbone_config_dict = model_config["backbone_config"]

        self.feature_names = feature_names
        self.feature_bagging_ratio = model_config.get("feature_bagging_ratio", 1.0)

        # 1. Embeddings
        self.embeddings = nn.ModuleDict()
        self.feature_dims: dict[str, int] = {}
        total_embed_dim = 0
        for feat in feature_names:
            emb, feat_dim = get_embedding(feat, vocab_sizes.get(feat, 1), config)
            self.embeddings[feat] = emb
            self.feature_dims[feat] = feat_dim
            total_embed_dim += feat_dim

        self.embedding_dim = config["embedding_dim"]

        # 2. Shared backbone
        self.backbone = build_backbone(
            backbone_config=backbone_config_dict,
            feature_names=feature_names,
            feature_dims=self.feature_dims,
            total_embed_dim=total_embed_dim,
            num_fields=len(feature_names),
        )
        self.input_dim = self.backbone.output_dim

        # --- Heads Initialization ---
        heads_config = model_config["heads"]
        self.num_heads = len(heads_config)
        self.heads = nn.ModuleList()

        for head_cfg in heads_config:
            head_hidden_dims = head_cfg["hidden_dims"]
            head_output_dim = (
                head_hidden_dims[-1] if head_hidden_dims else self.input_dim
            )
            self.heads.append(
                nn.Sequential(
                    ResidualMLP(
                        input_dim=cast(int, self.input_dim),
                        hidden_dims=head_hidden_dims,
                        activation=head_cfg["activation"],
                        dropout=head_cfg["dropout"],
                        use_layer_norm=head_cfg["use_layer_norm"],
                        use_skip_connections=head_cfg["use_skip_connections"],
                    ),
                    nn.Linear(cast(int, head_output_dim), 1),
                )
            )

        # --- Pre-generate Feature Bagging Masks ---
        if self.feature_bagging_ratio < 1.0:
            for i in range(self.num_heads):
                mask = torch.bernoulli(
                    torch.full((len(self.feature_names),), self.feature_bagging_ratio)
                )
                self.register_buffer(f"head_mask_{i}", mask)

        # --- Aggregation ---
        self.aggregation_method = model_config.get("aggregation_method", "mean")
        if self.aggregation_method == "gated":
            gating_hidden_dim = model_config.get("gating_hidden_dim", None)
            self.logit_gate = LogitGatingLayer(
                num_inputs=self.num_heads, hidden_dim=gating_hidden_dim
            )

        # --- Loss ---
        self.loss_fn = DiversityBCELoss(
            diversity_weight=model_config["diversity_weight"]
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        embeds_list = [
            self.embeddings[feat](x[:, i]) for i, feat in enumerate(self.feature_names)
        ]

        head_logits = []
        for head_idx, head in enumerate(self.heads):
            if self.feature_bagging_ratio < 1.0:
                mask = getattr(self, f"head_mask_{head_idx}")
                current_head_embeds = [
                    emb * mask[i] for i, emb in enumerate(embeds_list)
                ]
            else:
                current_head_embeds = embeds_list

            backbone_out = self.backbone(current_head_embeds)
            head_logits.append(head(backbone_out))

        # Stack: [K, Batch, 1]
        stacked_logits = torch.stack(head_logits, dim=0)

        # 3. Aggregate for final prediction
        if self.aggregation_method == "gated":
            # stacked_logits: [K, Batch, 1] -> [Batch, K]
            logits_for_gate = stacked_logits.squeeze(-1).permute(1, 0)
            aggregated_logits = self.logit_gate(logits_for_gate)  # [Batch, 1]
        else:
            aggregated_logits = stacked_logits.mean(dim=0)

        return {
            "logits": aggregated_logits,
            "aux_logits": stacked_logits,  # Pass stacked logits to loss
        }

    def compute_loss(self, output: ModelOutput, y_true: torch.Tensor) -> torch.Tensor:
        # Use the custom diversity loss on the stacked logits
        return self.loss_fn(output["aux_logits"], y_true)

    @classmethod
    def model_name(cls) -> str:
        return "multi_head_diversity"
