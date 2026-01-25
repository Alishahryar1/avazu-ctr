"""
Normalized Multi-Head Diversity Model for CTR Prediction.

Combines nGPT (Normalized Transformer) principles with multi-head diversity:
- All embeddings and weights are unit norm normalized
- Hidden state updates use LERP: h ← Norm(h + α(h_block - h))
- Learnable eigen learning rates control block contributions
- Multiple diverse heads with feature bagging for ensemble-like behavior

Reference: "nGPT: Normalized Transformer with Representation Learning
           on the Hypersphere" (Loshchilov et al., ICLR 2025)
"""

import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional

from src.models.architectures.base import BaseCTRModel, ModelOutput
from src.config_types import ConfigType
from src.config_types.normalized_multi_head_diversity_config import (
    NormalizedMultiHeadDiversityConfig,
)
from src.models.losses.diversity_loss import DiversityBCELoss
from src.models.layers.normalized_layers import (
    NormalizedEmbedding,
    NormalizedLinear,
    NormalizedResidualMLP,
    l2_normalize,
    WeightNormalizationCallback,
)
from src.models.layers.logit_gating import LogitGatingLayer


class NormalizedMultiHeadDiversityModel(BaseCTRModel):
    """
    Multi-Head Diversity model with nGPT-style normalization.

    Key features:
    - Normalized embeddings (unit norm vectors on hypersphere)
    - LERP-style residual updates with learnable eigen learning rates
    - Multiple diverse prediction heads with feature bagging
    - Scaled logits for proper softmax temperature control

    The model maintains hidden states on the hypersphere throughout,
    with each processing block (DCN, MLP) contributing controlled
    displacements via the LERP update rule.
    """

    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        feature_names: List[str],
        config: ConfigType,
    ):
        super().__init__()

        # Extract configs
        model_config: NormalizedMultiHeadDiversityConfig = config["model"]  # type: ignore
        backbone_config = model_config["backbone_config"]

        self.feature_names = feature_names
        self.feature_bagging_ratio = model_config.get("feature_bagging_ratio", 1.0)

        # nGPT parameters
        self.use_normalized_embeddings = model_config.get(
            "use_normalized_embeddings", True
        )
        self.use_normalized_weights = model_config.get("use_normalized_weights", True)
        self.use_lerp_updates = model_config.get("use_lerp_updates", True)
        self.normalize_before_head = model_config.get("normalize_before_head", True)

        alpha_init = model_config.get("alpha_init", 0.05)
        alpha_scale = model_config.get("alpha_scale", None)
        su_init = model_config.get("su_init", 1.0)
        sv_init = model_config.get("sv_init", 1.0)

        # --- Embeddings ---
        embedding_dim = config["embedding_dim"]
        self.embedding_dim = embedding_dim

        self.embeddings = nn.ModuleDict()
        self.feature_dims: Dict[str, int] = {}
        total_embed_dim = 0

        for feat in feature_names:
            vocab_size = vocab_sizes.get(feat, 1)

            if self.use_normalized_embeddings:
                emb = NormalizedEmbedding(
                    num_embeddings=vocab_size,
                    embedding_dim=embedding_dim,
                )
            else:
                # Fall back to standard embedding
                from src.models.utils import get_embedding

                emb, _ = get_embedding(feat, vocab_size, config)

            self.embeddings[feat] = emb
            self.feature_dims[feat] = embedding_dim
            total_embed_dim += embedding_dim

        self.total_embed_dim = total_embed_dim
        working_dim = total_embed_dim

        # --- Learnable scaling for concatenated embeddings ---
        # After concatenation, scale to restore proper magnitude
        self.embed_scale = nn.Parameter(torch.ones(working_dim))

        # --- Eigen learning rate for embedding aggregation step ---
        if alpha_scale is None:
            alpha_scale = 1.0 / math.sqrt(working_dim)
        self._alpha_scale = alpha_scale
        self._alpha_init = alpha_init

        # --- Optional SENET (simplified - uses normalized weights internally) ---
        self.use_senet = backbone_config.get("use_senet", False)
        if self.use_senet:
            from src.models.layers.senet import SENetLayer

            senet_dims = [self.feature_dims[f] for f in self.feature_names]
            self.senet = SENetLayer(
                num_fields=len(feature_names),
                feature_dims=senet_dims,
                squeeze_funcs=backbone_config["senet_squeeze_funcs"],
                reduction_ratio=backbone_config["senet_reduction_ratio"],
                hidden_activation=backbone_config["senet_hidden_activation"],
                excitation_activation=backbone_config["senet_excitation_activation"],
                num_groups=backbone_config.get("senet_num_groups", 1),
                reweight_mode=backbone_config.get("senet_reweight_mode", "feature"),
                use_fuse=backbone_config.get("senet_use_fuse", False),
                use_layer_norm=False,  # No LayerNorm in nGPT style
            )
            # Eigen LR for SENET block
            self.alpha_senet = nn.Parameter(torch.full((working_dim,), alpha_scale))

        # --- Optional Feature Gating ---
        self.use_feature_gating = backbone_config.get("use_feature_gating", False)
        if self.use_feature_gating:
            from src.models.layers.gating import FeatureGatingLayer

            self.feature_gating = FeatureGatingLayer(
                input_dim=working_dim,
                gating_activation=backbone_config.get(
                    "feature_gating_activation", "sigmoid"
                ),
                low_rank=backbone_config.get("feature_gating_low_rank", None),
            )
            self.alpha_gating = nn.Parameter(torch.full((working_dim,), alpha_scale))

        # --- Optional DCNv2 ---
        self.use_dcn = backbone_config.get("use_dcn", False)
        if self.use_dcn:
            from src.models.layers.cross_network import DCNv2

            self.dcn = DCNv2(
                input_dim=working_dim,
                num_layers=backbone_config["dcn_num_layers"],
                use_layernorm=False,  # No LayerNorm in nGPT style
                low_rank=backbone_config.get("dcn_low_rank", None),
            )
            self.alpha_dcn = nn.Parameter(torch.full((working_dim,), alpha_scale))

        # --- Backbone MLP (using normalized MLP) ---
        self.use_mlp = bool(backbone_config.get("mlp_hidden_dims", []))
        if self.use_mlp:
            mlp_hidden_dims = backbone_config["mlp_hidden_dims"]

            if self.use_normalized_weights:
                # Use nGPT-style normalized MLP
                # Note: NormalizedResidualMLP maintains dimension, so we use
                # hidden_dims as the intermediate dimensions
                self.mlp = NormalizedResidualMLP(
                    input_dim=working_dim,
                    hidden_dims=[
                        4 * working_dim
                    ] * len(  # Use 4x expansion like nGPT
                        mlp_hidden_dims
                    ),
                    alpha_init=alpha_init,
                    alpha_scale=alpha_scale,
                    su_init=su_init,
                    sv_init=sv_init,
                    dropout=backbone_config.get("mlp_dropout", 0.0),
                    use_glu=True,
                )
                # Output dimension stays the same with normalized MLP
                # We'll add a projection if needed
                if mlp_hidden_dims[-1] != working_dim:
                    self.mlp_proj = NormalizedLinear(
                        working_dim, mlp_hidden_dims[-1], bias=False
                    )
                    working_dim = mlp_hidden_dims[-1]
                else:
                    self.mlp_proj = None
            else:
                # Fall back to standard ResidualMLP
                from src.models.layers.mlp import ResidualMLP

                self.mlp = ResidualMLP(
                    input_dim=working_dim,
                    hidden_dims=mlp_hidden_dims,
                    activation=backbone_config.get("mlp_activation", "relu"),
                    dropout=backbone_config.get("mlp_dropout", 0.0),
                    use_layer_norm=False,
                    use_skip_connections=backbone_config.get(
                        "mlp_use_skip_connections", False
                    ),
                )
                working_dim = mlp_hidden_dims[-1]
                self.mlp_proj = None

        self.backbone_output_dim = working_dim

        # --- Prediction Heads ---
        heads_config = model_config["heads"]
        self.num_heads = len(heads_config)
        self.heads = nn.ModuleList()

        for head_cfg in heads_config:
            head_hidden_dims = head_cfg["hidden_dims"]

            if self.use_normalized_weights and head_hidden_dims:
                # Normalized head MLP
                head_mlp = NormalizedResidualMLP(
                    input_dim=self.backbone_output_dim,
                    hidden_dims=[4 * self.backbone_output_dim] * len(head_hidden_dims),
                    alpha_init=alpha_init,
                    alpha_scale=1.0 / math.sqrt(self.backbone_output_dim),
                    su_init=su_init,
                    sv_init=sv_init,
                    dropout=head_cfg.get("dropout", 0.0),
                    use_glu=True,
                )
                # Output layer with scaling for logits
                head_output = NormalizedLinear(
                    self.backbone_output_dim,
                    1,
                    bias=False,
                    scale_init=1.0,
                    scale_factor=1.0 / math.sqrt(self.backbone_output_dim),
                )
                self.heads.append(nn.ModuleDict({"mlp": head_mlp, "output": head_output}))
            else:
                # Standard head
                from src.models.layers.mlp import ResidualMLP

                head_output_dim = (
                    head_hidden_dims[-1] if head_hidden_dims else self.backbone_output_dim
                )
                self.heads.append(
                    nn.ModuleDict(
                        {
                            "mlp": ResidualMLP(
                                input_dim=self.backbone_output_dim,
                                hidden_dims=head_hidden_dims,
                                activation=head_cfg.get("activation", "relu"),
                                dropout=head_cfg.get("dropout", 0.0),
                                use_layer_norm=False,
                                use_skip_connections=head_cfg.get(
                                    "use_skip_connections", False
                                ),
                            )
                            if head_hidden_dims
                            else nn.Identity(),
                            "output": nn.Linear(head_output_dim, 1),
                        }
                    )
                )

        # --- Logit scaling (temperature control for normalized outputs) ---
        # In nGPT, logits are bounded in [-1, 1], so we need scaling
        self.logit_scale = nn.Parameter(
            torch.full(
                (self.num_heads,), 1.0 / math.sqrt(self.backbone_output_dim)
            )
        )
        self._logit_scale_init = 1.0  # Will be learned to ~60-100 typically

        # --- Feature Bagging Masks ---
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
            diversity_weight=model_config.get("diversity_weight", 0.1)
        )

        # Create weight normalization callback for use after optimizer step
        self._weight_norm_callback = WeightNormalizationCallback(self)

    def _get_alpha(self, alpha_param: nn.Parameter) -> torch.Tensor:
        """Get actual alpha using init/scale pattern."""
        return torch.abs(alpha_param) * (self._alpha_init / self._alpha_scale)

    def _lerp_update(
        self, h: torch.Tensor, h_block: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        """Apply LERP update: h ← Norm(h + α(h_block - h))"""
        if not self.use_lerp_updates:
            return h_block

        h_norm = l2_normalize(h, dim=-1)
        h_block_norm = l2_normalize(h_block, dim=-1)
        h_updated = h_norm + alpha * (h_block_norm - h_norm)
        return l2_normalize(h_updated, dim=-1)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        batch_size = x.size(0)

        # 1. Get embeddings
        embeds_list = []
        for i, feat in enumerate(self.feature_names):
            emb = self.embeddings[feat](x[:, i])  # [B, embed_dim]
            embeds_list.append(emb)

        # 2. Process each head
        head_logits = []

        for head_idx, head in enumerate(self.heads):
            # Apply feature bagging mask
            if self.feature_bagging_ratio < 1.0:
                mask = getattr(self, f"head_mask_{head_idx}")
                current_embeds = [emb * mask[i] for i, emb in enumerate(embeds_list)]
            else:
                current_embeds = embeds_list

            # Concatenate embeddings
            h = torch.cat(current_embeds, dim=1)  # [B, total_embed_dim]

            # Apply embedding scale
            h = h * self.embed_scale

            # Normalize to hypersphere
            h = l2_normalize(h, dim=-1)

            # --- Backbone processing with LERP updates ---

            # SENET
            if self.use_senet:
                h_senet = self.senet(current_embeds)
                alpha = self._get_alpha(self.alpha_senet)
                h = self._lerp_update(h, h_senet, alpha)

            # Feature Gating
            if self.use_feature_gating:
                h_gating = self.feature_gating(h)
                alpha = self._get_alpha(self.alpha_gating)
                h = self._lerp_update(h, h_gating, alpha)

            # DCN
            if self.use_dcn:
                h_dcn = self.dcn(h)
                alpha = self._get_alpha(self.alpha_dcn)
                h = self._lerp_update(h, h_dcn, alpha)

            # MLP (already has internal LERP updates if normalized)
            if self.use_mlp:
                if self.use_normalized_weights:
                    h = self.mlp(h, use_lerp=self.use_lerp_updates)
                    if self.mlp_proj is not None:
                        h = self.mlp_proj(h)
                        h = l2_normalize(h, dim=-1)
                else:
                    h = self.mlp(h)
                    if self.use_lerp_updates:
                        h = l2_normalize(h, dim=-1)

            # --- Head prediction ---
            if self.normalize_before_head:
                h = l2_normalize(h, dim=-1)

            # Head MLP
            if isinstance(head["mlp"], NormalizedResidualMLP):
                h_head = head["mlp"](h, use_lerp=self.use_lerp_updates)
            else:
                h_head = head["mlp"](h)

            # Output logits with scaling
            logits = head["output"](h_head)

            # Apply logit scaling (for proper temperature)
            actual_scale = (
                torch.abs(self.logit_scale[head_idx])
                * self._logit_scale_init
                / (1.0 / math.sqrt(self.backbone_output_dim))
            )
            logits = logits * actual_scale

            head_logits.append(logits)

        # Stack: [K, B, 1]
        stacked_logits = torch.stack(head_logits, dim=0)

        # 3. Aggregate
        if self.aggregation_method == "gated":
            logits_for_gate = stacked_logits.squeeze(-1).permute(1, 0)  # [B, K]
            aggregated_logits = self.logit_gate(logits_for_gate)  # [B, 1]
        else:
            aggregated_logits = stacked_logits.mean(dim=0)  # [B, 1]

        return {
            "logits": aggregated_logits,
            "aux_logits": stacked_logits,
        }

    def compute_loss(self, output: ModelOutput, y_true: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(output["aux_logits"], y_true)

    def normalize_weights(self):
        """
        Normalize all weight matrices after optimizer step.

        Call this after optimizer.step() to maintain normalized weights.
        This is crucial for nGPT-style training.
        """
        self._weight_norm_callback()

    @classmethod
    def model_name(cls) -> str:
        return "normalized_multi_head_diversity"
