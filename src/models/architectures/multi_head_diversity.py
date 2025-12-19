import torch
import torch.nn as nn
from typing import List, Dict

from src.models.architectures.base import BaseCTRModel, ModelOutput
from src.config_types import ConfigType, MultiHeadDiversityConfig
from src.models.losses.diversity_loss import DiversityBCELoss
from src.models.layers.mlp import ResidualMLP


class MultiHeadDiversityModel(BaseCTRModel):
    """
    Single model that mimics an ensemble using a shared backbone and
    multiple diverse heads. Supports Feature Bagging.
    """

    def __init__(
        self, vocab_sizes: Dict[str, int], feature_names: List[str], config: ConfigType
    ):
        super().__init__()

        # Extract specific config
        model_config: MultiHeadDiversityConfig = config["model"]  # type: ignore
        backbone_config_dict = model_config["backbone_config"]

        self.feature_names = feature_names
        self.feature_bagging_ratio = model_config.get("feature_bagging_ratio", 1.0)

        # --- Shared Backbone Initialization (Native) ---
        # 1. Embeddings
        embedding_dim = config["embedding_dim"]
        self.embeddings = nn.ModuleDict()
        self.feature_dims: Dict[str, int] = {}
        total_embed_dim = 0

        # Import utils locally to avoid circular imports if any
        from src.models.utils import get_embedding

        for feat in feature_names:
            emb, feat_dim = get_embedding(feat, vocab_sizes[feat], config)
            self.embeddings[feat] = emb
            self.feature_dims[feat] = feat_dim
            total_embed_dim += feat_dim

        projection_dim = config.get("embedding_projection_dim", None)
        self.use_projection = projection_dim is not None

        # 2. Projection (Optional)
        if self.use_projection and projection_dim is not None:
            self.projection = nn.Linear(total_embed_dim, projection_dim)
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
            working_dim = projection_dim
            self.embedding_dim = projection_dim // len(feature_names)
        else:
            working_dim = total_embed_dim
            self.embedding_dim = embedding_dim

        # 3. Layer Norm
        self.use_layer_norm = backbone_config_dict.get("use_layer_norm", False)
        if self.use_layer_norm:
            self.embed_ln = nn.LayerNorm(working_dim)

        # 4. SENET / Feature Gating (Optional)
        self.use_senet = backbone_config_dict.get("use_senet", False)
        self.use_feature_gating = backbone_config_dict.get("use_feature_gating", False)

        if self.use_senet:
            from src.models.layers.senet import SENetLayer

            if self.use_projection and projection_dim is not None:
                senet_dims = self.embedding_dim
            else:
                senet_dims = [self.feature_dims[f] for f in self.feature_names]

            self.senet = SENetLayer(
                num_fields=len(feature_names),
                feature_dims=senet_dims,
                squeeze_funcs=backbone_config_dict["senet_squeeze_funcs"],
                reduction_ratio=backbone_config_dict["senet_reduction_ratio"],
                hidden_activation=backbone_config_dict["senet_hidden_activation"],
                excitation_activation=backbone_config_dict[
                    "senet_excitation_activation"
                ],
                # Handle missing keys safely if config dict is partial, though ideally typed
                num_groups=backbone_config_dict.get("senet_num_groups", 1),
                reweight_mode=backbone_config_dict.get(
                    "senet_reweight_mode", "feature"
                ),
                use_fuse=backbone_config_dict.get("senet_use_fuse", False),
                use_layer_norm=backbone_config_dict.get("senet_use_layer_norm", False),
            )

        if self.use_feature_gating:
            from src.models.layers.gating import FeatureGatingLayer

            self.feature_gating = FeatureGatingLayer(
                input_dim=working_dim,
                gating_activation=backbone_config_dict.get(
                    "feature_gating_activation", "sigmoid"
                ),
                low_rank=backbone_config_dict.get("feature_gating_low_rank", None),
            )

        # 5. DCNv2 (Optional)
        self.use_dcn = backbone_config_dict.get("use_dcn", False)
        if self.use_dcn:
            from src.models.layers.cross_network import DCNv2

            self.dcn = DCNv2(
                input_dim=working_dim,
                num_layers=backbone_config_dict["dcn_num_layers"],
                use_layernorm=backbone_config_dict["dcn_use_layernorm"],
                low_rank=backbone_config_dict.get("dcn_low_rank", None),
            )

        self.input_dim = working_dim  # Final dimension after backbone processing

        # --- Heads Initialization ---
        heads_config = model_config["heads"]
        self.num_heads = len(heads_config)
        self.heads = nn.ModuleList()

        for head_cfg in heads_config:
            self.heads.append(
                ResidualMLP(
                    input_dim=self.input_dim,
                    hidden_dims=head_cfg["hidden_dims"],
                    output_dim=1,
                    activation=head_cfg["activation"],
                    dropout=head_cfg["dropout"],
                    use_layer_norm=head_cfg["use_layer_norm"],
                    use_skip_connections=head_cfg["use_skip_connections"],
                )
            )

        # --- Pre-generate Feature Bagging Masks ---
        if self.feature_bagging_ratio < 1.0:
            for i in range(self.num_heads):
                mask = torch.bernoulli(
                    torch.full((len(self.feature_names),), self.feature_bagging_ratio)
                )
                self.register_buffer(f"head_mask_{i}", mask)

        # --- Loss ---
        self.loss_fn = DiversityBCELoss(
            diversity_weight=model_config["diversity_weight"]
        )

    def shared_backbone_forward(self, dnn_input: torch.Tensor) -> torch.Tensor:
        """Applies the shared backbone layers (Projection, SENET/Gating, DCN)."""
        # Apply optional projection
        if self.use_projection:
            dnn_input = self.projection(dnn_input)

        # Apply layer norm
        if self.use_layer_norm:
            dnn_input = self.embed_ln(dnn_input)

        # Apply SENET
        if self.use_senet and self.use_projection:
            # After projection: split into uniform chunks for SENET
            senet_input = list(dnn_input.split(self.embedding_dim, dim=1))
            dnn_input = self.senet(senet_input)

        # Apply Feature Gating (alternative to SENET, works on concatenated tensor)
        if self.use_feature_gating:
            dnn_input = self.feature_gating(dnn_input)

        # Apply DCN
        if self.use_dcn:
            dnn_input = self.dcn(dnn_input)

        return dnn_input

    def forward(self, x: torch.Tensor) -> ModelOutput:
        # 1. Get Embeddings [Batch, Num_Features, Embed_Dim] (Implicitly represented as list of tensors)
        embeds_list = []
        for i, feat in enumerate(self.feature_names):
            embeds_list.append(self.embeddings[feat](x[:, i]))

        # 2. Process Heads with Feature Bagging
        head_logits = []

        for head_idx, head in enumerate(self.heads):
            current_head_embeds = []
            if self.feature_bagging_ratio < 1.0:
                # Use pre-generated mask for this head
                mask = getattr(self, f"head_mask_{head_idx}")
                current_head_embeds = [
                    emb * mask[i] for i, emb in enumerate(embeds_list)
                ]
            else:
                current_head_embeds = embeds_list

            # --- Backbone Processing for this Head ---
            # Now we have the masked embeddings.
            dnn_input = torch.cat(current_head_embeds, dim=1)

            # Rest of backbone (Projection -> SENET(uniform) -> Gating -> DCN)
            # Apply Projection
            if self.use_projection:
                dnn_input = self.projection(dnn_input)

            # Apply Layer Norm
            if self.use_layer_norm:
                dnn_input = self.embed_ln(dnn_input)

            # Apply SENET (Uniform)
            if self.use_senet:
                if self.use_projection:
                    senet_input = list(dnn_input.split(self.embedding_dim, dim=1))
                    dnn_input = self.senet(senet_input)
                else:
                    # SENet expects a list of embeddings, not concatenated tensor
                    dnn_input = self.senet(current_head_embeds)

            # Apply Feature Gating
            if self.use_feature_gating:
                dnn_input = self.feature_gating(dnn_input)

            # Apply DCN
            if self.use_dcn:
                dnn_input = self.dcn(dnn_input)

            # --- Head Prediction ---
            head_logits.append(head(dnn_input))

        # Stack: [K, Batch, 1]
        stacked_logits = torch.stack(head_logits, dim=0)

        # 3. Aggregate (Mean) for final prediction
        avg_logits = stacked_logits.mean(dim=0)

        return {
            "logits": avg_logits,
            "aux_logits": stacked_logits,  # Pass stacked logits to loss
        }

    def compute_loss(self, output: ModelOutput, y_true: torch.Tensor) -> torch.Tensor:
        # Use the custom diversity loss on the stacked logits
        return self.loss_fn(output["aux_logits"], y_true)

    @classmethod
    def model_name(cls) -> str:
        return "multi_head_diversity"
