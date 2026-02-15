"""Shared CTR backbone: embeddings -> LayerNorm -> SENet/FeatureGating -> DCN -> MLP."""

from typing import Any

import torch
import torch.nn as nn

from .cross_network import DCNv2
from .gating import FeatureGatingLayer
from .mlp import ResidualMLP
from .senet import SENetLayer


def _get_backbone_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract backbone config with defaults. Accepts GatedDCNConfig or backbone_config dict."""
    return {
        "use_layer_norm": config.get("use_layer_norm", False),
        "use_senet": config.get("use_senet", False),
        "use_feature_gating": config.get("use_feature_gating", False),
        "senet_squeeze_funcs": config.get("senet_squeeze_funcs", ["mean"]),
        "senet_reduction_ratio": config.get("senet_reduction_ratio", 3),
        "senet_hidden_activation": config.get("senet_hidden_activation", "relu"),
        "senet_excitation_activation": config.get(
            "senet_excitation_activation", "sigmoid"
        ),
        "senet_num_groups": config.get("senet_num_groups", 1),
        "senet_reweight_mode": config.get("senet_reweight_mode", "feature"),
        "senet_use_fuse": config.get("senet_use_fuse", False),
        "senet_use_layer_norm": config.get("senet_use_layer_norm", False),
        "feature_gating_activation": config.get("feature_gating_activation", "sigmoid"),
        "feature_gating_low_rank": config.get("feature_gating_low_rank"),
        "use_dcn": config.get("use_dcn", False),
        "dcn_num_layers": config.get("dcn_num_layers", 2),
        "dcn_use_layernorm": config.get("dcn_use_layernorm", False),
        "dcn_low_rank": config.get("dcn_low_rank"),
        "mlp_hidden_dims": config.get("mlp_hidden_dims", []),
        "mlp_activation": config.get("mlp_activation", "relu"),
        "mlp_dropout": config.get("mlp_dropout", 0.1),
        "mlp_use_skip_connections": config.get("mlp_use_skip_connections", False),
    }


def build_backbone(
    backbone_config: dict[str, Any],
    feature_names: list[str],
    feature_dims: dict[str, int],
    total_embed_dim: int,
    num_fields: int,
) -> nn.Module:
    """
    Build shared CTR backbone (LayerNorm -> SENet/FeatureGating -> DCN -> MLP).

    Does not include embeddings - caller provides those. Returns a module with
    forward(embeds: list[Tensor]) -> Tensor.
    """
    cfg = _get_backbone_config(backbone_config)

    if cfg["use_senet"] and cfg["use_feature_gating"]:
        raise ValueError(
            "Cannot enable both SENET and Feature Gating. "
            "Set either 'use_senet' or 'use_feature_gating' to False."
        )

    return CTRBackbone(
        total_embed_dim=total_embed_dim,
        num_fields=num_fields,
        feature_dims=[feature_dims[f] for f in feature_names],
        **cfg,
    )


class CTRBackbone(nn.Module):
    """
    Shared backbone: LayerNorm -> SENet/FeatureGating -> DCN -> MLP.

    Forward accepts list of embedding tensors (for SENet) or caller can pass
    pre-concatenated tensor via forward_from_concat.
    """

    def __init__(
        self,
        total_embed_dim: int,
        num_fields: int,
        feature_dims: list[int],
        use_layer_norm: bool = False,
        use_senet: bool = False,
        use_feature_gating: bool = False,
        senet_squeeze_funcs: list[str] | None = None,
        senet_reduction_ratio: int = 3,
        senet_hidden_activation: str = "relu",
        senet_excitation_activation: str = "sigmoid",
        senet_num_groups: int = 1,
        senet_reweight_mode: str = "feature",
        senet_use_fuse: bool = False,
        senet_use_layer_norm: bool = False,
        feature_gating_activation: str = "sigmoid",
        feature_gating_low_rank: int | None = None,
        use_dcn: bool = False,
        dcn_num_layers: int = 2,
        dcn_use_layernorm: bool = False,
        dcn_low_rank: int | None = None,
        mlp_hidden_dims: list[int] | None = None,
        mlp_activation: str = "relu",
        mlp_dropout: float = 0.1,
        mlp_use_skip_connections: bool = False,
    ):
        super().__init__()
        self.total_embed_dim = total_embed_dim
        self.num_fields = num_fields
        self.use_layer_norm = use_layer_norm
        self.use_senet = use_senet
        self.use_feature_gating = use_feature_gating
        self.use_dcn = use_dcn
        self.use_mlp = bool(mlp_hidden_dims)
        mlp_hidden_dims = mlp_hidden_dims or []

        if use_layer_norm:
            self.embed_ln = nn.LayerNorm(total_embed_dim)
        else:
            self.embed_ln = None

        if use_senet:
            self.senet = SENetLayer(
                num_fields=num_fields,
                feature_dims=feature_dims,
                squeeze_funcs=senet_squeeze_funcs or ["mean"],
                reduction_ratio=senet_reduction_ratio,
                hidden_activation=senet_hidden_activation,
                excitation_activation=senet_excitation_activation,
                num_groups=senet_num_groups,
                reweight_mode=senet_reweight_mode,
                use_fuse=senet_use_fuse,
                use_layer_norm=senet_use_layer_norm,
            )
        else:
            self.senet = None

        if use_feature_gating:
            self.feature_gating = FeatureGatingLayer(
                input_dim=total_embed_dim,
                gating_activation=feature_gating_activation,
                low_rank=feature_gating_low_rank,
            )
        else:
            self.feature_gating = None

        if use_dcn:
            self.dcn = DCNv2(
                total_embed_dim,
                num_layers=dcn_num_layers,
                use_layernorm=dcn_use_layernorm,
                low_rank=dcn_low_rank,
            )
        else:
            self.dcn = None

        working_dim = mlp_hidden_dims[-1] if mlp_hidden_dims else total_embed_dim
        self.mlp = ResidualMLP(
            input_dim=total_embed_dim,
            hidden_dims=mlp_hidden_dims,
            activation=mlp_activation,
            dropout=mlp_dropout,
            use_layer_norm=use_layer_norm,
            use_skip_connections=mlp_use_skip_connections,
        )
        self.output_dim = working_dim

    def forward(self, embeds: list[torch.Tensor]) -> torch.Tensor:
        """Process list of embeddings through backbone. Returns [Batch, output_dim]."""
        dnn_input = torch.cat(embeds, dim=1)

        if self.embed_ln is not None:
            dnn_input = self.embed_ln(dnn_input)

        if self.senet is not None:
            dnn_input = self.senet(embeds)

        if self.feature_gating is not None:
            dnn_input = self.feature_gating(dnn_input)

        if self.dcn is not None:
            dnn_input = self.dcn(dnn_input)

        return self.mlp(dnn_input)

    def forward_from_concat(self, dnn_input: torch.Tensor) -> torch.Tensor:
        """Process pre-concatenated tensor (e.g. when SENet not used)."""
        if self.embed_ln is not None:
            dnn_input = self.embed_ln(dnn_input)

        if self.feature_gating is not None:
            dnn_input = self.feature_gating(dnn_input)

        if self.dcn is not None:
            dnn_input = self.dcn(dnn_input)

        return self.mlp(dnn_input)
