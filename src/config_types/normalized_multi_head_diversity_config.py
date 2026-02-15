from typing import TypedDict
from .residual_mlp_config import ResidualMLPConfig


class NormalizedMultiHeadDiversityConfig(TypedDict):
    """
    Configuration for NormalizedMultiHeadDiversityModel.

    This model applies nGPT (Normalized Transformer) principles to the
    multi-head diversity architecture for CTR prediction. Key concepts:

    - All embeddings and weights are L2 normalized along embedding dimension
    - Hidden state updates use LERP: h ← Norm(h + α(h_block - h))
    - Learnable eigen learning rates (α) control block contributions
    - Scaling factors restore magnitude information after normalization
    - No LayerNorm/RMSNorm - replaced by unit norm normalization

    Reference: "nGPT: Normalized Transformer with Representation Learning
               on the Hypersphere" (Loshchilov et al., ICLR 2025)

    Attributes:
        backbone_type: Type of backbone model (e.g., 'gated_dcn').
        backbone_config: Dictionary configuration for the backbone.
        heads: List of configurations for each head.
        diversity_weight: Weight for the diversity loss term.
        feature_bagging_ratio: 1.0 = no bagging, < 1.0 = ratio of features per head.
        aggregation_method: 'mean' or 'gated'.
        gating_hidden_dim: Optional hidden dim for gated aggregation.

        # nGPT-specific parameters
        use_normalized_embeddings: Whether to normalize embedding vectors.
        use_normalized_weights: Whether to normalize weight matrices.
        alpha_init: Initial value for eigen learning rates (default: 0.05).
        alpha_scale: Scale factor for effective learning rate (default: 1/sqrt(d)).
        su_init: Initial value for MLP u scaling (default: 1.0).
        sv_init: Initial value for MLP v scaling (default: 1.0).
        use_lerp_updates: Whether to use LERP-style updates with eigen LR.
        normalize_before_head: Whether to normalize before each head.
    """

    backbone_type: str
    backbone_config: dict
    heads: list[ResidualMLPConfig]
    diversity_weight: float
    feature_bagging_ratio: float  # 1.0 = no bagging
    aggregation_method: str  # 'mean' | 'gated'
    gating_hidden_dim: int | None  # Optional hidden dim for gated aggregation

    # nGPT-specific parameters
    use_normalized_embeddings: bool  # Normalize embedding vectors to unit norm
    use_normalized_weights: bool  # Normalize weight matrices along embed dim
    alpha_init: float  # Initial eigen learning rate (typically ~1/num_layers or 0.05)
    alpha_scale: float | None  # Scale for effective learning rate in Adam (None = 1/sqrt(d))
    su_init: float  # Initial scaling for MLP u (default 1.0)
    sv_init: float  # Initial scaling for MLP v (default 1.0)
    use_lerp_updates: bool  # Use h ← Norm(h + α(h_block - h)) updates
    normalize_before_head: bool  # Normalize hidden state before each head
