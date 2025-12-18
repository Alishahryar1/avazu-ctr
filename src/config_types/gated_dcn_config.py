from typing import TypedDict

class GatedDCNConfig(TypedDict):
    # Model Architecture - DCN/Attention
    use_dcn: bool
    dcn_num_layers: int
    dcn_use_layernorm: bool
    dcn_low_rank: int | None  # None = full-rank, int = low-rank dimension
    
    # Model Architecture - SENET
    use_senet: bool
    senet_squeeze_funcs: list[str]  # Options: 'mean', 'max' - can combine multiple
    senet_reduction_ratio: int
    senet_hidden_activation: str  # Bottleneck hidden layer activation
    senet_excitation_activation: str  # Final excitation output activation
    senet_num_groups: int  # SENet+: Number of groups for grouped squeeze (1 = no grouping)
    senet_reweight_mode: str  # SENet+: 'feature' (one weight per field) or 'element' (weight per element)
    senet_use_fuse: bool  # SENet+: Add original to reweighted (residual)
    senet_use_layer_norm: bool  # SENet+: Apply LayerNorm after fuse
    
    # Model Architecture - Feature Gating
    use_feature_gating: bool  # Alternative to SENET (mutually exclusive)
    feature_gating_activation: str  # Options: sigmoid, tanh, relu, etc.
    feature_gating_low_rank: int | None  # None = full-rank, int = low-rank dimension
    
    # Model Architecture - MLP
    mlp_hidden_dims: list[int]
    mlp_activation: str
    mlp_use_skip_connections: bool  # Add residual/skip connections to MLP layers
    use_layer_norm: bool
    
    # Regularization
    mlp_dropout: float
    focal_loss_gamma: float
    label_smoothing: float
