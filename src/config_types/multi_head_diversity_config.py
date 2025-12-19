from typing import TypedDict, List
from .residual_mlp_config import ResidualMLPConfig

class MultiHeadDiversityConfig(TypedDict):
    """
    Configuration for MultiHeadDiversityModel.
    
    Attributes:
        backbone_type: Type of backbone model (e.g., 'gated_dcn').
        backbone_config: dictionary configuration for the backbone.
                         Note: Backbone's internal MLP will be stripped.
        heads: List of configurations for each head.
        diversity_weight: Weight for the diversity loss term.
    """
    backbone_type: str
    backbone_config: dict
    heads: List[ResidualMLPConfig]
    diversity_weight: float
    feature_bagging_ratio: float # 1.0 = no bagging, < 1.0 = ratio of features to keep per head
