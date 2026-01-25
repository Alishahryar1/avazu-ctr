from typing import Union
from .gated_dcn_config import GatedDCNConfig
from .stec_config import STECConfig
from .ensemble_config import EnsembleConfig
from .multi_head_diversity_config import MultiHeadDiversityConfig
from .normalized_multi_head_diversity_config import NormalizedMultiHeadDiversityConfig

# Type alias for any model configuration
ModelConfig = Union[
    GatedDCNConfig,
    EnsembleConfig,
    STECConfig,
    MultiHeadDiversityConfig,
    NormalizedMultiHeadDiversityConfig,
]
