from typing import TypedDict, List

class ResidualMLPConfig(TypedDict):
    """Configuration for a ResidualMLP layer."""
    hidden_dims: List[int]
    activation: str
    dropout: float
    use_layer_norm: bool
    use_skip_connections: bool
