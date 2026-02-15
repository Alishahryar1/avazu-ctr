from typing import TypedDict


class ResidualMLPConfig(TypedDict):
    """Configuration for a ResidualMLP layer."""

    hidden_dims: list[int]
    activation: str
    dropout: float
    use_layer_norm: bool
    use_skip_connections: bool
