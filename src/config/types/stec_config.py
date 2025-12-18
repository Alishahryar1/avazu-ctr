from typing import TypedDict

class STECConfig(TypedDict):
    stec_num_layers: int
    stec_num_heads: int
    stec_hidden_dim: int | None  # Defaults to 4 * embed_dim
    stec_dropout: float
    stec_use_ffn: bool
    stec_mlp_hidden_dims: list[int]
