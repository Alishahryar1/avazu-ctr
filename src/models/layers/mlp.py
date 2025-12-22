"""Residual MLP layer with skip connections."""

import torch.nn as nn

from src.models.utils import get_activation


class ResidualMLP(nn.Module):
    """
    MLP with optional skip connections (residual connections).

    When skip connections are enabled, each layer's output is added to its input,
    similar to ResNet. If dimensions differ between layers, a linear projection
    is used to match dimensions for the skip connection.

    Args:
        input_dim: Input dimension to the MLP
        hidden_dims: List of hidden layer dimensions (output will be last hidden dim)
        activation: Activation function name
        dropout: Dropout rate (0 = no dropout)
        use_layer_norm: Whether to apply layer normalization
        use_skip_connections: Whether to use residual/skip connections
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activation: str = "relu",
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        use_skip_connections: bool = False,
    ):
        super().__init__()
        self.use_skip_connections = use_skip_connections

        # Build layers
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList() if dropout > 0 else None
        self.projections = nn.ModuleList()  # For dimension matching in skip connections

        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            # Main linear layer
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

            # Optional layer norm for pre-norm (applied to input, so use input dim)
            if self.layer_norms is not None:
                self.layer_norms.append(nn.LayerNorm(dims[i]))

            # Activation
            self.activations.append(get_activation(activation))

            # Optional dropout (use Identity as placeholder if not used)
            if self.dropouts is not None:
                self.dropouts.append(nn.Dropout(dropout))

            # Projection for skip connection when dimensions differ
            # Use Identity when no projection needed (same dimensions or no skip)
            if use_skip_connections and dims[i] != dims[i + 1]:
                self.projections.append(nn.Linear(dims[i], dims[i + 1], bias=False))
            else:
                self.projections.append(nn.Identity())

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            identity = x

            # Pre-norm: Apply LayerNorm before the linear layer
            if self.layer_norms is not None:
                x = self.layer_norms[i](x)

            # Forward through layer
            x = layer(x)

            x = self.activations[i](x)

            if self.dropouts is not None:
                x = self.dropouts[i](x)

            # Skip connection (projection is Identity when dims match)
            if self.use_skip_connections:
                x = x + self.projections[i](identity)

        return x
