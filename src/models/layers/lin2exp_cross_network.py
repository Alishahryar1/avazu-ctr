"""Linear-to-Exponential Cross Network for FCNv2."""
import torch
import torch.nn as nn


class Linear2ExponentialCrossNetwork(nn.Module):
    """
    Linear-to-Exponential Cross Network (L2ECN) for FCNv2.
    
    Performs linear feature crossing first (with initial features),
    then exponential feature crossing (self-interaction).
    
    Args:
        input_dim: Input dimension.
        exp_num_layers: Number of exponential cross layers.
        lin_num_layers: Number of linear cross layers.
        batch_norm: Whether to use batch normalization.
        layer_norm: Whether to use layer normalization.
        net_dropout: Dropout rate.
        num_heads: Number of heads (for batch norm dimension).
    """
    def __init__(
        self,
        input_dim: int,
        exp_num_layers: int = 3,
        lin_num_layers: int = 3,
        batch_norm: bool = True,
        layer_norm: bool = False,
        net_dropout: float = 0.1,
        num_heads: int = 1
    ):
        super().__init__()
        self.exp_num_layers = exp_num_layers
        self.lin_num_layers = lin_num_layers
        self.num_heads = num_heads
        
        total_layers = exp_num_layers + lin_num_layers
        
        # Layer components
        self.layer_norm = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.w = nn.ModuleList()
        self.b = nn.ParameterList()
        self.gamma = nn.ParameterList()
        self.beta = nn.ParameterList()
        
        for i in range(total_layers):
            # Linear projection to half dimension
            self.w.append(nn.Linear(input_dim, input_dim // 2, bias=False))
            
            # Normalization
            if layer_norm:
                self.layer_norm.append(nn.LayerNorm(input_dim // 2))
            else:
                self.gamma.append(nn.Parameter(torch.ones(input_dim // 2)))
                self.beta.append(nn.Parameter(torch.zeros(input_dim // 2)))
            
            # Bias - initialize to zeros for stability
            self.b.append(nn.Parameter(torch.zeros(input_dim)))
            
            # Batch norm (applied on heads dimension)
            if batch_norm:
                self.batch_norm.append(nn.BatchNorm1d(num_heads))
            
            # Dropout
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout))
        
        # Output projection
        self.fc = nn.Linear(input_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for w in self.w:
            nn.init.xavier_uniform_(w.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def _cross_layer(
        self, 
        x: torch.Tensor, 
        x_anchor: torch.Tensor, 
        i: int
    ) -> torch.Tensor:
        """
        Single cross layer operation.
        
        Args:
            x: Current features [B, H, D] or [B, D]
            x_anchor: Anchor features for crossing
            i: Layer index
            
        Returns:
            Crossed features
        """
        # Linear projection: (B, H, D) -> (B, H, D/2)
        H_1 = self.w[i](x)
        
        # Apply normalization
        if len(self.layer_norm) > i:
            H_2 = self.layer_norm[i](H_1)
        else:
            H_2 = self.gamma[i] * H_1 + self.beta[i]
        
        # Concatenate: (B, H, D/2) + (B, H, D/2) -> (B, H, D)
        H = torch.cat([H_1, H_2], dim=-1)
        
        # Batch norm (on heads dimension)
        if len(self.batch_norm) > i:
            H = self.batch_norm[i](H)
        
        # Cross operation with residual: x_anchor * (H + b) + x
        x = x_anchor * (H + self.b[i]) + x
        
        # Dropout
        if len(self.dropout) > i:
            x = self.dropout[i](x)
        
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [B, H, D] or [B, D]
            
        Returns:
            Logits [B, H, 1] or [B, 1]
        """
        x1 = x  # Store for linear layers
        
        # Linear layers first (interaction with initial features)
        for i in range(self.lin_num_layers):
            x = self._cross_layer(x, x1, i)
        
        # Exponential layers (self-interaction)
        for i in range(self.exp_num_layers):
            layer_idx = i + self.lin_num_layers
            x = self._cross_layer(x, x, layer_idx)
        
        logit = self.fc(x)
        return logit
