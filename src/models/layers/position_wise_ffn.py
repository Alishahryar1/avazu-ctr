import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionWiseFFN(nn.Module):
    """
    Position-wise Feed-Forward Network as used in Transformers.
    
    FFN(x) = max(0, x @ W1 + b1) @ W2 + b2
    
    Args:
        embed_dim: Input/output dimension
        hidden_dim: Hidden layer dimension (typically 4x embed_dim)
        dropout: Dropout rate
    """
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply FFN: [B, F, D] -> [B, F, D]"""
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
