"""Deep Cross Network V2 (DCNv2) layer."""
import torch
import torch.nn as nn


class DCNv2(nn.Module):
    """
    Deep Cross Network V2 (Stacked).
    Explicitly captures high-order feature interactions.

    Supports low-rank decomposition: W = U @ V where U is (input_dim, rank)
    and V is (rank, input_dim). This reduces parameters from O(d^2) to O(2*d*r).
    """
    def __init__(self, input_dim, num_layers=2, use_layernorm=False, low_rank=None):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.use_layernorm = use_layernorm
        self.low_rank = low_rank

        # Parameters for Cross Layers
        if low_rank is not None:
            # Low-rank decomposition: W = U @ V
            self.U = nn.ParameterList([nn.Parameter(torch.randn(input_dim, low_rank)) for _ in range(num_layers)])
            self.V = nn.ParameterList([nn.Parameter(torch.randn(low_rank, input_dim)) for _ in range(num_layers)])
        else:
            # Full-rank weight matrix
            self.W = nn.ParameterList([nn.Parameter(torch.randn(input_dim, input_dim)) for _ in range(num_layers)])

        self.b = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)])

        # Optional LayerNorm for stability
        if use_layernorm:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_layers)])

        # Init
        self._init_weights()

    def _init_weights(self):
        if self.low_rank is not None:
            for u, v in zip(self.U, self.V):
                nn.init.xavier_uniform_(u)
                nn.init.xavier_uniform_(v)
        else:
            for w in self.W:
                nn.init.xavier_uniform_(w)

    def forward(self, x):
        # x: [Batch, Input_Dim]
        x0 = x
        xi = x

        for i in range(self.num_layers):
            # Pre-norm: Apply LayerNorm before the cross operation
            xi_normed = self.layer_norms[i](xi) if self.use_layernorm else xi

            # x_next = x0 * (W * xi_normed + b) + xi
            if self.low_rank is not None:
                # Low-rank: xi_normed @ U @ V + b
                feature_crossing = torch.matmul(torch.matmul(xi_normed, self.U[i]), self.V[i]) + self.b[i]
            else:
                # Full-rank: xi_normed @ W + b
                feature_crossing = torch.matmul(xi_normed, self.W[i]) + self.b[i]

            xi = x0 * feature_crossing + xi

        return xi
