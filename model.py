import torch
import torch.nn as nn

from config import ConfigType

def get_senet_activation(name: str) -> nn.Module:
    """Get activation function for SENET excitation layer by name."""
    activations = {
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "relu": nn.ReLU(),
        "softmax": nn.Softmax(dim=-1),
    }
    if name not in activations:
        raise ValueError(f"Unknown SENET activation: {name}. Choose from {list(activations.keys())}")
    return activations[name]


class SENetLayer(nn.Module):
    """
    Squeeze-and-Excitation Network (SENET) from FiBiNET paper.
    
    Modified to support multiple squeeze functions (mean, max, etc.) that can be
    used together. The squeeze outputs are concatenated before the excitation network.
    
    Reference: FiBiNET: Combining Feature Importance and Bilinear feature Interaction
    for Click-Through Rate Prediction (RecSys 2019)
    
    Args:
        num_fields: Number of feature fields
        embedding_dim: Dimension of each field's embedding
        squeeze_funcs: List of squeeze functions to use. Options: 'mean', 'max'
        reduction_ratio: Reduction ratio for the excitation network bottleneck
        excitation_activation: Activation function for the excitation output
    """
    def __init__(
        self,
        num_fields: int,
        embedding_dim: int,
        squeeze_funcs: list[str] = ["mean"],
        reduction_ratio: int = 3,
        excitation_activation: str = "sigmoid"
    ):
        super().__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.squeeze_funcs = squeeze_funcs
        
        # Validate squeeze functions
        valid_squeeze_funcs = {"mean", "max", "min"}
        for func in squeeze_funcs:
            if func not in valid_squeeze_funcs:
                raise ValueError(f"Unknown squeeze function: {func}. Choose from {valid_squeeze_funcs}")
        
        # Number of squeeze outputs determines input to excitation network
        num_squeeze_outputs = len(squeeze_funcs)
        squeeze_output_dim = num_fields * num_squeeze_outputs
        
        # Excitation network (2-layer MLP with bottleneck)
        reduced_dim = max(1, num_fields // reduction_ratio)
        self.excitation = nn.Sequential(
            nn.Linear(squeeze_output_dim, reduced_dim, bias=False),
            nn.ReLU(),
            nn.Linear(reduced_dim, num_fields, bias=False),
            get_senet_activation(excitation_activation)
        )
    
    def forward(self, x):
        # x shape: [Batch, Num_Fields * Embed_Dim]
        batch_size = x.size(0)
        
        # Reshape to [Batch, Num_Fields, Embed_Dim]
        x_3d = x.view(batch_size, self.num_fields, self.embedding_dim)
        
        # Squeeze: Pool each field's embedding to a scalar
        squeeze_outputs = []
        for func in self.squeeze_funcs:
            if func == "mean":
                # Mean pooling: [Batch, Num_Fields]
                squeezed = x_3d.mean(dim=-1)
            elif func == "max":
                # Max pooling: [Batch, Num_Fields]
                squeezed = x_3d.max(dim=-1).values
            elif func == "min":
                # Min pooling: [Batch, Num_Fields]
                squeezed = x_3d.min(dim=-1).values
            else:
                # Should never reach here due to validation in __init__
                raise ValueError(f"Unknown squeeze function: {func}")
            squeeze_outputs.append(squeezed)
        
        # Concatenate squeeze outputs: [Batch, Num_Fields * Num_Squeeze_Funcs]
        squeeze_concat = torch.cat(squeeze_outputs, dim=-1)
        
        # Excitation: Learn field importance weights [Batch, Num_Fields]
        field_weights = self.excitation(squeeze_concat)
        
        # Expand weights to match embedding dimension: [Batch, Num_Fields, 1]
        field_weights = field_weights.unsqueeze(-1)
        
        # Re-weight: Scale each field's embedding by its importance
        x_reweighted = x_3d * field_weights
        
        # Flatten back to [Batch, Num_Fields * Embed_Dim]
        return x_reweighted.view(batch_size, -1)

class DCNv2(nn.Module):
    """
    Deep Cross Network V2 (Parallel).
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
            # x_next = x0 * (W * xi + b) + xi
            if self.low_rank is not None:
                # Low-rank: xi @ U @ V + b
                feature_crossing = torch.matmul(torch.matmul(xi, self.U[i]), self.V[i]) + self.b[i]
            else:
                # Full-rank: xi @ W + b
                feature_crossing = torch.matmul(xi, self.W[i]) + self.b[i]
            
            xi = x0 * feature_crossing + xi
            
            # Apply LayerNorm if enabled
            if self.use_layernorm:
                xi = self.layer_norms[i](xi)
            
        return xi

def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "leaky_relu": nn.LeakyReLU(0.1),
        "tanh": nn.Tanh(),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
    return activations[name]


class GatedDCNModel(nn.Module):
    """
    Gated DCN Model for CTR prediction.
    
    Args:
        vocab_sizes: Dictionary mapping feature names to vocabulary sizes.
        feature_names: List of feature names in order.
        config: Configuration dictionary with model hyperparameters.
    """
    def __init__(self, vocab_sizes: dict[str, int], feature_names: list[str], config: ConfigType):
        super().__init__()
        self.feature_names = feature_names
        
        # Extract config values
        embedding_dim = config['embedding_dim']
        use_dcn = config['use_dcn']
        dcn_num_layers = config['dcn_num_layers']
        dcn_use_layernorm = config['dcn_use_layernorm']
        dcn_low_rank = config['dcn_low_rank']
        use_senet = config['use_senet']
        senet_squeeze_funcs = config['senet_squeeze_funcs']
        senet_reduction_ratio = config['senet_reduction_ratio']
        senet_activation = config['senet_activation']
        mlp_hidden_dims = config['mlp_hidden_dims']
        mlp_dropout = config['mlp_dropout']
        use_layer_norm = config['use_layer_norm']
        mlp_activation = config['mlp_activation']
        
        self.use_layer_norm = use_layer_norm
        self.use_dcn = use_dcn
        self.use_senet = use_senet
        self.num_fields = len(feature_names)
        self.embedding_dim = embedding_dim

        # 1. Embedding Layer with better initialization
        self.embeddings = nn.ModuleDict()
        total_dim = 0
        for feat in feature_names:
            emb = nn.Embedding(vocab_sizes[feat], embedding_dim)
            # Xavier initialization for embeddings
            nn.init.xavier_uniform_(emb.weight)
            self.embeddings[feat] = emb
            total_dim += embedding_dim

        # Layer norm after embeddings
        if use_layer_norm:
            self.embed_ln = nn.LayerNorm(total_dim)

        # 2. SENET (Squeeze-and-Excitation) - Optional
        if use_senet:
            self.senet = SENetLayer(
                num_fields=self.num_fields,
                embedding_dim=embedding_dim,
                squeeze_funcs=senet_squeeze_funcs,
                reduction_ratio=senet_reduction_ratio,
                excitation_activation=senet_activation
            )

        # 3. DCNv2 - Optional (supports low-rank decomposition)
        if use_dcn:
            self.dcn = DCNv2(total_dim, num_layers=dcn_num_layers, use_layernorm=dcn_use_layernorm, low_rank=dcn_low_rank)


        # 4. Enhanced MLP with LayerNorm and configurable activation
        layers: list[nn.Module] = []
        input_dim = total_dim
        for i, hidden_dim in enumerate(mlp_hidden_dims):
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(get_activation(mlp_activation))
            layers.append(nn.Dropout(mlp_dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))

        self.mlp = nn.Sequential(*layers)

        # Initialize MLP layers properly
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x shape: [Batch, Num_Features]

        # Flatten inputs into a single dense vector
        embeds = []
        for i, feat in enumerate(self.feature_names):
            embeds.append(self.embeddings[feat](x[:, i]))

        # Concatenate: [Batch, Total_Dim]
        dnn_input = torch.cat(embeds, dim=1)

        # Apply layer norm to embeddings
        if self.use_layer_norm:
            dnn_input = self.embed_ln(dnn_input)

        # Apply SENET (Feature Importance Reweighting) - Optional
        if self.use_senet:
            dnn_input = self.senet(dnn_input)

        # Apply Cross Network (Interactions) - Optional
        if self.use_dcn:
            dnn_input = self.dcn(dnn_input)

        # Final Prediction (no sigmoid here - we'll use BCEWithLogitsLoss)
        logits = self.mlp(dnn_input)
        return logits  # Return raw logits for numerical stability
