import torch
import torch.nn as nn

from config import ConfigType

def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "leaky_relu": nn.LeakyReLU(0.1),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "softmax": nn.Softmax(dim=-1),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
    return activations[name]


def compute_embedding_dim(vocab_size: int, config: ConfigType) -> int:
    """
    Compute optimal embedding dimension based on vocabulary size.
    
    Uses cardinality-based rules from config:
    - Smaller vocabularies get smaller embedding dimensions
    - Larger vocabularies get larger dimensions (more capacity needed)
    
    Args:
        vocab_size: Number of unique values for this feature
        config: Configuration dict containing embedding_dim_rules
        
    Returns:
        Optimal embedding dimension for this vocabulary size
    """
    if not config.get('use_variable_embeddings', False):
        return config['embedding_dim']
    
    # Check cardinality rules (sorted ascending by max_vocab_size)
    for max_vocab, embed_dim in config['embedding_dim_rules']:
        if vocab_size <= max_vocab:
            return embed_dim
    
    # Default to base embedding_dim for very large vocabularies
    return config['embedding_dim']


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
        squeeze_funcs: List of squeeze functions to use. Options: 'mean', 'max', 'min'
        reduction_ratio: Reduction ratio for the excitation network bottleneck
        excitation_activation: Activation function for the excitation output
    """
    # Hashmap of squeeze operations - single source of truth
    SQUEEZE_OPS = {
        "mean": lambda t: t.mean(dim=-1),
        "max": lambda t: t.max(dim=-1).values,
        "min": lambda t: t.min(dim=-1).values,
    }
    
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
        
        # Validate squeeze functions using the class-level hashmap
        for func in squeeze_funcs:
            if func not in self.SQUEEZE_OPS:
                raise ValueError(f"Unknown squeeze function: {func}. Choose from {list(self.SQUEEZE_OPS.keys())}")
        
        # Number of squeeze outputs determines input to excitation network
        num_squeeze_outputs = len(squeeze_funcs)
        squeeze_output_dim = num_fields * num_squeeze_outputs
        
        # Excitation network (2-layer MLP with bottleneck)
        reduced_dim = max(1, num_fields // reduction_ratio)
        self.excitation = nn.Sequential(
            nn.Linear(squeeze_output_dim, reduced_dim, bias=False),
            nn.ReLU(),
            nn.Linear(reduced_dim, num_fields, bias=False),
            get_activation(excitation_activation)
        )
    
    def forward(self, x):
        # x shape: [Batch, Num_Fields * Embed_Dim]
        batch_size = x.size(0)
        
        # Reshape to [Batch, Num_Fields, Embed_Dim]
        x_3d = x.view(batch_size, self.num_fields, self.embedding_dim)
        
        # Squeeze: Pool each field's embedding to a scalar using class-level hashmap
        squeeze_outputs = []
        for func in self.squeeze_funcs:
            squeezed = self.SQUEEZE_OPS[func](x_3d)
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


class FeatureGatingLayer(nn.Module):
    """
    The 'Fast' implementation of the Gated Attention paper.
    Instead of O(N^2) Self-Attention, we use O(N) Element-wise Gating.
    It learns to suppress noise (sparsity) and adds non-linearity.
    
    Supports low-rank decomposition: W = U @ V where U is (input_dim, rank)
    and V is (rank, input_dim). This reduces parameters from O(d^2) to O(2*d*r).
    """
    def __init__(self, input_dim, gating_activation: str = "sigmoid", low_rank: int | None = None):
        super().__init__()
        self.input_dim = input_dim
        self.low_rank = low_rank
        self.activation = get_activation(gating_activation)
        
        if low_rank is not None:
            # Low-rank decomposition: W = U @ V
            self.U = nn.Parameter(torch.randn(input_dim, low_rank))
            self.V = nn.Parameter(torch.randn(low_rank, input_dim))
            self.bias = nn.Parameter(torch.zeros(input_dim))
            # Xavier initialization
            nn.init.xavier_uniform_(self.U)
            nn.init.xavier_uniform_(self.V)
        else:
            # Full-rank linear layer
            self.gate_linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x shape: [Batch, Num_Features * Embed_Dim]
        
        # Calculate Gate Score
        if self.low_rank is not None:
            # Low-rank: x @ U @ V + bias
            gate_logits = torch.matmul(torch.matmul(x, self.U), self.V) + self.bias
        else:
            # Full-rank linear
            gate_logits = self.gate_linear(x)
        
        gate_score = self.activation(gate_logits)
        
        # Apply Gate
        return x * gate_score


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


class ResidualMLP(nn.Module):
    """
    MLP with optional skip connections (residual connections).
    
    When skip connections are enabled, each layer's output is added to its input,
    similar to ResNet. If dimensions differ between layers, a linear projection
    is used to match dimensions for the skip connection.
    
    Args:
        input_dim: Input dimension to the MLP
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension (typically 1 for CTR prediction)
        activation: Activation function name
        dropout: Dropout rate (0 = no dropout)
        use_layer_norm: Whether to apply layer normalization
        use_skip_connections: Whether to use residual/skip connections
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int = 1,
        activation: str = "relu",
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        use_skip_connections: bool = False
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
            
            # Optional layer norm (use Identity as placeholder if not used)
            if self.layer_norms is not None:
                self.layer_norms.append(nn.LayerNorm(dims[i + 1]))
            
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
        
        # Final output layer (no skip connection, no activation)
        self.output_layer = nn.Linear(hidden_dims[-1] if hidden_dims else input_dim, output_dim)
        
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
            
            # Forward through layer
            x = layer(x)
            
            if self.layer_norms is not None:
                x = self.layer_norms[i](x)
            
            x = self.activations[i](x)
            
            if self.dropouts is not None:
                x = self.dropouts[i](x)
            
            # Skip connection (projection is Identity when dims match)
            if self.use_skip_connections:
                x = x + self.projections[i](identity)
        
        # Final output layer
        return self.output_layer(x)


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
        use_feature_gating = config['use_feature_gating']
        feature_gating_activation = config['feature_gating_activation']
        mlp_hidden_dims = config['mlp_hidden_dims']
        mlp_dropout = config['mlp_dropout']
        use_layer_norm = config['use_layer_norm']
        mlp_activation = config['mlp_activation']
        
        # Validate mutual exclusivity
        if use_senet and use_feature_gating:
            raise ValueError(
                "Cannot enable both SENET and Feature Gating. "
                "Set either 'use_senet' or 'use_feature_gating' to False."
            )
        
        self.use_layer_norm = use_layer_norm
        self.use_dcn = use_dcn
        self.use_senet = use_senet
        self.use_feature_gating = use_feature_gating
        self.num_fields = len(feature_names)
        self.base_embedding_dim = embedding_dim  # Base/fallback dimension
        
        # Track per-feature embedding dimensions for variable embeddings
        use_variable_embeddings = config.get('use_variable_embeddings', False)
        feature_overrides = config.get('feature_embedding_overrides', {})
        projection_dim = config.get('embedding_projection_dim', None)

        # 1. Embedding Layer with variable dimensions per feature
        self.embeddings = nn.ModuleDict()
        self.feature_dims: dict[str, int] = {}  # Track dimension per feature
        total_embed_dim = 0
        
        for feat in feature_names:
            # Check for manual override first
            if feat in feature_overrides and 'embedding_dim' in feature_overrides[feat]:
                feat_dim = feature_overrides[feat]['embedding_dim']
            elif use_variable_embeddings:
                # Compute dimension based on cardinality
                feat_dim = compute_embedding_dim(vocab_sizes[feat], config)
            else:
                feat_dim = embedding_dim
            
            self.feature_dims[feat] = feat_dim
            emb = nn.Embedding(vocab_sizes[feat], feat_dim)
            # Xavier initialization for embeddings
            nn.init.xavier_uniform_(emb.weight)
            self.embeddings[feat] = emb
            total_embed_dim += feat_dim
        
        # Store dimensions for later use
        self.total_embed_dim = total_embed_dim
        self.use_projection = projection_dim is not None
        
        # 2. Optional Projection Layer to unify dimensions
        if self.use_projection and projection_dim is not None:
            self.projection = nn.Linear(total_embed_dim, projection_dim)
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
            working_dim = projection_dim
            # For SENET, we need uniform embedding dim after projection
            self.embedding_dim = projection_dim // self.num_fields
        else:
            working_dim = total_embed_dim
            # For SENET compatibility: only works with uniform embeddings
            # When variable embeddings without projection, SENET is disabled
            self.embedding_dim = embedding_dim if not use_variable_embeddings else embedding_dim

        # Layer norm after embeddings (before or after projection)
        if use_layer_norm:
            self.embed_ln = nn.LayerNorm(working_dim)

        # 3. SENET (Squeeze-and-Excitation) - Optional
        # Note: SENET requires uniform embedding dimensions per field
        if use_senet:
            if use_variable_embeddings and not self.use_projection:
                raise ValueError(
                    "SENET requires uniform embedding dimensions. "
                    "Either disable 'use_variable_embeddings', enable 'embedding_projection_dim', "
                    "or disable 'use_senet'."
                )
            senet_embed_dim = self.embedding_dim if self.embedding_dim else embedding_dim
            self.senet = SENetLayer(
                num_fields=self.num_fields,
                embedding_dim=senet_embed_dim,
                squeeze_funcs=senet_squeeze_funcs,
                reduction_ratio=senet_reduction_ratio,
                excitation_activation=senet_activation
            )

        # 2b. Feature Gating Layer - Optional (alternative to SENET)
        if use_feature_gating:
            feature_gating_low_rank = config['feature_gating_low_rank']
            self.feature_gating = FeatureGatingLayer(
                input_dim=working_dim,
                gating_activation=feature_gating_activation,
                low_rank=feature_gating_low_rank
            )

        # 4. DCNv2 - Optional (supports low-rank decomposition)
        if use_dcn:
            self.dcn = DCNv2(working_dim, num_layers=dcn_num_layers, use_layernorm=dcn_use_layernorm, low_rank=dcn_low_rank)


        # 5. Enhanced MLP with LayerNorm, configurable activation, and optional skip connections
        mlp_use_skip_connections = config['mlp_use_skip_connections']
        self.mlp = ResidualMLP(
            input_dim=working_dim,
            hidden_dims=mlp_hidden_dims,
            output_dim=1,
            activation=mlp_activation,
            dropout=mlp_dropout,
            use_layer_norm=use_layer_norm,
            use_skip_connections=mlp_use_skip_connections
        )

    def forward(self, x):
        # x shape: [Batch, Num_Features]

        # Flatten inputs into a single dense vector
        embeds = []
        for i, feat in enumerate(self.feature_names):
            embeds.append(self.embeddings[feat](x[:, i]))

        # Concatenate: [Batch, Total_Embed_Dim]
        dnn_input = torch.cat(embeds, dim=1)
        
        # Apply optional projection to unify dimensions
        if self.use_projection:
            dnn_input = self.projection(dnn_input)

        # Apply layer norm to embeddings
        if self.use_layer_norm:
            dnn_input = self.embed_ln(dnn_input)

        # Apply SENET (Feature Importance Reweighting) - Optional
        if self.use_senet:
            dnn_input = self.senet(dnn_input)

        # Apply Feature Gating - Optional (alternative to SENET)
        if self.use_feature_gating:
            dnn_input = self.feature_gating(dnn_input)

        # Apply Cross Network (Interactions) - Optional
        if self.use_dcn:
            dnn_input = self.dcn(dnn_input)

        # Final Prediction (no sigmoid here - we'll use BCEWithLogitsLoss)
        logits = self.mlp(dnn_input)
        return logits  # Return raw logits for numerical stability
