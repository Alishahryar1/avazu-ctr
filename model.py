import torch
import torch.nn as nn

class FeatureGatingLayer(nn.Module):
    """
    The 'Fast' implementation of the Gated Attention paper.
    Instead of O(N^2) Self-Attention, we use O(N) Element-wise Gating.
    It learns to suppress noise (sparsity) and adds non-linearity.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.gate_linear = nn.Linear(input_dim, input_dim)
        # self.norm = nn.LayerNorm(input_dim) # Helps stability

    def forward(self, x):
        # x shape: [Batch, Num_Features * Embed_Dim]
        
        # Calculate Gate Score (Sigmoid forces 0-1 range)
        gate_score = torch.sigmoid(self.gate_linear(x))
        
        # Apply Gate
        return x * gate_score

class DCNv2(nn.Module):
    """
    Deep Cross Network V2 (Parallel).
    Explicitly captures high-order feature interactions.
    """
    def __init__(self, input_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        
        # Parameters for Cross Layers
        self.W = nn.ParameterList([nn.Parameter(torch.randn(input_dim, input_dim)) for _ in range(num_layers)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers)])
        
        # Init
        for w in self.W:
            nn.init.xavier_uniform_(w)

    def forward(self, x):
        # x: [Batch, Input_Dim]
        x0 = x
        xi = x
        
        for i in range(self.num_layers):
            # x_next = x0 * (W * xi + b) + xi
            # We use linear layer logic for W * xi + b
            feature_crossing = torch.matmul(xi, self.W[i]) + self.b[i]
            xi = x0 * feature_crossing + xi
            
        return xi

class GatedDCNModel(nn.Module):
    def __init__(self, vocab_sizes, embedding_dim, feature_names, dcn_num_layers=2, mlp_hidden_dims=[256, 128], mlp_dropout=0.2):
        super().__init__()
        self.feature_names = feature_names
        
        # 1. Embedding Layer
        self.embeddings = nn.ModuleDict()
        total_dim = 0
        for feat in feature_names:
            self.embeddings[feat] = nn.Embedding(vocab_sizes[feat], embedding_dim)
            total_dim += embedding_dim
            
        # 2. Feature Gating (Replaces SENet)
        self.gating = FeatureGatingLayer(total_dim)
        
        # 3. DCNv2
        self.dcn = DCNv2(total_dim, num_layers=dcn_num_layers)
        
        # 4. Final MLP
        layers = []
        input_dim = total_dim
        for hidden_dim in mlp_hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(mlp_dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: [Batch, Num_Features]
        
        # Flatten inputs into a single dense vector
        embeds = []
        for i, feat in enumerate(self.feature_names):
            embeds.append(self.embeddings[feat](x[:, i]))
        
        # Concatenate: [Batch, Total_Dim]
        dnn_input = torch.cat(embeds, dim=1)
        
        # Apply Gating (Sparsity & Reweighting)
        gated_input = self.gating(dnn_input)
        
        # Apply Cross Network (Interactions)
        cross_out = self.dcn(gated_input)
        
        # Final Prediction
        logits = self.mlp(cross_out)
        return torch.sigmoid(logits)
