import torch
import torch.nn as nn

class GatingNetwork(nn.Module):
    """
    Learns to weight the experts based on input features.
    Output: [Batch, Num_Experts] (Sum to 1 via Softmax)
    """
    def __init__(
        self, 
        vocab_sizes: dict[str, int], 
        feature_names: list[str], 
        num_experts: int,
        embedding_dim: int = 4  # Keep this small for efficiency
    ):
        super().__init__()
        self.feature_names = feature_names
        
        # 1. Lightweight embeddings specific to the Gate
        # We don't need huge embeddings here, just enough to identify context
        self.embeddings = nn.ModuleDict({
            feat: nn.Embedding(size, embedding_dim) 
            for feat, size in vocab_sizes.items()
        })
        
        # 2. Prediction Layer
        # Input size = num_features * embedding_dim
        total_input_dim = len(feature_names) * embedding_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_experts),
            nn.Softmax(dim=1)  # Ensures weights sum to 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Num_Features]
        
        # 1. Look up embeddings
        embedded = []
        for i, feat_name in enumerate(self.feature_names):
            # Get column i
            col = x[:, i].long()
            emb = self.embeddings[feat_name](col)
            embedded.append(emb)
            
        # 2. Flatten: [Batch, Num_Features * Emb_Dim]
        concat = torch.cat(embedded, dim=1)
        
        # 3. Predict Weights: [Batch, Num_Experts]
        weights = self.mlp(concat)
        return weights