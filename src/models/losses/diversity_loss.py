import torch
import torch.nn as nn
import torch.nn.functional as F

class DiversityBCELoss(nn.Module):
    """
    Loss function that encourages diversity among multiple heads.
    
    Combines binary cross entropy for each head with a diversity penalty (variance).
    Loss = Mean(BCE(head_i, y)) - diversity_weight * Mean(Variance(heads))
    """
    def __init__(self, diversity_weight: float = 0.1):
        super().__init__()
        self.diversity_weight = diversity_weight
        # We use BCEWithLogitsLoss per head
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, aux_logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            aux_logits: Stacked logits from heads [Num_Heads, Batch, 1]
            y_true: Ground truth labels [Batch, 1]
        
        Returns:
            Scalar loss
        """
        # aux_logits shape: [K, B, 1]
        # y_true shape: [B, 1]
        
        # 1. Compute BCE for each head
        # We can expand y_true to match aux_logits or iterate. 
        # Iterating is simple.
        total_bce = torch.tensor(0.0, device=aux_logits.device)
        num_heads = aux_logits.shape[0]
        
        for i in range(num_heads):
            total_bce += self.bce(aux_logits[i], y_true)
            
        avg_bce = total_bce / num_heads
        
        # 2. Compute Variance across heads for each sample
        # Variance of logits across the K dimension
        # var(dim=0) returns variance across heads [B, 1]
        if num_heads > 1:
            # We want to maximize variance, so we subtract it
            variance = torch.var(aux_logits, dim=0).mean()
            loss = avg_bce - self.diversity_weight * variance
        else:
            loss = avg_bce
            
        return loss
