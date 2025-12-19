import torch
import torch.nn as nn
import torch.nn.functional as F

class DiversityBCELoss(nn.Module):
    """
    Loss function using Negative Correlation Learning (NCL) for multi-head diversity.
    
    NCL encourages heads to make different errors by penalizing correlation between
    each head's prediction and the ensemble mean. This is bounded and stable unlike
    raw variance maximization.
    
    Loss = Mean(BCE(head_i, y)) - diversity_weight * NCL_term
    NCL_term = Mean((prob_i - ensemble_prob)^2)
    """
    def __init__(self, diversity_weight: float = 0.1):
        super().__init__()
        self.diversity_weight = diversity_weight

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
        num_heads = aux_logits.shape[0]
        
        # Convert to probabilities for NCL computation (bounded 0-1)
        probs = torch.sigmoid(aux_logits)  # [K, B, 1]
        ensemble_prob = probs.mean(dim=0)   # [B, 1]
        
        # 1. BCE loss for each head
        total_bce = torch.tensor(0.0, device=aux_logits.device, dtype=aux_logits.dtype)
        for i in range(num_heads):
            total_bce = total_bce + F.binary_cross_entropy_with_logits(aux_logits[i], y_true)
        avg_bce = total_bce / num_heads
        
        # 2. NCL diversity term
        if num_heads > 1:
            # Each head's deviation from ensemble mean
            deviations = probs - ensemble_prob  # [K, B, 1]
            
            # NCL: squared deviations encourage negative correlation
            # This is bounded since probs are in [0, 1], so deviations in [-1, 1]
            ncl_term = (deviations ** 2).mean()
            
            # Subtract to encourage diversity (maximize disagreement)
            loss = avg_bce - self.diversity_weight * ncl_term
            
            # Safety clamp to prevent numerical issues (loss should stay positive)
            loss = torch.clamp(loss, min=1e-7)
        else:
            loss = avg_bce
            
        return loss
