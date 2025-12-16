"""Loss functions for CTR prediction."""
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Focuses on hard examples by down-weighting easy ones.
    """
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Optional class weights

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss

        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            loss = alpha_t * loss

        return loss.mean()


class TriBCELoss(nn.Module):
    """
    Triple BCE Loss for FCNv2 dual-path architecture.
    
    Computes weighted loss from three predictions:
    - y_pred: Combined prediction (average of both paths)
    - y_d: E2L path prediction (Exponential-to-Linear)
    - y_s: L2E path prediction (Linear-to-Exponential)
    
    Loss = BCE(y_pred) + weight_d * BCE(y_d) + weight_s * BCE(y_s)
    
    Weights are dynamically computed based on relative performance:
    - weight_d = max(0, loss_d - loss)  
    - weight_s = max(0, loss_s - loss)
    
    This encourages the weaker branch to improve while not penalizing
    the stronger branch.
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
    
    def forward(
        self, 
        y_pred: torch.Tensor, 
        y_d: torch.Tensor, 
        y_s: torch.Tensor, 
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the triple BCE loss.
        
        Args:
            y_pred: Combined logits [B, 1]
            y_d: E2L path logits [B, 1]
            y_s: L2E path logits [B, 1]
            y_true: Ground truth labels [B, 1]
            
        Returns:
            Weighted combined loss
        """
        # Compute individual losses
        loss = self.bce(y_pred, y_true)
        loss_d = self.bce(y_d, y_true)
        loss_s = self.bce(y_s, y_true)
        
        # Compute dynamic weights (penalize branches that perform worse than combined)
        weight_d = (loss_d - loss).clamp(min=0)
        weight_s = (loss_s - loss).clamp(min=0)
        
        # Combined weighted loss
        total_loss = loss + loss_d * weight_d + loss_s * weight_s
        
        return total_loss
