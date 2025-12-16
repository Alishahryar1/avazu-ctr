"""Loss functions for CTR prediction models.

These are placed in the models package to avoid circular imports between
models and training modules.
"""
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Focuses on hard examples by down-weighting easy ones.
    """
    def __init__(self, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Optional class weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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


class KBCELoss(nn.Module):
    """
    K-way BCE Loss for multi-branch architectures.
    
    Generalized loss function that computes weighted BCE from k+1 predictions:
    - y_pred: Combined prediction (e.g., average of all branches)
    - y_branches: List of k individual branch predictions
    
    Loss = BCE(y_pred) + sum_i(weight_i * BCE(y_i))
    
    Weights are dynamically computed based on relative performance:
    - weight_i = max(0, loss_i - loss_combined)
    
    This encourages weaker branches to improve while not penalizing
    stronger branches.
    
    Works with any number of branches (FCN uses 2, Ensemble uses k models).
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
    
    def forward(
        self, 
        y_pred: torch.Tensor, 
        y_branches: list[torch.Tensor], 
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the k-way BCE loss.
        
        Args:
            y_pred: Combined logits [B, 1]
            y_branches: List of branch logits, each [B, 1]
            y_true: Ground truth labels [B, 1]
            
        Returns:
            Weighted combined loss
        """
        # Compute combined loss
        loss = self.bce(y_pred, y_true)
        
        # Compute individual branch losses and dynamic weights
        total_loss = loss
        for y_branch in y_branches:
            loss_i = self.bce(y_branch, y_true)
            # Penalize branches that perform worse than combined
            weight_i = (loss_i - loss).clamp(min=0)
            total_loss = total_loss + loss_i * weight_i
        
        return total_loss
