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
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss

        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            loss = alpha_t * loss

        return loss.mean()
