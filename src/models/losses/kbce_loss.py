import torch
import torch.nn as nn


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

    Works with any number of branches (Ensemble uses k models).
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(
        self, y_pred: torch.Tensor, y_branches: list[torch.Tensor], y_true: torch.Tensor
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
            gap = (loss_i - loss).detach()
            # Penalize branches that perform worse than combined
            weight_i = 0.1 + gap.clamp(min=0)
            total_loss = total_loss + loss_i * weight_i

        return total_loss
