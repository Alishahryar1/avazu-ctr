"""Base model interface for CTR prediction models.

This module defines the abstract base class that all CTR models must implement.
Using this interface allows trainer and inference code to work with any model
without model-specific branching.
"""

from abc import ABC, abstractmethod
from typing import TypedDict

import torch
import torch.nn as nn

from src.models.types import ModelOutput


class BaseCTRModel(ABC, nn.Module):
    """
    Abstract base class for all CTR prediction models.

    All models must implement:
    - forward(): Returns standardized ModelOutput dict
    - compute_loss(): Handles model-specific loss computation (loss internalized)
    - get_predictions(): Returns probabilities for inference
    - model_name(): Class method returning the model's registered name
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Forward pass returning standardized output.

        Args:
            x: Input features [B, Num_Features]

        Returns:
            ModelOutput dict with 'logits' and optional 'aux_logits'
        """
        pass

    @abstractmethod
    def compute_loss(self, output: ModelOutput, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for this model's output using internal loss function.

        Args:
            output: Output from forward()
            y_true: Ground truth labels [B, 1]

        Returns:
            Scalar loss tensor
        """
        pass

    def get_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities for inference.

        Args:
            x: Input features [B, Num_Features]

        Returns:
            Probabilities [B, 1]
        """
        output = self.forward(x)
        return torch.sigmoid(output["logits"])

    @classmethod
    @abstractmethod
    def model_name(cls) -> str:
        """Return the model's registered name for the model registry."""
        pass
