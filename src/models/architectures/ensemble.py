"""Ensemble model for CTR prediction."""
import torch
import torch.nn as nn

from src.config.config import ConfigType
from src.models.architectures.base_model import GatedDCNModel


class EnsembleModel(nn.Module):
    """
    Ensemble of k identical GatedDCNModel instances.

    Each model is initialized with a different random seed for diversity.
    During inference, predictions are averaged across all models.
    During training, each model can be trained independently or jointly.

    Args:
        vocab_sizes: Dictionary mapping feature names to vocabulary sizes.
        feature_names: List of feature names in order.
        config: Configuration dictionary with model hyperparameters.
        k: Number of models in the ensemble (default from config['ensemble_k']).
        base_seed: Base seed for reproducibility. Model i uses seed = base_seed + i.
    """
    def __init__(
        self,
        vocab_sizes: dict[str, int],
        feature_names: list[str],
        config: ConfigType,
        k: int | None = None,
        base_seed: int | None = None
    ):
        super().__init__()

        # Get ensemble size from config or use provided k
        self.k: int = k if k is not None else config['ensemble_k']
        self.base_seed: int = base_seed if base_seed is not None else config['seed']
        self.ensemble_aggregation: str = config['ensemble_aggregation']

        # Create k models with different random initializations
        self.models = nn.ModuleList()
        for i in range(self.k):
            # Set unique seed for each model's initialization
            torch.manual_seed(self.base_seed + i)
            model = GatedDCNModel(vocab_sizes, feature_names, config)
            self.models.append(model)

        # Reset to base seed after initialization
        torch.manual_seed(self.base_seed)

    def forward(self, x, return_all_logits: bool = False):
        """
        Forward pass through all models in the ensemble.

        Args:
            x: Input tensor of shape [Batch, Num_Features]
            return_all_logits: If True, return logits from all models instead of aggregated.
                               Useful for training individual models.

        Returns:
            If return_all_logits=False: Aggregated logits [Batch, 1]
            If return_all_logits=True: Stacked logits [K, Batch, 1]
        """
        all_logits = []
        for model in self.models:
            logits = model(x)
            all_logits.append(logits)

        # Stack all logits: [K, Batch, 1]
        stacked_logits = torch.stack(all_logits, dim=0)

        if return_all_logits:
            return stacked_logits

        # Aggregate predictions
        if self.ensemble_aggregation == 'mean':
            # Average logits (equivalent to geometric mean of probabilities in log-space)
            return stacked_logits.mean(dim=0)
        elif self.ensemble_aggregation == 'median':
            return stacked_logits.median(dim=0).values
        else:
            raise ValueError(f"Unknown aggregation method: {self.ensemble_aggregation}")

    def forward_single(self, x, model_idx: int):
        """
        Forward pass through a single model in the ensemble.
        Useful for training individual models.

        Args:
            x: Input tensor of shape [Batch, Num_Features]
            model_idx: Index of the model to use (0 to k-1)

        Returns:
            Logits from the specified model [Batch, 1]
        """
        if model_idx < 0 or model_idx >= self.k:
            raise ValueError(f"model_idx must be in range [0, {self.k-1}], got {model_idx}")
        return self.models[model_idx](x)

    def get_model(self, idx: int) -> "GatedDCNModel":
        """Get a specific model from the ensemble."""
        model = self.models[idx]
        assert isinstance(model, GatedDCNModel)
        return model

    def num_models(self) -> int:
        """Return the number of models in the ensemble."""
        return self.k
